//! Part 5: exit handling.
//!
//! The AMD counterpart of `hv::exit_decision` plus the carry-out half of
//! `hv::vmexit::handler`. It does the same acts Intel does, from the SVM
//! exit code instead of the VMX exit reason:
//!
//! - **decode** ([`decode`]): read `exit_code` and the `exit_info` fields and
//!   turn them into a neutral [`Exit`], reading the guest's operands (CPUID
//!   leaf, MSR number, OUT value) from its registers.
//! - **act**: run the real instruction for the guest and write the result
//!   back ([`apply_cpuid`], [`apply_rdmsr`]), then advance RIP past it
//!   ([`advance_rip`]). CPUID hides the virtualization bit, as Intel hides
//!   VMX ([`cpuid_policy`]); here it is SVM.
//!
//! Scope matches Intel section 5: CPUID, XSETBV, RDMSR/WRMSR, IN/OUT, HLT
//! and VMMCALL. Nested page faults (part 6), MTRRs (7), NMIs (8) and Hyper-V (9)
//! come later; [`decode`] returns [`Exit::Other`] for everything else.

use core::arch::x86_64::CpuidResult;

use super::control::{ControlArea, MsrAccess, msrpm_bit};
use super::features::GuestFeatures;
use super::vmcb::SaveArea;
use crate::hal::amd::register_access::{GuestRegister, GuestRegisters};
use crate::hal::types::IoSize;
use crate::hv::exit_decision::is_canonical_48;
use crate::hv::exit_decision::merge_in;
use crate::hv::features::CPUID_1_ECX_OSXSAVE;
use crate::hv::hyperv::SYNTHETIC_MSRS;
use crate::x86::cpuid::CPUID_7_ECX_OSPKE;
use crate::x86::cr::{CR4_OSXSAVE, CR4_PKE};
use crate::x86::msr::{IA32_KERNEL_GS_BASE, IA32_SYSENTER_CS, IA32_SYSENTER_EIP, IA32_SYSENTER_ESP};
use crate::x86::mtrr::{self, MtrrCapabilities, MtrrMsr};

/// SVM exit codes (AMD64 APM Vol. 2, Appendix C). A control-register or
/// exception intercept has its own code; only the ones this part handles are
/// named.
pub mod code {
    pub const CPUID: u64 = 0x72;
    pub const HLT: u64 = 0x78;
    pub const IOIO: u64 = 0x7B;
    pub const MSR: u64 = 0x7C;
    pub const VMMCALL: u64 = 0x81;
    pub const XSETBV: u64 = 0x8D;
    pub const NPF: u64 = 0x400;
}

/// What an exit turned out to be, in vendor-neutral terms. The same shape a
/// shared decider would consume for either backend.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Exit {
    /// CPUID with EAX = `leaf`, ECX = `subleaf`.
    Cpuid { leaf: u32, subleaf: u32 },
    /// RDMSR of `msr` (guest ECX).
    Rdmsr { msr: u32 },
    /// WRMSR of `value` (guest EDX:EAX) to `msr` (guest ECX).
    Wrmsr { msr: u32, value: u64 },
    /// IN of `size` from `port`.
    In { port: u16, size: IoSize },
    /// OUT of `value` (`size` wide) to `port`.
    Out { port: u16, size: IoSize, value: u32 },
    /// HLT: resume the guest.
    Hlt,
    /// VMMCALL: the guest's hypercall (Aleph0's own, or Hyper-V's later).
    Vmmcall,
    /// XSETBV of `value` (guest EDX:EAX) to XCR `xcr` (guest ECX). Exits
    /// only while `control::intercept4::XSETBV` is set.
    Xsetbv { xcr: u32, value: u64 },
    /// A nested page fault at `gpa`, with the page-fault `error_code`
    /// (`exit_info_1`). The AMD counterpart of an EPT violation.
    NestedPageFault { gpa: u64, error_code: u64 },
    /// Any exit this part does not yet handle, with its raw `exit_code`.
    Other { exit_code: u64 },
}

/// `exit_info_1` of an MSR exit: bit 0 is the access (0 = RDMSR, 1 = WRMSR).
const MSR_EXIT_IS_WRITE: u64 = 1 << 0;

/// IOIO `exit_info_1` layout (APM Vol. 2, 15.10.2).
mod ioio {
    pub const TYPE_IN: u64 = 1 << 0;
    pub const SZ8: u64 = 1 << 4;
    pub const SZ16: u64 = 1 << 5;
    pub const SZ32: u64 = 1 << 6;
}

/// The port and width of an IOIO exit from its `exit_info_1`.
fn io_port_size(info1: u64) -> (u16, IoSize) {
    let port = (info1 >> 16) as u16;
    let size = if info1 & ioio::SZ8 != 0 {
        IoSize::Byte
    } else if info1 & ioio::SZ16 != 0 {
        IoSize::Word
    } else {
        debug_assert!(info1 & ioio::SZ32 != 0);
        IoSize::Dword
    };
    (port, size)
}

/// Decodes an exit into a neutral [`Exit`], reading the guest's operand
/// registers through `regs`. `exit_code`, `info1` and `info2` are the VMCB
/// control fields ([`run::RunOutcome`](super::run::RunOutcome)).
///
/// `regs` reads cannot fail on AMD (the VMCB is memory), so the operand reads
/// are unwrapped.
pub fn decode<R: GuestRegisters>(exit_code: u64, info1: u64, info2: u64, regs: &R) -> Exit {
    let gp = |register| regs.read(register).ok().unwrap_or(0);
    match exit_code {
        code::CPUID => Exit::Cpuid {
            leaf: gp(GuestRegister::Rax) as u32,
            subleaf: gp(GuestRegister::Rcx) as u32,
        },
        code::MSR => {
            let msr = gp(GuestRegister::Rcx) as u32;
            if info1 & MSR_EXIT_IS_WRITE != 0 {
                let value = (gp(GuestRegister::Rdx) << 32) | (gp(GuestRegister::Rax) & 0xFFFF_FFFF);
                Exit::Wrmsr { msr, value }
            } else {
                Exit::Rdmsr { msr }
            }
        }
        code::IOIO => {
            let (port, size) = io_port_size(info1);
            if info1 & ioio::TYPE_IN != 0 {
                Exit::In { port, size }
            } else {
                Exit::Out {
                    port,
                    size,
                    value: out_value(gp(GuestRegister::Rax), size),
                }
            }
        }
        code::HLT => Exit::Hlt,
        code::VMMCALL => Exit::Vmmcall,
        code::XSETBV => Exit::Xsetbv {
            xcr: gp(GuestRegister::Rcx) as u32,
            value: (gp(GuestRegister::Rdx) << 32) | (gp(GuestRegister::Rax) & 0xFFFF_FFFF),
        },
        code::NPF => Exit::NestedPageFault {
            gpa: info2,
            error_code: info1,
        },
        other => Exit::Other { exit_code: other },
    }
}

/// CPUID.80000001:ECX bit 2: SVM. Cleared for the guest, as Intel clears the
/// VMX bit: a guest that saw it would try VMRUN, which always exits, and
/// Aleph0 does not nest SVM.
const CPUID_EXT_ECX_SVM: u32 = 1 << 2;

/// `bit` if `cr4` has `cr4_bit` set, else 0.
fn mirrors(cr4: u64, cr4_bit: u64, bit: u32) -> u32 {
    if cr4 & cr4_bit != 0 { bit } else { 0 }
}

/// What the guest sees for CPUID `leaf`/`subleaf`, given what the processor
/// returned in the host (`raw`), the guest's CR4 (`guest_cr4`) and what
/// bring-up offers it (`features`). Mirrors Intel's `cpuid_policy`:
///
/// - CPUID.80000001:ECX.SVM is cleared.
/// - CPUID.1:ECX.OSXSAVE and CPUID.(7,0):ECX.OSPKE copy the guest's
///   CR4.OSXSAVE and CR4.PKE. CPUID ran in the host, where they copy the
///   host's CR4, not the guest's.
/// - The bits of what bring-up does not offer are cleared
///   ([`GuestFeatures::hidden`]).
pub fn cpuid_policy(leaf: u32, subleaf: u32, raw: CpuidResult, guest_cr4: u64, features: GuestFeatures) -> CpuidResult {
    let hidden = features.hidden();
    match (leaf, subleaf) {
        (1, _) => CpuidResult {
            ecx: (raw.ecx & !(CPUID_1_ECX_OSXSAVE | hidden.leaf_1_ecx))
                | mirrors(guest_cr4, CR4_OSXSAVE, CPUID_1_ECX_OSXSAVE),
            ..raw
        },
        (7, 0) => CpuidResult {
            ebx: raw.ebx & !hidden.leaf_7_ebx,
            ecx: (raw.ecx & !(CPUID_7_ECX_OSPKE | hidden.leaf_7_ecx)) | mirrors(guest_cr4, CR4_PKE, CPUID_7_ECX_OSPKE),
            ..raw
        },
        (0xD, 1) => CpuidResult {
            eax: raw.eax & !hidden.leaf_d_1_eax,
            ..raw
        },
        (0x8000_0001, _) => CpuidResult {
            ecx: raw.ecx & !CPUID_EXT_ECX_SVM,
            edx: raw.edx & !hidden.extended_edx,
            ..raw
        },
        _ => raw,
    }
}

/// The VMCB field holding the guest's value of `msr` while the host runs,
/// or `None` for an MSR the CPU does not switch, whose real register is the
/// guest's. VMLOAD and VMSAVE switch these with the guest's VMCB around each
/// VMRUN (AMD64 APM Vol. 2, 15.5.2), so a passthrough RDMSR or WRMSR of one
/// reads and writes this field: a real RDMSR in the host would reach the
/// host's value. KVM's `svm_get_msr` and `svm_set_msr` do the same.
pub fn vmcb_held_msr(save: &mut SaveArea, msr: u32) -> Option<&mut u64> {
    match msr {
        IA32_KERNEL_GS_BASE => Some(&mut save.kernel_gs_base),
        IA32_STAR => Some(&mut save.star),
        IA32_LSTAR => Some(&mut save.lstar),
        IA32_CSTAR => Some(&mut save.cstar),
        IA32_SFMASK => Some(&mut save.sfmask),
        IA32_SYSENTER_CS => Some(&mut save.sysenter_cs),
        IA32_SYSENTER_ESP => Some(&mut save.sysenter_esp),
        IA32_SYSENTER_EIP => Some(&mut save.sysenter_eip),
        _ => None,
    }
}

/// IA32_STAR, IA32_LSTAR, IA32_CSTAR and IA32_SFMASK: the SYSCALL MSRs.
pub const IA32_STAR: u32 = 0xC000_0081;
pub const IA32_LSTAR: u32 = 0xC000_0082;
pub const IA32_CSTAR: u32 = 0xC000_0083;
pub const IA32_SFMASK: u32 = 0xC000_0084;

/// Which way an RDMSR or WRMSR exit goes: Intel's `MsrRoute`, from the
/// MSRPM's three ranges instead of the MSR bitmap's two.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MsrRoute {
    /// One of the MSRs bring-up intercepts besides the MTRRs (the
    /// `intercepted` list, `exit_maps::fill`): pass it through.
    Intercepted,
    /// One of the guest's MTRRs: its copy (`guest_mtrrs`).
    Mtrr(MtrrMsr),
    /// An address an MTRR could have that this CPU has no MSR at: #GP.
    AbsentMtrr,
    /// One of Hyper-V's synthetic MSRs, with Hyper-V's interface present:
    /// forward it (part 9).
    Synthetic,
    /// Outside the MSRPM's ranges, so it exits whatever the map says, and
    /// not forwarded. A real RDMSR/WRMSR of a missing MSR would #GP in the
    /// host, so the guest gets the #GP.
    OutOfRange,
    /// In range but its bit is clear, so it should never have exited: the
    /// map is broken. Stop.
    NotIntercepted,
}

/// Which [`MsrRoute`] `msr` takes, for a guest whose MTRR MSRs are `mtrr`,
/// with `intercepted` the other MSRs the MSRPM intercepts, and with Hyper-V's
/// interface (`hyperv`) or without. Intel's `route`.
pub fn route_msr(msr: u32, intercepted: &[u32], mtrr: MtrrCapabilities, hyperv: bool) -> MsrRoute {
    if msrpm_bit(msr, MsrAccess::Read).is_none() {
        if hyperv && SYNTHETIC_MSRS.contains(&msr) {
            MsrRoute::Synthetic
        } else {
            MsrRoute::OutOfRange
        }
    } else if intercepted.contains(&msr) {
        MsrRoute::Intercepted
    } else if let Some(msr) = mtrr.msr(msr) {
        MsrRoute::Mtrr(msr)
    } else if mtrr::is_msr_address(msr) {
        MsrRoute::AbsentMtrr
    } else {
        MsrRoute::NotIntercepted
    }
}

/// What to do with an RDMSR or WRMSR exit, as Intel's `rdmsr` and `wrmsr`
/// decide it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MsrAction {
    /// RDMSR or WRMSR of the real MSR, in the host, for the guest.
    Passthrough,
    /// RDMSR of the guest's MTRR `msr`: from its copy.
    MtrrRead { msr: MtrrMsr },
    /// WRMSR of `value` to the guest's MTRR `msr`: into its copy, and the NPT
    /// retyped. `value` is one WRMSR would take.
    MtrrWrite { msr: MtrrMsr, value: u64 },
    /// Forward to Hyper-V (part 9).
    Synthetic,
    /// #GP(0) to the guest ([`EventInjection::GENERAL_PROTECTION`]).
    GeneralProtection,
    /// The MSR should not have exited: stop.
    Unexpected,
}

/// What an RDMSR of `msr` gets.
pub fn rdmsr_action(msr: u32, intercepted: &[u32], mtrr: MtrrCapabilities, hyperv: bool) -> MsrAction {
    match route_msr(msr, intercepted, mtrr, hyperv) {
        MsrRoute::Intercepted => MsrAction::Passthrough,
        MsrRoute::Mtrr(msr) => MsrAction::MtrrRead { msr },
        MsrRoute::Synthetic => MsrAction::Synthetic,
        MsrRoute::AbsentMtrr | MsrRoute::OutOfRange => MsrAction::GeneralProtection,
        MsrRoute::NotIntercepted => MsrAction::Unexpected,
    }
}

/// What a WRMSR of `value` to `msr` gets. As on Intel, every intercepted
/// MSR holds an address, so a non-canonical value gets the #GP the real
/// WRMSR would raise in the host; and an MTRR value the MSR would refuse
/// gets its #GP, the copy staying as it was.
pub fn wrmsr_action(msr: u32, value: u64, intercepted: &[u32], mtrr: MtrrCapabilities, hyperv: bool) -> MsrAction {
    match route_msr(msr, intercepted, mtrr, hyperv) {
        MsrRoute::Intercepted if !is_canonical_48(value) => MsrAction::GeneralProtection,
        MsrRoute::Intercepted => MsrAction::Passthrough,
        MsrRoute::Mtrr(msr) => match mtrr.check(msr, value) {
            Ok(()) => MsrAction::MtrrWrite { msr, value },
            Err(_) => MsrAction::GeneralProtection,
        },
        MsrRoute::Synthetic => MsrAction::Synthetic,
        MsrRoute::AbsentMtrr | MsrRoute::OutOfRange => MsrAction::GeneralProtection,
        MsrRoute::NotIntercepted => MsrAction::Unexpected,
    }
}

/// Writes a CPUID result into the guest's EAX/EBX/ECX/EDX.
pub fn apply_cpuid<R: GuestRegisters>(regs: &mut R, result: CpuidResult) -> Result<(), R::Error> {
    regs.write(GuestRegister::Rax, u64::from(result.eax))?;
    regs.write(GuestRegister::Rbx, u64::from(result.ebx))?;
    regs.write(GuestRegister::Rcx, u64::from(result.ecx))?;
    regs.write(GuestRegister::Rdx, u64::from(result.edx))?;
    Ok(())
}

/// Writes an RDMSR result into the guest's EDX:EAX, the way the CPU returns
/// it. The high 32 bits of RAX and RDX are cleared, as RDMSR leaves them.
pub fn apply_rdmsr<R: GuestRegisters>(regs: &mut R, value: u64) -> Result<(), R::Error> {
    regs.write(GuestRegister::Rax, value & 0xFFFF_FFFF)?;
    regs.write(GuestRegister::Rdx, value >> 32)?;
    Ok(())
}

/// Writes an IN result into the guest's AL/AX/EAX, keeping the upper bits of
/// RAX for an 8- or 16-bit read, as the real IN does ([`merge_in`]).
pub fn apply_in<R: GuestRegisters>(regs: &mut R, size: IoSize, value: u32) -> Result<(), R::Error> {
    let old = regs.read(GuestRegister::Rax)?;
    regs.write(GuestRegister::Rax, merge_in(old, size, value))
}

// XCR0 components with rules of their own (AMD64 APM Vol. 2, 11.5.2; the
// same rules as `hv::exit_decision`'s, whose `xcr0_valid` is private there).
const XCR0_X87: u64 = 1 << 0;
const XCR0_SSE: u64 = 1 << 1;
const XCR0_AVX: u64 = 1 << 2;
const XCR0_MPX: u64 = 0b11 << 3;
const XCR0_AVX512: u64 = 0b111 << 5;
const XCR0_AMX: u64 = 0b11 << 17;

/// Whether XSETBV takes `value` for XCR0 on a CPU supporting the XCR0 bits
/// `supported`: no bit outside `supported`, x87 set, AVX only with SSE,
/// AVX-512 all or none and only with SSE and AVX, MPX and AMX both or
/// neither.
pub fn xcr0_valid(value: u64, supported: u64) -> bool {
    let all_or_none = |bits: u64| value & bits == 0 || value & bits == bits;
    let sse_avx = XCR0_SSE | XCR0_AVX;
    value & !supported == 0
        && value & XCR0_X87 != 0
        && (value & XCR0_AVX == 0 || value & XCR0_SSE != 0)
        && all_or_none(XCR0_AVX512)
        && (value & XCR0_AVX512 == 0 || value & sse_avx == sse_avx)
        && all_or_none(XCR0_MPX)
        && all_or_none(XCR0_AMX)
}

/// The value an OUT of `size` sends: AL, AX or EAX.
fn out_value(rax: u64, size: IoSize) -> u32 {
    match size {
        IoSize::Byte => u32::from(rax as u8),
        IoSize::Word => u32::from(rax as u16),
        IoSize::Dword => rax as u32,
    }
}

/// What to do with an XSETBV exit.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum XsetbvAction {
    /// Run XSETBV of `value` to XCR0 in the host
    /// (`x86::cr::xsetbv`), then advance RIP. VMRUN and #VMEXIT do not
    /// switch XCR0, so the guest's XCR0 is simply the CPU's. `value` is one XCR0 takes, so the real XSETBV cannot
    /// fault.
    Run { value: u64 },
    /// Queue [`EventInjection::GENERAL_PROTECTION`] ([`queue_event`]) and
    /// leave RIP on the XSETBV, as the CPU would fault.
    GeneralProtection,
}

/// XSETBV of `value` to XCR `xcr`: run in the host when it is XCR0 and a
/// value XCR0 takes, #GP(0) otherwise. Intel's `exit_decision::xsetbv`.
///
/// As on Intel, the CPU checks the #UD (guest CR4.OSXSAVE clear) and the
/// #GP (CPL > 0) before the intercept: SVM checks instruction intercepts
/// after simple exceptions such as #GP for a wrong CPL or #UD, and before
/// exceptions based on operand values (AMD64 APM Vol. 2, 15.9). So only the
/// operands are checked here. Without XSETBV in the host
/// ([`GuestFeatures::xsetbv`]) the supported bits are 0, so every value
/// gets the #GP.
pub fn xsetbv_action(xcr: u32, value: u64, features: GuestFeatures) -> XsetbvAction {
    if xcr == 0 && xcr0_valid(value, features.xcr0) {
        XsetbvAction::Run { value }
    } else {
        XsetbvAction::GeneralProtection
    }
}

/// An event-injection value: the VMCB's EVENTINJ field (AMD64 APM Vol. 2,
/// 15.20). The AMD counterpart of Intel's `InterruptionInfo` written to
/// VM_ENTRY_INTERRUPTION_INFO. Same low 32 bits; the error code sits in bits
/// 63:32 of the same field instead of a separate VMCS field.
///
/// ```text
///  63         32  31    30  12  11          10   8  7      0
/// ┌─────────────┬─────┬──────┬────────────┬───────┬────────┐
/// │ error code  │valid│ res. │ error code │ type  │ vector │
/// │             │     │      │   valid    │       │        │
/// └─────────────┴─────┴──────┴────────────┴───────┴────────┘
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EventInjection(pub u64);

impl EventInjection {
    /// Bit 31 (V): inject at the next VMRUN.
    pub const VALID: u64 = 1 << 31;
    /// Type 2 (bits 10:8): NMI.
    const TYPE_NMI: u64 = 2;
    /// Type 3 (bits 10:8): exception.
    const TYPE_EXCEPTION: u64 = 3;
    /// Bit 11 (EV): push the error code in bits 63:32.
    const ERROR_CODE_VALID: u64 = 1 << 11;
    /// Vector of NMI.
    const NMI_VECTOR: u64 = 2;
    /// Vector of #UD.
    const UD_VECTOR: u64 = 6;
    /// Vector of #GP.
    const GP_VECTOR: u64 = 13;

    /// Injects an NMI at the next VMRUN: valid, type NMI, vector 2. The CPU
    /// clears V after the event is taken, so it injects once.
    pub const NMI: Self = Self(Self::VALID | (Self::TYPE_NMI << 8) | Self::NMI_VECTOR);

    /// Injects #GP with error code 0 at the next VMRUN, delivered before the
    /// faulting instruction runs again.
    pub const GENERAL_PROTECTION: Self =
        Self(Self::VALID | (Self::TYPE_EXCEPTION << 8) | Self::ERROR_CODE_VALID | Self::GP_VECTOR);

    /// Injects #UD, which has no error code, at the next VMRUN.
    pub const INVALID_OPCODE: Self = Self(Self::VALID | (Self::TYPE_EXCEPTION << 8) | Self::UD_VECTOR);

    /// Whether the field describes an event at all (bit 31).
    pub fn is_valid(self) -> bool {
        self.0 & Self::VALID != 0
    }

    /// Whether the CPU pushes the error code in bits 63:32 (bit 11).
    pub fn delivers_error_code(self) -> bool {
        self.0 & Self::ERROR_CODE_VALID != 0
    }

    /// The vector (bits 7:0), if the field is valid.
    pub fn vector(self) -> Option<u8> {
        self.is_valid().then_some(self.0 as u8)
    }

    /// The error code (bits 63:32).
    pub fn error_code(self) -> u32 {
        (self.0 >> 32) as u32
    }
}

/// Queues `event` for the next VMRUN, with error code 0 if it delivers one,
/// through the guest's own IDT at the instruction RIP points to. Intel's
/// `queue_exception`: one write here, since EVENTINJ holds the error code
/// too.
pub fn queue_event(control: &mut ControlArea, event: EventInjection) {
    // Error code 0: the constants leave bits 63:32 clear.
    control.event_inj = event.0 & 0xFFFF_FFFF;
}

/// Moves the guest's RIP to `next_rip`, past the instruction that exited.
/// `next_rip` is the VMCB's `next_rip` field when the CPU has NRIPS; without
/// it, the caller computes the length and passes `rip + length`.
pub fn advance_rip<R: GuestRegisters>(regs: &mut R, next_rip: u64) -> Result<(), R::Error> {
    regs.write(GuestRegister::Rip, next_rip)
}

#[cfg(test)]
#[path = "../../../tests/unit/hal/amd/exit.rs"]
mod tests;
