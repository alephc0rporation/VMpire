//! Part 9: Hyper-V's hypercalls, forwarded with VMMCALL. The AMD
//! counterpart of `hal::intel::hypercall`; the rest of Hyper-V's interface
//! (detection, synthetic MSRs, the MSR forwarders) is shared:
//! `hv::hyperv`.
//!
//! On AMD, Hyper-V's hypercall page holds VMMCALL, not VMCALL, so the
//! guest's hypercalls exit with code VMMCALL (0x81, `exit::code::VMMCALL`,
//! with `control::intercept4::VMMCALL` set; without that intercept VMMCALL
//! raises #UD in the guest). Aleph0 repeats each one with VMMCALL itself,
//! outside the guest, so L0 Hyper-V serves it as an access of Aleph0's own
//! virtual processor, as on Intel:
//!
//! ```text
//! guest (L2)                    Aleph0 (L1)                        Hyper-V (L0)
//! VMMCALL, RCX/RDX/R8 ─#VMEXIT─▶ forward_hypercall: same RCX/RDX/R8,
//!                    0x81        VMMCALL ─────────────────────────▶ serves it
//!        RAX, RCX, RDX, R8 ◀──── into the guest's registers ◀─────── RAX = status
//! ```
//!
//! A hypercall before the guest enabled the hypercall page gets #UD from
//! Hyper-V; the host fault handler resumes it at the forwarder's recovery
//! point ([`recovery`]), and the guest gets the #UD at its own VMMCALL.

use crate::hal::types::{HypercallRegisters, Refused};
use crate::hv::hyperv::msr_recovery;
use crate::hv::vmcall;

/// IDT vector the hypercall forwarder recovers from.
const INVALID_OPCODE: u8 = 6;

// The hypercall forwarder, Microsoft x64 ABI (`efiapi`), returning 0 when
// VMMCALL ran and 1 when Hyper-V raised #UD: Intel's
// `hyperv_forward_hypercall` with VMMCALL (0F 01 D9) for VMCALL.
core::arch::global_asm!(
    r#"
    .globl svm_hyperv_forward_hypercall
    .globl svm_hyperv_hypercall_instruction
    .globl svm_hyperv_hypercall_refused
svm_hyperv_forward_hypercall:
    push rcx
    mov rax, rcx
    mov rcx, [rax]
    mov rdx, [rax + 8]
    mov r8, [rax + 16]
svm_hyperv_hypercall_instruction:
    vmmcall
    pop r9
    mov [r9], rcx
    mov [r9 + 8], rdx
    mov [r9 + 16], r8
    mov [r9 + 24], rax
    xor eax, eax
    ret
svm_hyperv_hypercall_refused:
    pop r9
    mov eax, 1
    ret
"#
);

unsafe extern "efiapi" {
    fn svm_hyperv_forward_hypercall(registers: *mut HypercallRegisters) -> u64;
}

unsafe extern "C" {
    static svm_hyperv_hypercall_instruction: u8;
    static svm_hyperv_hypercall_refused: u8;
}

/// Where the host fault handler resumes after exception `vector` at `rip`:
/// the forwarder's `_refused` label, when `rip` is a forwarding instruction
/// and `vector` the exception Hyper-V refuses it with: #UD for the VMMCALL
/// here, #GP for the shared MSR forwarders ([`msr_recovery`]). `None` for
/// any other fault, which stays a bug in Aleph0.
pub fn recovery(rip: u64, vector: u8) -> Option<u64> {
    let hypercall = &raw const svm_hyperv_hypercall_instruction;
    if rip == hypercall as u64 && vector == INVALID_OPCODE {
        return Some(&raw const svm_hyperv_hypercall_refused as u64);
    }
    msr_recovery(rip, vector)
}

/// The guest's hypercall in `registers`, made again outside the guest:
/// Hyper-V's results replace them.
///
/// # Safety
///
/// Outside the guest under Hyper-V, at CPL 0, with the guest stopped at its
/// VMMCALL exit and the host IDT loaded, whose #UD gate recovers through
/// [`recovery`]. The hypercall acts on Aleph0's virtual processor and on the
/// guest pages the GPAs name, as the guest asked.
pub unsafe fn forward_hypercall(registers: &mut HypercallRegisters) -> Result<(), Refused> {
    // SAFETY: the caller's guarantees; the forwarder reads and writes only
    // `registers` and its own stack.
    match unsafe { svm_hyperv_forward_hypercall(registers) } {
        0 => Ok(()),
        _ => Err(Refused),
    }
}

/// What a VMMCALL of the firmware, or of the OS after it, gets: Intel's
/// `firmware_vmcall`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VmmcallAction {
    /// Aleph0's `ExitBootServices` event ran in the guest
    /// ([`vmcall::BOOT_SERVICES_ENDED`]): resume past it, in the runtime
    /// output phase from now on.
    BootServicesEnded,
    /// A call that only tests the exit ([`vmcall::TEST`]): resume past it.
    Test,
    /// A Hyper-V hypercall with input value `input`: forward it.
    Hypercall { input: u64 },
    /// #UD, as on hardware without a hypervisor, or as Hyper-V gives from
    /// CPL > 0.
    InvalidOpcode,
}

/// What a VMMCALL with RCX = `rcx` from CPL `cpl` gets, with Hyper-V's
/// interface (`hyperv`) or without:
///
/// | CPL | RCX | Action |
/// |---|---|---|
/// | > 0 | anything | #UD, as Hyper-V gives |
/// | 0 | one of Aleph0's own (`hv::vmcall`) | handled here |
/// | 0 | anything else, with Hyper-V's interface | a hypercall, forwarded |
/// | 0 | anything else, without it | #UD |
///
/// The CPL is the VMCB save area's CPL field.
pub fn vmmcall_action(cpl: u8, rcx: u64, hyperv: bool) -> VmmcallAction {
    if cpl != 0 {
        return VmmcallAction::InvalidOpcode;
    }
    match rcx {
        vmcall::BOOT_SERVICES_ENDED => VmmcallAction::BootServicesEnded,
        vmcall::TEST => VmmcallAction::Test,
        input if hyperv => VmmcallAction::Hypercall { input },
        _ => VmmcallAction::InvalidOpcode,
    }
}

#[cfg(test)]
#[path = "../../../tests/unit/hal/amd/hyperv.rs"]
mod tests;
