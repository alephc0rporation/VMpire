//! The AMD exit handler: the run loop that enters the guest, reads why it
//! stopped, acts, and enters it again. The AMD counterpart of
//! `hv::vmexit::handler` with `hv::exit_decision`: it puts parts 4 to 9
//! together.
//!
//! Each round:
//!
//! 1. With GIF clear ([`clgi`]), fold in an NMI the host NMI handler took,
//!    and decide what to do about a pending one ([`events::nmi_plan`]). An
//!    NMI that arrives after this is held, and VMRUN exits with code NMI at
//!    once.
//! 2. VMRUN ([`Vcpu::run`]); its exit sets GIF again, and a held NMI is then
//!    taken through the host IDT.
//! 3. Read the exit ([`exit::decode`] and the codes it leaves as
//!    [`Exit::Other`]), and act: emulate the instruction and move RIP past
//!    it (the VMCB's NRIP), inject an exception, grant a protected page, or
//!    stop.
//! 4. Re-deliver an event the exit interrupted ([`events::redelivery`]).
//!
//! Every exit prints one line on serial (COM2), as on Intel.

use core::arch::asm;
use core::arch::x86_64::__cpuid_count;

use super::events::{self, DebugExit, NmiState};
use super::exit::{
    self, EventInjection, Exit, MsrAction, XsetbvAction, apply_cpuid, apply_in, apply_rdmsr, cpuid_policy, queue_event,
    rdmsr_action, wrmsr_action, xsetbv_action,
};
use super::features;
use super::guest_mtrrs;
use super::npt;
use super::registers::SvmGuestRegisters;
use super::run::Vcpu;
use super::test_guest::ExpectedExits;
use crate::hal::amd::register_access::{GuestRegister, GuestRegisters};
use crate::hal::types::IoSize;
use crate::host::take_pending_nmi;
use crate::hv::guest::{GUEST_DATA_SEED, GUEST_EMULATION_FAIL_MARKER, GUEST_EPT_VALUE, GUEST_MARKER, GUEST_NMI_MARKER};
use crate::hv::hyperv;
use crate::port;

/// MSRs the MSRPM intercepts for the test guest besides the MTRRs, passed
/// through: IA32_KERNEL_GS_BASE, as Intel's test guest. VMSAVE and VMLOAD
/// switch it, so its passthrough reads and writes the VMCB's field
/// ([`exit::vmcb_held_msr`]).
pub const INTERCEPTED_MSRS: &[u32] = &[crate::x86::msr::IA32_KERNEL_GS_BASE];

/// How many main VMMCALLs the test guest makes; the last one ends the run.
const PLANNED_MAIN_VMMCALLS: u32 = 2;

/// Exception exit codes: 0x40 plus the vector.
const EXCEPTION_BASE: u64 = 0x40;
/// #VMEXIT SHUTDOWN: a triple fault in the guest.
const EXIT_SHUTDOWN: u64 = 0x7F;
/// IOIO EXITINFO1 bits 2 and 3: a string instruction (INS/OUTS) or REP.
const IOIO_STRING_OR_REP: u64 = 0b1100;
/// NPF EXITINFO1 bits 0 and 1: the page was present, and the access a
/// write: the protected page's write.
const NPF_PRESENT_WRITE: u64 = 0b11;

/// What the run counted.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct RunStats {
    pub cpuid: u32,
    pub xsetbv: u32,
    pub msr_reads: u32,
    pub msr_writes: u32,
    pub mtrr_reads: u32,
    pub mtrr_writes: u32,
    pub io_in: u32,
    pub io_out: u32,
    pub hlt: u32,
    pub exceptions_injected: u32,
    pub nmis_injected: u32,
    pub nmis_handled: u32,
}

impl RunStats {
    /// Whether the counts the test guest's script fixes are `expected`'s.
    pub fn matches(&self, expected: ExpectedExits) -> bool {
        self.cpuid == expected.cpuid
            && self.xsetbv == expected.xsetbv
            && self.msr_reads == expected.msr_reads
            && self.msr_writes == expected.msr_writes
            && self.mtrr_reads == expected.mtrr_reads
            && self.mtrr_writes == expected.mtrr_writes
            && self.io_in == expected.io_in
            && self.io_out == expected.io_out
            && self.hlt == expected.hlt
            && self.exceptions_injected == 0
    }
}

/// How the guest stopped.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GuestStop {
    /// The last main VMMCALL, after `exits` exits.
    Finished { exits: u64, stats: RunStats },
    /// VMRUN refused the guest state (exit code -1).
    InvalidState { exits: u64 },
    /// An exit Aleph0 does not handle, number `exit`, with its code, RIP and
    /// RAX.
    Unexpected { exit: u64, code: u64, rip: u64, rax: u64 },
    /// A check in the guest's script failed (`ud2` with the fail marker) at
    /// `rip`.
    GuestCheckFailed { exit: u64, rip: u64 },
    /// A main VMMCALL came before the write to the data page faulted.
    WriteNotProtected { exit: u64 },
    /// Main VMMCALL #1 did not read back the data page as planned.
    DataPageMismatch { exit: u64, rcx: u64, rdx: u64 },
    /// A nested page fault other than the protected pages' writes.
    NestedPageFault {
        exit: u64,
        gpa: u64,
        error_code: u64,
        rip: u64,
    },
    /// An event to re-deliver, with an action that needs the same VMRUN's
    /// one injection or moves RIP.
    RedeliveryConflict { exit: u64, code: u64 },
}

/// The test guest's progress through its script: Intel's `ExitState`
/// scenario fields.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
struct Scenario {
    main_vmmcalls: u32,
    writes_granted: u32,
    stack_grants: u32,
}

/// What an exit leads to.
enum Action {
    /// Resume: past the instruction (`advance`), with `inject` queued.
    Resume {
        advance: bool,
        inject: Option<EventInjection>,
    },
    /// The guest's run is over.
    Stop(GuestStop),
}

const RESUME_PAST: Action = Action::Resume {
    advance: true,
    inject: None,
};
const RESUME_AT: Action = Action::Resume {
    advance: false,
    inject: None,
};
const INJECT_GP: Action = Action::Resume {
    advance: false,
    inject: Some(EventInjection::GENERAL_PROTECTION),
};

/// Clears GIF: NMIs and interrupts are held until VMRUN sets it again in the
/// guest, or STGI in the host.
fn clgi() {
    // SAFETY: SVM is on (EFER.SVME), which CLGI needs; it changes only GIF.
    unsafe { asm!("clgi", options(nomem, nostack, preserves_flags)) };
}

/// Runs the test guest in `vcpu` until it stops, with `host_state` the page
/// VMSAVE and VMLOAD keep the host's state in.
pub fn run(vcpu: &mut Vcpu, host_state: u64) -> GuestStop {
    let mut nmi = NmiState::default();
    let mut scenario = Scenario::default();
    let mut stats = RunStats::default();
    let mut exits: u64 = 0;
    loop {
        // 1. NMIs, with GIF clear.
        clgi();
        nmi = nmi.with_host_nmi(take_pending_nmi());
        let event_queued = vcpu.vmcb.control.event_inj & EventInjection::VALID != 0;
        let (plan, next) = events::nmi_plan(nmi, vcpu.vmcb.control.int_state, vcpu.vmcb.save.rflags, event_queued);
        nmi = next;
        if plan.inject {
            stats.nmis_injected += 1;
        }
        if let Some(rflags) = events::apply(plan, &mut vcpu.vmcb.control) {
            vcpu.vmcb.save.rflags = rflags;
        }

        // 2. VMRUN.
        // SAFETY: SVM is on, the VMCB describes the guest bring-up filled,
        // and `host_state` is a live page of its own, never freed.
        let outcome = unsafe { vcpu.run(host_state) };
        exits += 1;
        if outcome.is_invalid() {
            return GuestStop::InvalidState { exits };
        }
        // The CPU took the queued event or reported it in EXITINTINFO; the
        // first run's flush is done.
        vcpu.vmcb.control.event_inj = 0;
        vcpu.vmcb.control.tlb_control = super::control::tlb_control::NONE;
        let rip = vcpu.vmcb.save.rip;
        nmi = nmi.at_exit(rip);
        let redelivery = events::redelivery(&vcpu.vmcb.control, true);

        // 3. The exit.
        let code = outcome.exit_code;
        let rax = vcpu.vmcb.save.rax;
        crate::serial_println!(
            "{} VM exit #{exits}: {} (code {code:#x}) at rip={rip:#018x} rax={rax:#x} info1={:#x} info2={:#x}.",
            crate::TAG,
            name(code),
            outcome.exit_info_1,
            outcome.exit_info_2,
        );
        let action = act(
            vcpu,
            outcome.exit_code,
            outcome.exit_info_1,
            outcome.exit_info_2,
            exits,
            &mut nmi,
            &mut scenario,
            &mut stats,
        );

        // 4. Resume, re-delivering an interrupted event.
        match action {
            Action::Stop(GuestStop::Finished { .. }) => return GuestStop::Finished { exits, stats },
            Action::Stop(stop) => return stop,
            Action::Resume { advance, inject } => {
                if let Some(event) = redelivery {
                    if advance || inject.is_some() {
                        return GuestStop::RedeliveryConflict { exit: exits, code };
                    }
                    events::queue(&mut vcpu.vmcb.control, event);
                }
                if let Some(event) = inject {
                    queue_event(&mut vcpu.vmcb.control, event);
                    stats.exceptions_injected += 1;
                }
                if advance {
                    vcpu.vmcb.save.rip = vcpu.vmcb.control.next_rip;
                    // Skipping the instruction ends an STI or MOV SS shadow,
                    // as Intel's `after_skipped_instruction`.
                    vcpu.vmcb.control.int_state &= !events::INTERRUPT_SHADOW;
                }
            }
        }
    }
}

/// What exit `code` is, for the report.
fn name(code: u64) -> &'static str {
    match code {
        exit::code::CPUID => "CPUID",
        exit::code::HLT => "HLT",
        exit::code::IOIO => "IOIO",
        exit::code::MSR => "MSR",
        exit::code::VMMCALL => "VMMCALL",
        exit::code::XSETBV => "XSETBV",
        exit::code::NPF => "nested page fault",
        events::code::NMI => "NMI",
        events::code::IRET => "IRET",
        EXIT_SHUTDOWN => "shutdown",
        0x40..=0x5F => "exception",
        _ => "other",
    }
}

/// Acts on exit `code` of the test guest in `vcpu`.
#[allow(clippy::too_many_arguments)]
fn act(
    vcpu: &mut Vcpu,
    code: u64,
    info1: u64,
    info2: u64,
    exit: u64,
    nmi: &mut NmiState,
    scenario: &mut Scenario,
    stats: &mut RunStats,
) -> Action {
    let rip = vcpu.vmcb.save.rip;
    let rax = vcpu.vmcb.save.rax;
    let unexpected = Action::Stop(GuestStop::Unexpected { exit, code, rip, rax });
    match code {
        events::code::NMI => {
            // The NMI stays pending until STGI; the host NMI handler marks
            // it for the guest, and the next round injects it.
            RESUME_AT
        }
        events::code::IRET => {
            let (plan, next) = events::iret_exit(*nmi, rip);
            *nmi = next;
            events::apply(plan, &mut vcpu.vmcb.control);
            RESUME_AT
        }
        events::code::DEBUG => match events::debug_exit(*nmi, vcpu.vmcb.save.rflags) {
            (DebugExit::NmiStep(plan), next) => {
                *nmi = next;
                if let Some(rflags) = events::apply(plan, &mut vcpu.vmcb.control) {
                    vcpu.vmcb.save.rflags = rflags;
                }
                RESUME_AT
            }
            (DebugExit::Guest, _) => unexpected,
        },
        EXIT_SHUTDOWN => unexpected,
        0x46 if rax as u32 == GUEST_EMULATION_FAIL_MARKER => Action::Stop(GuestStop::GuestCheckFailed { exit, rip }),
        c if (EXCEPTION_BASE..EXCEPTION_BASE + 32).contains(&c) => unexpected,
        _ => {
            let decoded = {
                let regs = SvmGuestRegisters::new(&mut vcpu.vmcb, &mut vcpu.gprs);
                exit::decode(code, info1, info2, &regs)
            };
            act_decoded(vcpu, decoded, info1, exit, scenario, stats).unwrap_or(unexpected)
        }
    }
}

/// Acts on an exit [`exit::decode`] understood; `None` for one the test
/// guest never makes.
fn act_decoded(
    vcpu: &mut Vcpu,
    decoded: Exit,
    info1: u64,
    exit: u64,
    scenario: &mut Scenario,
    stats: &mut RunStats,
) -> Option<Action> {
    let guest_features = features::published();
    Some(match decoded {
        Exit::Cpuid { leaf, subleaf } => {
            stats.cpuid += 1;
            let raw = __cpuid_count(leaf, subleaf);
            let result = cpuid_policy(leaf, subleaf, raw, vcpu.vmcb.save.cr4, guest_features);
            let mut regs = SvmGuestRegisters::new(&mut vcpu.vmcb, &mut vcpu.gprs);
            let _ = apply_cpuid(&mut regs, result);
            RESUME_PAST
        }
        Exit::Xsetbv { xcr, value } => {
            stats.xsetbv += 1;
            match xsetbv_action(xcr, value, guest_features) {
                XsetbvAction::Run { value } => {
                    // SAFETY: bring-up set the host's CR4.OSXSAVE when it
                    // offered XSETBV (`GuestFeatures::host_cr4`), and the
                    // value is one XCR0 takes.
                    unsafe { crate::x86::cr::xsetbv(value) };
                    RESUME_PAST
                }
                XsetbvAction::GeneralProtection => INJECT_GP,
            }
        }
        Exit::Rdmsr { msr } => {
            match rdmsr_action(msr, INTERCEPTED_MSRS, guest_mtrrs::capabilities(), hyperv::offered()) {
                MsrAction::Passthrough => {
                    stats.msr_reads += 1;
                    let value = match exit::vmcb_held_msr(&mut vcpu.vmcb.save, msr) {
                        Some(field) => *field,
                        // SAFETY: an MSR bring-up intercepts, which the CPU
                        // has, and which the CPU does not switch.
                        None => unsafe { crate::x86::msr::rdmsr(msr) },
                    };
                    let mut regs = SvmGuestRegisters::new(&mut vcpu.vmcb, &mut vcpu.gprs);
                    let _ = apply_rdmsr(&mut regs, value);
                    RESUME_PAST
                }
                MsrAction::MtrrRead { msr } => {
                    stats.mtrr_reads += 1;
                    let value = guest_mtrrs::read(msr).ok()?;
                    let mut regs = SvmGuestRegisters::new(&mut vcpu.vmcb, &mut vcpu.gprs);
                    let _ = apply_rdmsr(&mut regs, value);
                    RESUME_PAST
                }
                MsrAction::Synthetic => {
                    // SAFETY: outside the guest at CPL 0 under Hyper-V, the
                    // guest stopped at its RDMSR, the host IDT loaded.
                    match unsafe { hyperv::forward_rdmsr(msr) } {
                        Ok(value) => {
                            let mut regs = SvmGuestRegisters::new(&mut vcpu.vmcb, &mut vcpu.gprs);
                            let _ = apply_rdmsr(&mut regs, value);
                            RESUME_PAST
                        }
                        Err(_) => INJECT_GP,
                    }
                }
                MsrAction::GeneralProtection => INJECT_GP,
                MsrAction::MtrrWrite { .. } | MsrAction::Unexpected => return None,
            }
        }
        Exit::Wrmsr { msr, value } => {
            match wrmsr_action(
                msr,
                value,
                INTERCEPTED_MSRS,
                guest_mtrrs::capabilities(),
                hyperv::offered(),
            ) {
                MsrAction::Passthrough => {
                    stats.msr_writes += 1;
                    match exit::vmcb_held_msr(&mut vcpu.vmcb.save, msr) {
                        Some(field) => *field = value,
                        // SAFETY: an intercepted MSR the CPU does not switch,
                        // with a canonical value, as `wrmsr_action` checked.
                        None => unsafe { crate::x86::msr::wrmsr(msr, value) },
                    }
                    RESUME_PAST
                }
                MsrAction::MtrrWrite { msr, value } => {
                    stats.mtrr_writes += 1;
                    // SAFETY: exit context: the guest is stopped.
                    unsafe { guest_mtrrs::write(msr, value, &mut vcpu.vmcb.control) }.ok()?;
                    RESUME_PAST
                }
                MsrAction::Synthetic => {
                    // SAFETY: as for the RDMSR.
                    match unsafe { hyperv::forward_wrmsr(msr, value) } {
                        Ok(()) => RESUME_PAST,
                        Err(_) => INJECT_GP,
                    }
                }
                MsrAction::GeneralProtection => INJECT_GP,
                MsrAction::MtrrRead { .. } | MsrAction::Unexpected => return None,
            }
        }
        Exit::In { port: number, size } => {
            if info1 & IOIO_STRING_OR_REP != 0 {
                return None;
            }
            stats.io_in += 1;
            // SAFETY: a port bring-up intercepts, read for the guest.
            let value = unsafe {
                match size {
                    IoSize::Byte => u32::from(port::inb(number)),
                    IoSize::Word => u32::from(port::inw(number)),
                    IoSize::Dword => port::inl(number),
                }
            };
            let mut regs = SvmGuestRegisters::new(&mut vcpu.vmcb, &mut vcpu.gprs);
            let _ = apply_in(&mut regs, size, value);
            RESUME_PAST
        }
        Exit::Out {
            port: number,
            size,
            value,
        } => {
            if info1 & IOIO_STRING_OR_REP != 0 {
                return None;
            }
            stats.io_out += 1;
            // SAFETY: a port bring-up intercepts, written for the guest.
            unsafe {
                match size {
                    IoSize::Byte => port::outb(number, value as u8),
                    IoSize::Word => port::outw(number, value as u16),
                    IoSize::Dword => port::outl(number, value),
                }
            }
            RESUME_PAST
        }
        Exit::Hlt => {
            stats.hlt += 1;
            RESUME_PAST
        }
        Exit::Vmmcall => {
            let regs = SvmGuestRegisters::new(&mut vcpu.vmcb, &mut vcpu.gprs);
            let rax = regs.read(GuestRegister::Rax).unwrap_or(0);
            let rcx = regs.read(GuestRegister::Rcx).unwrap_or(0);
            let rdx = regs.read(GuestRegister::Rdx).unwrap_or(0);
            vmmcall(rax, rcx, rdx, exit, scenario, stats)?
        }
        Exit::NestedPageFault { gpa, error_code } => nested_page_fault(vcpu, gpa, error_code, exit, scenario),
        Exit::Other { .. } => return None,
    })
}

/// A VMMCALL of the test guest: its NMI handler's, or its next main one.
/// Intel's `scenario::vmcall`. `None` for a VMMCALL it never makes.
fn vmmcall(rax: u64, rcx: u64, rdx: u64, exit: u64, scenario: &mut Scenario, stats: &mut RunStats) -> Option<Action> {
    if rax as u32 == GUEST_NMI_MARKER {
        stats.nmis_handled += 1;
        crate::serial_println!(
            "{} The guest's NMI handler ran: NMI reinjected and handled.",
            crate::TAG
        );
        return Some(RESUME_PAST);
    }
    if rax != u64::from(GUEST_MARKER) + u64::from(scenario.main_vmmcalls) {
        return None;
    }
    if scenario.writes_granted == 0 {
        return Some(Action::Stop(GuestStop::WriteNotProtected { exit }));
    }
    if scenario.main_vmmcalls == 0 && (rcx as u32 != GUEST_DATA_SEED || rdx as u32 != GUEST_EPT_VALUE) {
        return Some(Action::Stop(GuestStop::DataPageMismatch { exit, rcx, rdx }));
    }
    scenario.main_vmmcalls += 1;
    crate::serial_println!(
        "{} Main VMMCALL #{} handled (data page read {rcx:#x}, then {rdx:#x}).",
        crate::TAG,
        scenario.main_vmmcalls
    );
    if scenario.main_vmmcalls < PLANNED_MAIN_VMMCALLS {
        Some(RESUME_PAST)
    } else {
        Some(Action::Stop(GuestStop::Finished {
            exits: exit,
            stats: *stats,
        }))
    }
}

/// A nested page fault: the test guest's write to its protected data page,
/// or the reinjected NMI's delivery pushing onto its protected stack page,
/// each granted once; any other stops with the walk of the GPA. Intel's
/// `scenario::ept_violation`.
fn nested_page_fault(vcpu: &mut Vcpu, gpa: u64, error_code: u64, exit: u64, scenario: &mut Scenario) -> Action {
    let rip = vcpu.vmcb.save.rip;
    let protected = npt::protected_page_for(gpa);
    let write = error_code & NPF_PRESENT_WRITE == NPF_PRESENT_WRITE;
    if let (Some(page), true) = (protected, write) {
        let stack = page.gpa == npt::nmi_stack_page();
        let first = if stack {
            scenario.stack_grants == 0
        } else {
            scenario.writes_granted == 0
        };
        if first {
            // SAFETY: the NPT tables are in place; the guest is stopped.
            let flush = unsafe { page.grant_write(&mut vcpu.vmcb.control) };
            if stack {
                scenario.stack_grants += 1;
            } else {
                scenario.writes_granted += 1;
            }
            crate::serial_println!(
                "{} {} page {:#x} made writable (flush {}); the {} runs again.",
                crate::TAG,
                if stack { "Stack" } else { "Data" },
                page.gpa,
                flush.name(),
                if stack { "NMI delivery" } else { "write" },
            );
            return RESUME_AT;
        }
    }
    Action::Stop(GuestStop::NestedPageFault {
        exit,
        gpa,
        error_code,
        rip,
    })
}
