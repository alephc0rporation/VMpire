//! Part 8: NMIs and the events an exit interrupted. The AMD counterpart of
//! the NMI handling in `hv::exit_decision` (`scenario.rs` and the
//! redelivery in `mod.rs`).
//!
//! As on Intel, an NMI belongs to the guest: Aleph0 intercepts it and
//! reinjects it, through EVENTINJ ([`EventInjection::NMI`]). The two
//! CPUs get there differently:
//!
//! - **Arrival.** With `intercept3::NMI` set, an NMI in the guest is a
//!   #VMEXIT with code [`code::NMI`], and the NMI stays pending: VMRUN's exit
//!   leaves GIF clear, and the CPU takes it through the host IDT once STGI
//!   sets GIF (AMD64 APM Vol. 2, 15.17; KVM's `nmi_interception` does
//!   nothing else). So every NMI reaches the host NMI handler, which marks it
//!   pending, as Intel's does for an NMI in root operation; the exit handler
//!   folds it in ([`NmiState::with_host_nmi`]).
//! - **Blocking.** VMX keeps the guest's NMI blocking in the VMCS and opens
//!   an NMI window by itself. SVM does neither: an injected event is
//!   delivered whatever the guest's state (APM 15.20), so Aleph0 tracks the
//!   guest's NMI masking itself, as KVM does (`HF_NMI_MASK`): set when it
//!   injects an NMI, cleared once the guest's IRET has run. An NMI may also
//!   not be injected inside an interrupt shadow (the VMCB's `int_state`
//!   bit 0, after STI or MOV SS), as KVM's `svm_nmi_blocked`.
//! - **Waiting.** SVM has no NMI-window exit, so Aleph0 makes its own the
//!   way KVM does (`arch/x86/kvm/svm/svm.c`): injecting an NMI turns the IRET
//!   intercept on ([`INTERCEPT_IRET`]); the IRET exit comes before the IRET
//!   runs, so it records the IRET's RIP and drops the intercept, and the
//!   guest counts as unmasked once its RIP has moved past it
//!   (`svm_complete_interrupts`). Whatever still blocks the NMI then (the
//!   IRET not yet run, or an interrupt shadow) is single-stepped over: TF
//!   and RF set in the guest's RFLAGS, #DB intercepted, and the guest's own
//!   TF and RF put back at the #DB exit (`svm_enable_nmi_window`,
//!   `db_interception`, `disable_nmi_singlestep`). Each step ends in an exit
//!   where the NMI is looked at again.
//!
//! The event re-delivery is Intel's: EXITINTINFO has the EVENTINJ layout, so
//! an event the exit interrupted mid-delivery is re-armed for the next VMRUN
//! ([`redelivery`]), an interrupted NMI as a true NMI.

use super::control::ControlArea;
use super::exit::EventInjection;

/// SVM exit codes this part handles (AMD64 APM Vol. 2, Appendix C). Like the
/// intercept bits of word 3, they are 0x60 plus the bit number.
pub mod code {
    /// A physical NMI while the guest ran (intercept word 3 bit 1).
    pub const NMI: u64 = 0x61;
    /// The guest's IRET (intercept word 3 bit 20), before it runs.
    pub const IRET: u64 = 0x74;
    /// A #DB in the guest (exception intercept, vector 1): 0x40 + 1.
    pub const DEBUG: u64 = 0x41;
}

/// Intercept word 3 bit 20: IRET exits, before it runs. Set while the guest
/// is in its NMI handler, to learn when it unmasks NMIs.
pub const INTERCEPT_IRET: u32 = 1 << 20;

/// `int_state` bit 0 (VMCB offset 0x068): the guest is in an interrupt
/// shadow, after STI or MOV SS. Intel's blocking by STI and by MOV SS in
/// one bit.
pub const INTERRUPT_SHADOW: u32 = 1 << 0;

/// EVENTINJ type 4 (bits 10:8): a software interrupt, INTn. Re-injecting
/// one needs the address of the instruction after it, the VMCB's NRIP.
const TYPE_SOFTWARE_INTERRUPT: u64 = 4;
/// EVENTINJ type 6: a software exception, INT3 or INTO, which needs NRIP
/// the same way.
const TYPE_SOFTWARE_EXCEPTION: u64 = 6;

/// RFLAGS bit 8 (TF): trap after the next instruction, a #DB.
pub const RFLAGS_TF: u64 = 1 << 8;
/// RFLAGS bit 16 (RF): no instruction breakpoint on the next instruction,
/// so the step is not taken over by a debug breakpoint at it.
pub const RFLAGS_RF: u64 = 1 << 16;
/// The exception-intercept bit of #DB (vector 1), in the control area's
/// `intercept_exceptions`. Its exit code is 0x40 + 1 ([`code::DEBUG`]).
pub const INTERCEPT_DB: u32 = 1 << 1;

/// What the exit handler remembers between exits about the guest's NMIs.
/// Intel's `ExitState` keeps `nmi_pending`, and the VMCS keeps the
/// blocking; here Aleph0 keeps both, as KVM's `vcpu_svm` does.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct NmiState {
    /// An NMI arrived and is not injected yet.
    pub pending: bool,
    /// The guest runs an NMI handler Aleph0 injected and has not ended it
    /// with IRET: another NMI waits. KVM's `nmi_masked`.
    pub masked: bool,
    /// The RIP of the guest's IRET, once its IRET exit came: NMIs unmask as
    /// soon as the guest's RIP is elsewhere. KVM's `awaiting_iret_completion`
    /// with `nmi_iret_rip`.
    pub iret_rip: Option<u64>,
    /// While Aleph0 single-steps the guest for a pending NMI: the guest's own
    /// TF and RF, to put back at the #DB exit. KVM's `nmi_singlestep` with
    /// `nmi_singlestep_guest_rflags`.
    pub single_step: Option<u64>,
}

impl NmiState {
    /// The state once an NMI the host NMI handler took (`arrived`) is
    /// folded in: pending for the guest. At most one NMI waits, as on real
    /// hardware.
    pub fn with_host_nmi(self, arrived: bool) -> Self {
        Self {
            pending: self.pending || arrived,
            ..self
        }
    }

    /// The state at an exit with the guest at `rip`: unmasked once the
    /// IRET it was stopped at has run (KVM's `svm_complete_interrupts`).
    /// Every exit starts with this.
    pub fn at_exit(self, rip: u64) -> Self {
        match self.iret_rip {
            Some(iret) if iret != rip => Self {
                masked: false,
                iret_rip: None,
                ..self
            },
            _ => self,
        }
    }
}

/// The intercepts and RFLAGS an exit leaves the guest with for its NMIs:
/// each `Some` is a change to make before the next VMRUN.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct NmiPlan {
    /// Write [`EventInjection::NMI`] to EVENTINJ.
    pub inject: bool,
    /// Turn the IRET intercept ([`INTERCEPT_IRET`]) on or off.
    pub intercept_iret: Option<bool>,
    /// Turn the #DB intercept ([`INTERCEPT_DB`]) on or off.
    pub intercept_db: Option<bool>,
    /// The guest's RFLAGS to write.
    pub rflags: Option<u64>,
}

/// What to do about a pending NMI at this exit, with the guest's
/// `int_state` and `rflags` as the exit left them, and whether another event
/// is already queued for the next VMRUN (`event_queued`), and the state
/// after: KVM's `svm_nmi_blocked`, `svm_inject_nmi` and
/// `svm_enable_nmi_window`.
///
/// - Nothing pending, or already single-stepping: nothing changes.
/// - Neither masked, nor in a shadow, nor behind another event: inject,
///   mask, and turn the IRET intercept on.
/// - Masked, before the IRET exit: the IRET intercept, on since the
///   injection, brings the next exit.
/// - Otherwise (the IRET not yet run, a shadow, or the other event, which
///   takes this VMRUN's one injection): single-step.
pub fn nmi_plan(state: NmiState, int_state: u32, rflags: u64, event_queued: bool) -> (NmiPlan, NmiState) {
    if !state.pending || state.single_step.is_some() {
        return (NmiPlan::default(), state);
    }
    let shadow = int_state & INTERRUPT_SHADOW != 0;
    if !state.masked && !shadow && !event_queued {
        let plan = NmiPlan {
            inject: true,
            intercept_iret: Some(true),
            ..NmiPlan::default()
        };
        return (
            plan,
            NmiState {
                pending: false,
                masked: true,
                ..state
            },
        );
    }
    if state.masked && state.iret_rip.is_none() {
        let plan = NmiPlan {
            intercept_iret: Some(true),
            ..NmiPlan::default()
        };
        return (plan, state);
    }
    let plan = NmiPlan {
        intercept_db: Some(true),
        rflags: Some(rflags | RFLAGS_TF | RFLAGS_RF),
        ..NmiPlan::default()
    };
    (
        plan,
        NmiState {
            single_step: Some(rflags & (RFLAGS_TF | RFLAGS_RF)),
            ..state
        },
    )
}

/// The IRET exit, with the guest at its IRET (`rip`): KVM's
/// `iret_interception`. NMIs stay masked until the IRET has run, which
/// [`NmiState::at_exit`] sees at a later exit; the intercept goes off.
pub fn iret_exit(state: NmiState, rip: u64) -> (NmiPlan, NmiState) {
    let plan = NmiPlan {
        intercept_iret: Some(false),
        ..NmiPlan::default()
    };
    (
        plan,
        NmiState {
            iret_rip: Some(rip),
            ..state
        },
    )
}

/// What a #DB exit was.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DebugExit {
    /// Aleph0's own single step for a pending NMI: done, with this plan.
    NmiStep(NmiPlan),
    /// The guest's own #DB: re-inject it.
    Guest,
}

/// A #DB exit ([`code::DEBUG`]), with the guest's `rflags`: KVM's
/// `db_interception` and `disable_nmi_singlestep`. Aleph0's single step
/// ends, the guest's own TF and RF are put back, and the #DB intercept goes
/// off; the NMI is looked at again at this exit ([`nmi_plan`]). A #DB while
/// Aleph0 was not stepping is the guest's.
pub fn debug_exit(state: NmiState, rflags: u64) -> (DebugExit, NmiState) {
    let Some(saved) = state.single_step else {
        return (DebugExit::Guest, state);
    };
    let plan = NmiPlan {
        intercept_db: Some(false),
        rflags: Some((rflags & !(RFLAGS_TF | RFLAGS_RF)) | saved),
        ..NmiPlan::default()
    };
    (
        DebugExit::NmiStep(plan),
        NmiState {
            single_step: None,
            ..state
        },
    )
}

/// Writes `plan`'s changes into `control` and returns the guest's RFLAGS to
/// write, if any (it lives in the save area). Merging plans is the caller's:
/// the IRET or #DB exit's plan first, then [`nmi_plan`]'s.
pub fn apply(plan: NmiPlan, control: &mut ControlArea) -> Option<u64> {
    if plan.inject {
        control.event_inj = EventInjection::NMI.0;
    }
    match plan.intercept_iret {
        Some(true) => control.intercept_word3 |= INTERCEPT_IRET,
        Some(false) => control.intercept_word3 &= !INTERCEPT_IRET,
        None => {}
    }
    match plan.intercept_db {
        Some(true) => control.intercept_exceptions |= INTERCEPT_DB,
        Some(false) => control.intercept_exceptions &= !INTERCEPT_DB,
        None => {}
    }
    plan.rflags
}

/// The event the exit interrupted mid-delivery, from `control`'s
/// EXITINTINFO, re-armed for the next VMRUN; `None` when the exit
/// interrupted no delivery, or when it was a software interrupt or exception
/// and the CPU has no NRIP (`next_rip`), without which it cannot be
/// re-injected (Intel stops the same way without VM_EXIT_INSTRUCTION_LEN).
///
/// Vector 2 is re-armed as a true NMI, whatever type EXITINTINFO gives it,
/// as Intel's redelivery does for Hyper-V's type-3 report of an NMI.
pub fn redelivery(control: &ControlArea, next_rip: bool) -> Option<EventInjection> {
    let info = EventInjection(u64::from(control.exit_int_info) | (u64::from(control.exit_int_info_err) << 32));
    if !info.is_valid() {
        return None;
    }
    let event_type = (info.0 >> 8) & 0b111;
    if info.vector() == Some(2) && matches!(event_type, 2 | 3) {
        return Some(EventInjection::NMI);
    }
    if matches!(event_type, TYPE_SOFTWARE_INTERRUPT | TYPE_SOFTWARE_EXCEPTION) && !next_rip {
        return None;
    }
    // Bits 30:12 are reserved in EVENTINJ; the error code, in bits 63:32,
    // matters only with EV.
    let error_code = if info.delivers_error_code() {
        info.0 & 0xFFFF_FFFF_0000_0000
    } else {
        0
    };
    Some(EventInjection((info.0 & 0x8000_0FFF) | error_code))
}

/// Has the next VMRUN deliver `event`: Intel's injection of an NMI or a
/// re-delivered event, with its error code, in one write.
pub fn queue(control: &mut ControlArea, event: EventInjection) {
    control.event_inj = event.0;
}

#[cfg(test)]
#[path = "../../../tests/unit/hal/amd/events.rs"]
mod tests;
