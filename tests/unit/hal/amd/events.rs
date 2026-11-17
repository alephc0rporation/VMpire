//! Unit tests for `src/hal/amd/events.rs`.
//!
//! Compiled only by `cargo test-host`. That file includes this one with
//! `#[path]` as its child module `tests`, so `use super::*` reaches its
//! private items. They mirror the NMI and redelivery tests of
//! `tests/unit/vmx/exit_decision.rs`.

use super::*;
use crate::hal::amd::vmcb::Vmcb;

const PENDING: NmiState = NmiState {
    pending: true,
    masked: false,
    iret_rip: None,
    single_step: None,
};
/// RFLAGS with only the reserved bit 1 and IF set.
const RFLAGS: u64 = 0x202;
/// The RIP of the guest's NMI handler's IRET.
const IRET_RIP: u64 = 0x7e00_1234;

#[test]
fn an_nmi_while_unblocked_is_injected_masks_the_next_and_intercepts_iret() {
    let (plan, next) = nmi_plan(PENDING, 0, RFLAGS, false);
    assert_eq!(
        plan,
        NmiPlan {
            inject: true,
            intercept_iret: Some(true),
            ..NmiPlan::default()
        }
    );
    assert_eq!(
        next,
        NmiState {
            pending: false,
            masked: true,
            ..PENDING
        }
    );
}

#[test]
fn a_masked_nmi_waits_for_the_iret_exit() {
    let masked = NmiState {
        masked: true,
        ..PENDING
    };
    let (plan, next) = nmi_plan(masked, 0, RFLAGS, false);
    assert_eq!(
        plan,
        NmiPlan {
            intercept_iret: Some(true),
            ..NmiPlan::default()
        }
    );
    assert_eq!(next, masked);
}

#[test]
fn the_iret_exit_records_the_iret_and_drops_the_intercept() {
    let masked = NmiState {
        masked: true,
        ..PENDING
    };
    let (plan, next) = iret_exit(masked, IRET_RIP);
    assert_eq!(plan.intercept_iret, Some(false));
    assert_eq!(next.iret_rip, Some(IRET_RIP));
    assert!(next.masked);
}

#[test]
fn after_the_iret_exit_the_guest_is_single_stepped_over_the_iret() {
    let at_iret = NmiState {
        masked: true,
        iret_rip: Some(IRET_RIP),
        ..PENDING
    };
    // Still at the IRET: masked, so step.
    let at_iret = at_iret.at_exit(IRET_RIP);
    let (plan, stepping) = nmi_plan(at_iret, 0, RFLAGS, false);
    assert_eq!(
        plan,
        NmiPlan {
            intercept_db: Some(true),
            rflags: Some(RFLAGS | RFLAGS_TF | RFLAGS_RF),
            ..NmiPlan::default()
        }
    );
    assert_eq!(stepping.single_step, Some(0));
    // While stepping, nothing more is done.
    assert_eq!(nmi_plan(stepping, 0, RFLAGS, false).0, NmiPlan::default());

    // The #DB after the IRET: past it, so unmasked, the guest's TF and RF
    // back, and the NMI goes in.
    let after = stepping.at_exit(IRET_RIP + 2);
    assert!(!after.masked);
    let (debug, after) = debug_exit(after, 0x302 | RFLAGS_RF);
    assert_eq!(
        debug,
        DebugExit::NmiStep(NmiPlan {
            intercept_db: Some(false),
            rflags: Some(0x202),
            ..NmiPlan::default()
        })
    );
    assert!(nmi_plan(after, 0, 0x202, false).0.inject);
}

#[test]
fn an_nmi_inside_an_interrupt_shadow_is_single_stepped_past_it() {
    let (plan, stepping) = nmi_plan(PENDING, INTERRUPT_SHADOW, RFLAGS, false);
    assert_eq!(plan.rflags, Some(RFLAGS | RFLAGS_TF | RFLAGS_RF));
    assert_eq!(plan.intercept_db, Some(true));
    assert!(!plan.inject);
    assert!(stepping.pending);
}

#[test]
fn the_guests_own_tf_survives_the_step() {
    let (_, stepping) = nmi_plan(PENDING, INTERRUPT_SHADOW, RFLAGS | RFLAGS_TF, false);
    assert_eq!(stepping.single_step, Some(RFLAGS_TF));
    let (DebugExit::NmiStep(plan), _) = debug_exit(stepping, RFLAGS | RFLAGS_TF | RFLAGS_RF) else {
        panic!("Aleph0's step")
    };
    assert_eq!(plan.rflags, Some(RFLAGS | RFLAGS_TF));
}

#[test]
fn a_db_aleph0_did_not_cause_is_the_guests() {
    assert_eq!(debug_exit(PENDING, RFLAGS), (DebugExit::Guest, PENDING));
}

#[test]
fn nothing_pending_means_nothing_to_do() {
    let idle = NmiState::default();
    assert_eq!(
        nmi_plan(idle, INTERRUPT_SHADOW, RFLAGS, false),
        (NmiPlan::default(), idle)
    );
}

#[test]
fn an_nmi_from_the_host_becomes_pending_once() {
    let idle = NmiState::default();
    assert_eq!(idle.with_host_nmi(false), idle);
    assert_eq!(idle.with_host_nmi(true), PENDING);
    assert_eq!(PENDING.with_host_nmi(true), PENDING);
}

#[test]
fn apply_writes_the_injection_intercepts_and_returns_the_rflags() {
    let mut vmcb = Box::new(Vmcb::zeroed());
    let plan = NmiPlan {
        inject: true,
        intercept_iret: Some(true),
        intercept_db: Some(true),
        rflags: Some(0x302),
    };
    assert_eq!(apply(plan, &mut vmcb.control), Some(0x302));
    assert_eq!(vmcb.control.event_inj, EventInjection::NMI.0);
    assert_ne!(vmcb.control.intercept_word3 & INTERCEPT_IRET, 0);
    assert_ne!(vmcb.control.intercept_exceptions & INTERCEPT_DB, 0);
    let off = NmiPlan {
        intercept_iret: Some(false),
        intercept_db: Some(false),
        ..NmiPlan::default()
    };
    assert_eq!(apply(off, &mut vmcb.control), None);
    assert_eq!(vmcb.control.intercept_word3 & INTERCEPT_IRET, 0);
    assert_eq!(vmcb.control.intercept_exceptions & INTERCEPT_DB, 0);
}

#[test]
fn the_exit_codes_are_0x60_plus_the_intercept_bit() {
    assert_eq!(code::NMI, 0x60 + 1);
    assert_eq!(code::IRET, 0x60 + u64::from(INTERCEPT_IRET.trailing_zeros()));
    assert_eq!(code::DEBUG, 0x40 + u64::from(INTERCEPT_DB.trailing_zeros()));
}

/// A VMCB whose EXITINTINFO holds `info`, with error code `error`.
fn exit_int_info(info: u32, error: u32) -> Box<Vmcb> {
    let mut vmcb = Box::new(Vmcb::zeroed());
    vmcb.control.exit_int_info = info;
    vmcb.control.exit_int_info_err = error;
    vmcb
}

#[test]
fn no_valid_exitintinfo_means_no_redelivery() {
    assert_eq!(redelivery(&exit_int_info(0x0B0E, 4).control, true), None);
}

#[test]
fn an_interrupted_nmi_is_re_delivered_as_an_nmi_of_either_type() {
    for info in [0x8000_0202, 0x8000_0302] {
        assert_eq!(
            redelivery(&exit_int_info(info, 0).control, true),
            Some(EventInjection::NMI),
            "{info:#x}"
        );
    }
}

#[test]
fn an_interrupted_page_fault_keeps_its_error_code() {
    // #PF: type 3, EV, vector 14, error code 0x6.
    let event = redelivery(&exit_int_info(0x8000_0B0E, 6).control, true).unwrap();
    assert_eq!(event.vector(), Some(14));
    assert!(event.delivers_error_code());
    assert_eq!(event.error_code(), 6);
    // Without EV the error code is dropped.
    let event = redelivery(&exit_int_info(0x8000_0320, 6).control, true).unwrap();
    assert_eq!(event.error_code(), 0);
}

#[test]
fn reserved_bits_are_cleared_on_the_way_back() {
    let event = redelivery(&exit_int_info(0x8000_F030, 0).control, true).unwrap();
    assert_eq!(event.0, 0x8000_0030);
}

#[test]
fn a_software_interrupt_needs_nrip() {
    // INT 0x80: type 4.
    assert!(redelivery(&exit_int_info(0x8000_0480, 0).control, true).is_some());
    assert_eq!(redelivery(&exit_int_info(0x8000_0480, 0).control, false), None);
    // INT3: type 6.
    assert_eq!(redelivery(&exit_int_info(0x8000_0603, 0).control, false), None);
}

#[test]
fn queue_writes_the_event_with_its_error_code() {
    let mut vmcb = Box::new(Vmcb::zeroed());
    queue(&mut vmcb.control, EventInjection(0x0000_0006_8000_0B0E));
    assert_eq!(vmcb.control.event_inj, 0x0000_0006_8000_0B0E);
}

#[test]
fn an_nmi_behind_another_queued_event_is_single_stepped() {
    let (plan, stepping) = nmi_plan(PENDING, 0, RFLAGS, true);
    assert!(!plan.inject);
    assert_eq!(plan.rflags, Some(RFLAGS | RFLAGS_TF | RFLAGS_RF));
    assert!(stepping.pending);
}
