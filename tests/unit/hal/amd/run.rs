//! Unit tests for `src/hal/amd/run.rs`.
//!
//! Compiled only by `cargo test-host`. The run loop itself is SVM assembly
//! that faults outside a real AMD guest, so it is not called here; these
//! cover the exit-field decode and that it reads the right VMCB offsets.

use super::*;

#[test]
fn read_pulls_the_exit_fields_from_the_control_area() {
    let mut vmcb = Box::new(Vmcb::zeroed());
    vmcb.control.exit_code = 0x72; // VMEXIT_CPUID
    vmcb.control.exit_code_hi = 0;
    vmcb.control.exit_info_1 = 0x1234;
    vmcb.control.exit_info_2 = 0x5678;
    let outcome = RunOutcome::read(&vmcb);
    assert_eq!(
        outcome,
        RunOutcome {
            exit_code: 0x72,
            exit_info_1: 0x1234,
            exit_info_2: 0x5678,
        }
    );
    assert!(!outcome.is_invalid());
}

#[test]
fn the_high_half_of_the_exit_code_is_combined() {
    let mut vmcb = Box::new(Vmcb::zeroed());
    vmcb.control.exit_code = 0x0000_0000;
    vmcb.control.exit_code_hi = 0xFFFF_FFFF;
    // -1 across both halves is VMEXIT_INVALID: vmrun refused the guest.
    let outcome = RunOutcome::read(&vmcb);
    assert_eq!(outcome.exit_code, 0xFFFF_FFFF_0000_0000);
    assert!(!outcome.is_invalid(), "only all-ones is invalid");

    vmcb.control.exit_code = 0xFFFF_FFFF;
    let outcome = RunOutcome::read(&vmcb);
    assert_eq!(outcome.exit_code, VMEXIT_INVALID);
    assert!(outcome.is_invalid());
}

#[test]
fn a_fresh_vmcb_reports_a_zero_exit_before_it_runs() {
    let vmcb = Vmcb::zeroed();
    let outcome = RunOutcome::read(&vmcb);
    assert_eq!(outcome.exit_code, 0);
    assert!(!outcome.is_invalid());
}

#[test]
fn vmexit_invalid_is_minus_one() {
    assert_eq!(VMEXIT_INVALID, u64::MAX);
}
