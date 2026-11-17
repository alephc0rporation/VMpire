//! Unit tests for `src/hal/amd/hyperv.rs`.
//!
//! Compiled only by `cargo test-host`. The forwarder is assembled into the
//! test binary too, so its recovery point is checked against its bytes, as
//! Intel's.

use super::*;
use crate::hv::hyperv::msr_recovery;

fn address(label: *const u8) -> u64 {
    label as u64
}

#[test]
fn the_vmmcall_recovers_from_ud_and_everything_else_goes_to_the_msr_forwarders() {
    const GENERAL_PROTECTION: u8 = 13;
    let instruction = address(&raw const svm_hyperv_hypercall_instruction);
    let refused = address(&raw const svm_hyperv_hypercall_refused);
    assert_eq!(recovery(instruction, INVALID_OPCODE), Some(refused));
    assert_eq!(recovery(instruction, GENERAL_PROTECTION), None);
    assert_eq!(recovery(instruction + 1, INVALID_OPCODE), None);
    // Any other address is the shared MSR forwarders' to recover.
    for rip in [0, instruction + 1, refused] {
        assert_eq!(recovery(rip, GENERAL_PROTECTION), msr_recovery(rip, GENERAL_PROTECTION));
    }
}

#[test]
fn the_recovery_point_sits_on_the_vmmcall() {
    // SAFETY: the label is in the forwarder's assembled code, in the test
    // binary's text for as long as it runs, and never written.
    let bytes = unsafe { *(&raw const svm_hyperv_hypercall_instruction).cast::<[u8; 3]>() };
    assert_eq!(bytes, [0x0F, 0x01, 0xD9], "VMMCALL");
}

#[test]
fn a_vmmcall_from_cpl_above_0_gets_ud() {
    for rcx in [vmcall::BOOT_SERVICES_ENDED, 0x5C] {
        assert_eq!(vmmcall_action(3, rcx, true), VmmcallAction::InvalidOpcode);
    }
}

#[test]
fn aleph0s_own_calls_are_handled_and_the_rest_forwarded_under_hyper_v() {
    assert_eq!(
        vmmcall_action(0, vmcall::BOOT_SERVICES_ENDED, false),
        VmmcallAction::BootServicesEnded
    );
    assert_eq!(vmmcall_action(0, vmcall::TEST, true), VmmcallAction::Test);
    assert_eq!(
        vmmcall_action(0, 0x1_005C, true),
        VmmcallAction::Hypercall { input: 0x1_005C }
    );
    assert_eq!(vmmcall_action(0, 0x1_005C, false), VmmcallAction::InvalidOpcode);
}
