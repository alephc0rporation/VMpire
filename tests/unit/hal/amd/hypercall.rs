//! Unit tests for `src/hal/amd/hypercall.rs`.
//!
//! Compiled only by `cargo test-host`. The forwarder is assembled into the
//! test binary too, so its recovery point is checked against its bytes, as
//! Intel's.

use super::*;
use crate::hal::Hypercall as _;

fn address(label: *const u8) -> u64 {
    label as u64
}

#[test]
fn the_vmmcall_recovers_from_ud_at_its_refused_label() {
    const GENERAL_PROTECTION: u8 = 13;
    let point = Hypercall::recovery_point();
    let instruction = address(&raw const svm_hyperv_hypercall_instruction);
    let refused = address(&raw const svm_hyperv_hypercall_refused);
    assert_eq!(address(point.instruction), instruction);
    assert_eq!(address(point.resume), refused);
    assert_eq!(point.vector, INVALID_OPCODE);
    // The host fault handler's lookup (`hv::hyperv::recovery`) finds it, and
    // only for #UD at that very instruction.
    assert_eq!(crate::hv::hyperv::recovery(instruction, INVALID_OPCODE), Some(refused));
    assert_eq!(crate::hv::hyperv::recovery(instruction, GENERAL_PROTECTION), None);
    assert_eq!(crate::hv::hyperv::recovery(instruction + 1, INVALID_OPCODE), None);
}

#[test]
fn the_recovery_point_sits_on_the_vmmcall() {
    // SAFETY: the label is in the forwarder's assembled code, in the test
    // binary's text for as long as it runs, and never written.
    let bytes = unsafe { *(&raw const svm_hyperv_hypercall_instruction).cast::<[u8; 3]>() };
    assert_eq!(bytes, [0x0F, 0x01, 0xD9], "VMMCALL");
}
