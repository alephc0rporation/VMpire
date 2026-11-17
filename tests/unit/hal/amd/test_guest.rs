//! Unit tests for `src/hal/amd/test_guest.rs`.
//!
//! Compiled only by `cargo test-host`. The guest code is assembled into the
//! test binary too, so its bytes are checked here, as Intel's.

use super::*;

#[test]
fn the_guest_code_fits_in_one_page_and_the_nmi_handler_follows_the_main_code() {
    assert!(guest_code().len() <= PAGE_SIZE);
    assert!(nmi_handler_offset() > 0 && nmi_handler_offset() < guest_code().len());
}

#[test]
fn the_guest_makes_its_vmmcalls_and_no_vmcall() {
    let code = guest_code();
    let count = |pattern: &[u8]| code.windows(pattern.len()).filter(|window| *window == pattern).count();
    // VMMCALL 0F 01 D9: main #1, main #2 and the NMI handler's.
    assert_eq!(count(&[0x0F, 0x01, 0xD9]), 3);
    // VMCALL 0F 01 C1 is Intel's, which would raise #UD on AMD.
    assert_eq!(count(&[0x0F, 0x01, 0xC1]), 0);
}
