//! Unit tests for `src/hal/amd/control.rs`.
//!
//! Compiled only by `cargo test-host`. The control-area offsets are checked
//! at compile time in that file; these check the bitmap index math and the
//! intercept bit positions against the APM.

use super::*;
use std::collections::HashSet;

#[test]
fn a_zeroed_control_area_intercepts_nothing() {
    let control = ControlArea::zeroed();
    assert_eq!(control.intercept_word3, 0);
    assert_eq!(control.intercept_word4, 0);
    assert_eq!(control.nested_ctl, 0);
    assert_eq!(size_of::<ControlArea>(), 0x400);
}

#[test]
fn the_intercept_bits_match_the_apm() {
    // Word 3 (APM Vol. 2, Table 15-7).
    assert_eq!(intercept3::CPUID, 1 << 18);
    assert_eq!(intercept3::HLT, 1 << 24);
    assert_eq!(intercept3::IOIO_PROT, 1 << 27);
    assert_eq!(intercept3::MSR_PROT, 1 << 28);
    // Word 4 (Table 15-8).
    assert_eq!(intercept4::VMRUN, 1 << 0);
    assert_eq!(intercept4::VMMCALL, 1 << 1);
}

#[test]
fn the_three_msrpm_ranges_cover_the_architectural_msrs() {
    // Low MSRs, the SYSCALL/EFER block, and the AMD-specific block.
    assert!(msrpm_bit(0x0000_0000, MsrAccess::Read).is_some());
    assert!(msrpm_bit(0x0000_0277, MsrAccess::Read).is_some()); // IA32_PAT
    assert!(msrpm_bit(0xC000_0080, MsrAccess::Write).is_some()); // EFER
    assert!(msrpm_bit(0xC001_0114, MsrAccess::Read).is_some()); // VM_CR
    // Gaps between the ranges are not covered.
    assert_eq!(msrpm_bit(0x0000_2000, MsrAccess::Read), None);
    assert_eq!(msrpm_bit(0x8000_0000, MsrAccess::Read), None);
    assert_eq!(msrpm_bit(0xC001_2000, MsrAccess::Read), None);
}

#[test]
fn the_read_and_write_bits_of_an_msr_are_adjacent() {
    let (read_byte, read_bit) = msrpm_bit(0, MsrAccess::Read).unwrap();
    let (write_byte, write_bit) = msrpm_bit(0, MsrAccess::Write).unwrap();
    assert_eq!((read_byte, read_bit), (0, 0));
    assert_eq!((write_byte, write_bit), (0, 1));
    // Four MSRs (eight bits) per byte: MSR 4's read bit starts the next byte.
    assert_eq!(msrpm_bit(4, MsrAccess::Read), Some((1, 0)));
}

#[test]
fn every_msrpm_bit_is_within_the_map_and_unique() {
    let mut seen = HashSet::new();
    for range in MSRPM_RANGES {
        for msr in [range.start, range.start + 1, range.end - 1] {
            for access in [MsrAccess::Read, MsrAccess::Write] {
                let (byte, bit) = msrpm_bit(msr, access).unwrap();
                assert!(byte < MSRPM_SIZE, "MSR {msr:#x} byte {byte} past the map");
                assert!(bit < 8);
                assert!(seen.insert((byte, bit)), "MSR {msr:#x} {access:?} collides");
            }
        }
    }
}

#[test]
fn the_highest_port_bit_is_the_last_byte_of_two_pages() {
    assert_eq!(iopm_bit(0), (0, 0));
    assert_eq!(iopm_bit(8), (1, 0));
    // Port 0xFFFF is the last of 65536 bits = 8192 bytes.
    assert_eq!(iopm_bit(0xFFFF), (0x1FFF, 7));
    // Every port's byte fits within the allocated IOPM.
    for port in [0u16, 8, 0x8000, 0xFFFF] {
        let (byte, _) = iopm_bit(port);
        assert!(byte < IOPM_SIZE, "port {port:#x} byte {byte} past the IOPM");
    }
}
