//! Unit tests for `src/hal/amd/exit_maps.rs`.
//!
//! Compiled only by `cargo test-host`. That file includes this one with
//! `#[path]` as its child module `tests`, so `use super::*` reaches its
//! private items.

use super::*;

/// The maps [`fill`] makes for `msrs` and `ports`.
fn filled(msrs: &[u32], ports: &[u16]) -> (Vec<u8>, Vec<u8>) {
    let (mut msrpm, mut iopm) = (vec![0; MSRPM_SIZE], vec![0; IOPM_SIZE]);
    fill(msrs, ports, &mut msrpm, &mut iopm);
    (msrpm, iopm)
}

fn bit_set(map: &[u8], (offset, bit): (usize, u8)) -> bool {
    map[offset] & (1 << bit) != 0
}

#[test]
fn every_mtrr_address_exits_on_read_and_write() {
    let (msrpm, iopm) = filled(&[], &[]);
    for msr in mtrr::every_msr_address() {
        for access in [MsrAccess::Read, MsrAccess::Write] {
            assert!(bit_set(&msrpm, msrpm_bit(msr, access).unwrap()), "{msr:#x}");
        }
    }
    // Nothing else: IA32_EFER, IA32_PAT and every port pass through.
    assert!(!bit_set(&msrpm, msrpm_bit(0xC000_0080, MsrAccess::Read).unwrap()));
    assert!(!bit_set(&msrpm, msrpm_bit(0x277, MsrAccess::Write).unwrap()));
    assert!(iopm.iter().all(|&byte| byte == 0));
}

#[test]
fn the_given_msrs_and_ports_exit_too() {
    let (msrpm, iopm) = filled(&[0xC000_0102], &[0x80, 0x3F8]);
    assert!(bit_set(&msrpm, msrpm_bit(0xC000_0102, MsrAccess::Read).unwrap()));
    assert!(bit_set(&msrpm, msrpm_bit(0xC000_0102, MsrAccess::Write).unwrap()));
    assert!(bit_set(&iopm, iopm_bit(0x80)));
    assert!(bit_set(&iopm, iopm_bit(0x3F8)));
    assert!(!bit_set(&iopm, iopm_bit(0x81)));
}

#[test]
fn an_msr_outside_the_ranges_sets_nothing() {
    let (msrpm, _) = filled(&[0x4000_0020], &[]);
    let (mtrrs_only, _) = filled(&[], &[]);
    assert_eq!(msrpm, mtrrs_only);
}
