//! Unit tests for `src/hal/amd/features.rs`.
//!
//! Compiled only by `cargo test-host`. That file includes this one with
//! `#[path]` as its child module `tests`, so `use super::*` reaches its
//! private items.

use super::*;

/// CPUID.1:ECX with XSAVE (bit 26) set, OSXSAVE (bit 27) clear.
const LEAF_1_ECX: u32 = 0xF7FA_3223;
/// x87, SSE, AVX and the three AVX-512 components.
const XCR0_SUPPORTED: u64 = 0xE7;

#[test]
fn a_cpu_with_xsave_gets_xsetbv_and_no_cpuid_bit_hidden() {
    let features = GuestFeatures::new(LEAF_1_ECX, XCR0_SUPPORTED);
    assert!(features.xsetbv());
    assert_eq!(features.xcr0, XCR0_SUPPORTED);
    assert_eq!(features.hidden(), HiddenCpuid::NONE);
}

#[test]
fn without_xsave_xsetbv_is_off_and_xsave_is_hidden() {
    let features = GuestFeatures::new(LEAF_1_ECX & !CPUID_1_ECX_XSAVE, XCR0_SUPPORTED);
    assert!(!features.xsetbv());
    assert_eq!(features.xcr0, 0);
    assert_eq!(features.host_cr4(0x2668), 0x2668);
    assert_eq!(
        features.hidden(),
        HiddenCpuid {
            leaf_1_ecx: CPUID_1_ECX_XSAVE,
            ..HiddenCpuid::NONE
        }
    );
}

#[test]
fn host_cr4_gets_osxsave_when_xsetbv_runs_in_the_host() {
    let features = GuestFeatures::new(LEAF_1_ECX, XCR0_SUPPORTED);
    assert_eq!(features.host_cr4(0x2668), 0x2668 | CR4_OSXSAVE);
    assert_eq!(features.host_cr4(0x2668 | CR4_OSXSAVE), 0x2668 | CR4_OSXSAVE);
}

#[test]
fn nothing_published_hides_xsave() {
    assert!(!GuestFeatures::NONE.xsetbv());
    assert_eq!(GuestFeatures::NONE.hidden().leaf_1_ecx, CPUID_1_ECX_XSAVE);
}

#[test]
fn published_reads_back_what_was_published() {
    let features = GuestFeatures::new(LEAF_1_ECX, XCR0_SUPPORTED);
    publish(features);
    assert_eq!(published(), features);
    publish(GuestFeatures::NONE);
    assert_eq!(published(), GuestFeatures::NONE);
}
