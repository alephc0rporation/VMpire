//! Unit tests for `src/hal/amd/svm.rs`.
//!
//! Compiled only by `cargo test-host`. That file includes this one with
//! `#[path]` as its child module `tests`, so `use super::*` reaches its
//! private items. These cover the pure capability decode; the enabling steps
//! run only on a real AMD CPU.

use super::*;

/// A supported CPU: SVM, NPT, SVML, NRIPS, 64 ASIDs, VM_CR clear.
fn supported() -> SvmSupport {
    SvmSupport {
        svm: true,
        features: SVM_FEATURE_NP | SVM_FEATURE_SVML | SVM_FEATURE_NRIPS,
        vm_cr: 0,
        asids: 64,
    }
}

#[test]
fn a_supported_cpu_is_usable() {
    assert_eq!(supported().usable(), Ok(()));
}

#[test]
fn no_cpuid_svm_bit_means_not_supported() {
    let support = SvmSupport {
        svm: false,
        ..supported()
    };
    assert_eq!(support.usable(), Err(SvmUnusable::NotSupported));
}

#[test]
fn svmdis_locked_is_disabled_by_bios() {
    let support = SvmSupport {
        vm_cr: VM_CR_SVMDIS | VM_CR_LOCK,
        ..supported()
    };
    assert_eq!(support.usable(), Err(SvmUnusable::DisabledLocked));
}

#[test]
fn svmdis_without_the_svml_feature_does_not_lock() {
    // Without SVML, the LOCK bit is not honored, so SVMDIS is not locking.
    let support = SvmSupport {
        features: SVM_FEATURE_NP,
        vm_cr: VM_CR_SVMDIS | VM_CR_LOCK,
        ..supported()
    };
    assert_eq!(support.usable(), Ok(()));
}

#[test]
fn svmdis_set_but_unlocked_is_still_usable() {
    // SVMDIS without LOCK can be cleared, so the CPU is usable.
    let support = SvmSupport {
        vm_cr: VM_CR_SVMDIS,
        ..supported()
    };
    assert_eq!(support.usable(), Ok(()));
}

#[test]
fn no_asids_is_unusable() {
    let support = SvmSupport {
        asids: 0,
        ..supported()
    };
    assert_eq!(support.usable(), Err(SvmUnusable::NoAsids));
}

#[test]
fn feature_queries_read_the_edx_flags() {
    assert!(supported().nested_paging());
    assert!(supported().next_rip_save());
    let no_npt = SvmSupport {
        features: SVM_FEATURE_NRIPS,
        ..supported()
    };
    assert!(!no_npt.nested_paging());
    assert!(no_npt.next_rip_save());
}

#[test]
fn the_svme_bit_is_efer_bit_12() {
    assert_eq!(EFER_SVME, 1 << 12);
    assert_eq!(VM_HSAVE_PA, 0xC001_0117);
    assert_eq!(VM_CR, 0xC001_0114);
}
