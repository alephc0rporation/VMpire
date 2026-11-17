//! Unit tests for `src/hal/amd/guest_mtrrs.rs`.
//!
//! Compiled only by `cargo test-host`. The copy is one static, never
//! published on the host (`publish` needs the tables of a real
//! `Npt::build`), so a single test covers it, as Intel's.

use super::*;

#[test]
fn the_copy_is_refused_while_lent_out_and_before_it_is_published() {
    let def_type = MtrrCapabilities::new(8, 46).unwrap().msr(0x2FF).unwrap();
    GUEST_MTRRS.in_use.store(true, Ordering::Relaxed);
    assert_eq!(with(|_| ()), Err(Unavailable::InUse));
    assert_eq!(read(def_type), Err(MtrrAccessFailure::InUse));
    assert_eq!(capabilities(), MtrrCapabilities::NONE);
    GUEST_MTRRS.in_use.store(false, Ordering::Relaxed);

    assert_eq!(with(|_| ()), Err(Unavailable::NotPublished));
    assert_eq!(read(def_type), Err(MtrrAccessFailure::NotPublished));
    assert!(registers().is_none());
    assert!(!GUEST_MTRRS.in_use.load(Ordering::Relaxed));
}
