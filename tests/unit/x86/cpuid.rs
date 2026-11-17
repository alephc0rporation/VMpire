//! Unit tests for `src/x86/cpuid.rs`.
//!
//! Compiled only by `cargo test-host`. That file includes this one with
//! `#[path]` as its child module `tests`, so `use super::*` reaches its
//! private items.

use super::*;

/// CPUID leaf 0 with `name` split into EBX, EDX, ECX as the CPU returns it.
fn leaf0(name: &[u8; 12]) -> CpuidResult {
    let word = |i: usize| u32::from_le_bytes(name[i..i + 4].try_into().unwrap());
    CpuidResult {
        eax: 0x16,
        ebx: word(0),
        edx: word(4),
        ecx: word(8),
    }
}

#[test]
fn the_vendor_string_picks_the_backend() {
    assert_eq!(Vendor::from_leaf0(leaf0(b"GenuineIntel")), Vendor::Intel);
    assert_eq!(Vendor::from_leaf0(leaf0(b"AuthenticAMD")), Vendor::Amd);
    assert_eq!(
        Vendor::from_leaf0(leaf0(b"HygonGenuine")),
        Vendor::Other(*b"HygonGenuine")
    );
}

#[test]
fn intels_leaf_0_as_the_cpu_returns_it() {
    // EBX "Genu", EDX "ineI", ECX "ntel".
    let raw = CpuidResult {
        eax: 0x16,
        ebx: 0x756E_6547,
        ecx: 0x6C65_746E,
        edx: 0x4965_6E69,
    };
    assert_eq!(Vendor::from_leaf0(raw), Vendor::Intel);
}
