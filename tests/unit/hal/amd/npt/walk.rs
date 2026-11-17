//! Unit tests for `src/hal/amd/npt/walk.rs`.
//!
//! Compiled only by `cargo test-host`. That file includes this one with
//! `#[path]` as its child module `tests`, so `use super::*` reaches its
//! private items.

use std::collections::HashMap;

use super::super::{NPT_RWX, NptEntry};
use super::*;
use crate::x86::mtrr::MemoryType::{Uncacheable as UC, WriteBack as WB};
use crate::x86::page_table::level::{Level, Pd, Pdpt, Pml4, Pt};

const RULES: EntryRules = EntryRules {
    physical_address_bits: 46,
    huge_pages: true,
    nx: true,
};

/// [`RULES`] on a CPU without 1 GiB pages, with the host's EFER.NXE clear.
const BASIC: EntryRules = EntryRules {
    huge_pages: false,
    nx: false,
    ..RULES
};

const ALL_LEVELS: [WalkLevel; 4] = [WalkLevel::Pml4, WalkLevel::Pdpt, WalkLevel::Pd, WalkLevel::Pt];

#[test]
fn every_entry_npt_build_makes_is_valid() {
    let table = NptEntry::<Pml4>::table(0x7eaa_9000).raw();
    for level in [WalkLevel::Pml4, WalkLevel::Pdpt, WalkLevel::Pd] {
        assert!(check_entry(table, level, RULES).is_empty(), "{level:?}");
    }
    for memory_type in [UC, WB] {
        let huge = NptEntry::<Pdpt>::huge_page(3 << 30, memory_type).raw();
        assert!(check_entry(huge, WalkLevel::Pdpt, RULES).is_empty());
        let large = NptEntry::<Pd>::large_page(0x7ec0_0000, memory_type).raw();
        assert!(check_entry(large, WalkLevel::Pd, RULES).is_empty());
        for access in [NPT_RWX, NPT_READ, NPT_READ | NPT_WRITE] {
            let page = NptEntry::<Pt>::page(0x7eab_0000, access, memory_type, RULES).raw();
            assert!(check_entry(page, WalkLevel::Pt, RULES).is_empty());
        }
    }
}

#[test]
fn an_entry_that_is_not_present_is_never_reserved() {
    for raw in [0, 0xFFFF_FFFF_FFFF_FFFE] {
        for level in ALL_LEVELS {
            assert!(check_entry(raw, level, BASIC).is_empty(), "{raw:#x} at {level:?}");
        }
    }
}

#[test]
fn the_reserved_bits_follow_the_level_and_the_cpu() {
    // Address bits from MAXPHYADDR (46) up.
    assert_eq!(
        check_entry(PRESENT | (1 << 46), WalkLevel::Pt, RULES).reserved_bits,
        1 << 46
    );
    // NX without EFER.NXE.
    assert_eq!(
        check_entry(PRESENT | NO_EXECUTE, WalkLevel::Pt, BASIC).reserved_bits,
        NO_EXECUTE
    );
    assert!(check_entry(PRESENT | NO_EXECUTE, WalkLevel::Pt, RULES).is_empty());
    // PS and bit 8 in a PML4 entry.
    assert_eq!(
        check_entry(PRESENT | PAGE_SIZE_BIT | NONLEAF_BIT_8, WalkLevel::Pml4, RULES).reserved_bits,
        PAGE_SIZE_BIT | NONLEAF_BIT_8
    );
    // A 2 MiB page's address bits 20:13; bit 12 is its PAT bit.
    let large = PRESENT | PAGE_SIZE_BIT | PAT_LARGE | (1 << 13);
    assert_eq!(check_entry(large, WalkLevel::Pd, RULES).reserved_bits, 1 << 13);
    // A 1 GiB page on a CPU without them.
    let huge = PRESENT | PAGE_SIZE_BIT;
    assert_eq!(check_entry(huge, WalkLevel::Pdpt, BASIC).reserved_bits, PAGE_SIZE_BIT);
    assert!(check_entry(huge, WalkLevel::Pdpt, RULES).is_empty());
}

/// Tables in a map from physical address to entry, for [`walk`]'s `read`.
fn reader(entries: &HashMap<u64, u64>) -> impl Fn(u64) -> Option<u64> + '_ {
    |address| entries.get(&address).copied()
}

/// PML4 at 0x1000, PDPT at 0x2000, PD at 0x3000, PT at 0x4000, mapping GPA
/// 0x7eb1_d000 through a PT, read-only, and GPA 0x4000_0000 by a UC 2 MiB
/// page.
fn tables() -> HashMap<u64, u64> {
    let gpa = 0x7eb1_d000u64;
    let mut entries = HashMap::new();
    entries.insert(0x1000, NptEntry::<Pml4>::table(0x2000).raw());
    entries.insert(
        0x2000 + 8 * Pdpt::index(gpa) as u64,
        NptEntry::<Pdpt>::table(0x3000).raw(),
    );
    entries.insert(0x2000 + 8, NptEntry::<Pdpt>::table(0x3000).raw());
    entries.insert(0x3000 + 8 * Pd::index(gpa) as u64, NptEntry::<Pd>::table(0x4000).raw());
    entries.insert(
        0x4000 + 8 * Pt::index(gpa) as u64,
        NptEntry::<Pt>::page(gpa, NPT_READ, WB, RULES).raw(),
    );
    entries.insert(
        0x3000 + 8 * Pd::index(0x4000_0000) as u64,
        NptEntry::<Pd>::large_page(0x4000_0000, UC).raw(),
    );
    entries
}

#[test]
fn a_walk_reaches_a_read_only_4k_page() {
    let entries = tables();
    let walked = walk(Ncr3::new(0x1000), 0x7eb1_d123, RULES, reader(&entries));
    assert_eq!(
        walked.end,
        WalkEnd::Mapped {
            hpa: 0x7eb1_d123,
            size: 0x1000,
            access: NPT_READ,
            memory_type: Some(WB),
        }
    );
    assert_eq!(walked.entries().count(), 4);
}

#[test]
fn a_walk_reaches_a_uc_2m_page() {
    let entries = tables();
    let walked = walk(Ncr3::new(0x1000), 0x4012_3456, RULES, reader(&entries));
    assert_eq!(
        walked.end,
        WalkEnd::Mapped {
            hpa: 0x4012_3456,
            size: 0x20_0000,
            access: NPT_RWX,
            memory_type: Some(UC),
        }
    );
}

#[test]
fn a_walk_stops_at_what_it_cannot_use() {
    let mut entries = tables();
    // An empty PD entry: not present.
    entries.insert(0x3000 + 8 * Pd::index(0x4020_0000) as u64, 0);
    let walked = walk(Ncr3::new(0x1000), 0x4020_0000, RULES, reader(&entries));
    assert_eq!(walked.end, WalkEnd::NotPresent);
    // A PDPT entry without U/S.
    entries.insert(0x2000 + 8, NptEntry::<Pdpt>::table(0x3000).raw() & !USER);
    let walked = walk(Ncr3::new(0x1000), 0x4000_0000, RULES, reader(&entries));
    assert_eq!(walked.end, WalkEnd::Supervisor);
    // A reserved bit.
    entries.insert(0x2000 + 8, NptEntry::<Pdpt>::table(0x3000).raw() | (1 << 50));
    let walked = walk(Ncr3::new(0x1000), 0x4000_0000, RULES, reader(&entries));
    assert_eq!(walked.end, WalkEnd::Reserved);
    // A table outside the ones given.
    let walked = walk(Ncr3::new(0x9000), 0, RULES, reader(&entries));
    assert_eq!(walked.end, WalkEnd::OutsideTables { table: 0x9000 });
    // Past 48 bits.
    let walked = walk(Ncr3::new(0x1000), 1 << 48, RULES, reader(&entries));
    assert_eq!(walked.end, WalkEnd::BeyondWalk);
}

#[test]
fn write_and_execute_are_what_every_level_allows() {
    let mut entries = tables();
    // The PD entry above the 2 MiB page's region read-only through its PDPT.
    entries.insert(
        0x2000 + 8,
        NptEntry::<Pdpt>::table(0x3000).raw() & !WRITABLE | NO_EXECUTE,
    );
    let walked = walk(Ncr3::new(0x1000), 0x4000_0000, RULES, reader(&entries));
    let WalkEnd::Mapped { access, .. } = walked.end else {
        panic!("{:?}", walked.end)
    };
    assert_eq!(access, NPT_READ);
}

#[test]
fn the_end_prints_for_the_report() {
    let mapped = WalkEnd::Mapped {
        hpa: 0x1000,
        size: 0x1000,
        access: NPT_READ | NPT_WRITE,
        memory_type: Some(WB),
    };
    assert_eq!(
        mapped.to_string(),
        "mapped to HPA 0x1000 by a 4 KiB page, WB, R=1 W=1 X=0"
    );
    assert_eq!(
        EntryFaults { reserved_bits: 0x100 }.to_string(),
        "reserved bits 0x100 set"
    );
}
