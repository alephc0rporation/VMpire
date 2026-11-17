//! Unit tests for `src/x86/identity_map.rs`.
//!
//! Compiled only by `cargo test-host`. That file includes this one with
//! `#[path]` as its child module `tests`, so `use super::*` reaches its
//! private items. The tests of the maps built from these (EPT's) are in
//! `tests/unit/vmx/ept.rs`.

use super::*;

/// MAXPHYADDR on the Hyper-V test VM: CPUID 0x80000008 EAX 0x302e.
const HYPER_V_BITS: u8 = 46;

#[test]
fn map_end_is_the_highest_region_rounded_up_to_a_gib() {
    // 2 GiB of RAM in two regions, plus firmware MMIO just below 4 GiB.
    let regions = [(0, 0x1000), (0x0100_0000, 0x7_F000), (0xFFC0_0000, 0x400)];
    assert_eq!(memory_map_end(regions.into_iter()), Ok(4 * SIZE_1G));
}

#[test]
fn map_end_on_a_gib_boundary_is_not_rounded_further() {
    assert_eq!(memory_map_end([(0, 0x8_0000)].into_iter()), Ok(2 * SIZE_1G));
}

#[test]
fn a_region_past_the_top_of_the_address_space_ends_at_u64_max() {
    assert_eq!(memory_map_end([(u64::MAX - 0xFFF, 2)].into_iter()), Ok(u64::MAX));
    assert_eq!(memory_map_end([(0, u64::MAX)].into_iter()), Ok(u64::MAX));
}

#[test]
fn empty_memory_map_is_refused() {
    assert_eq!(memory_map_end(core::iter::empty()), Err(MapError::EmptyMemoryMap));
    assert_eq!(memory_map_end([(0, 0)].into_iter()), Err(MapError::EmptyMemoryMap));
}

#[test]
fn the_address_space_ends_at_maxphyaddr_or_where_a_4_level_walk_stops() {
    assert_eq!(address_space_end(36), 1 << 36);
    assert_eq!(address_space_end(HYPER_V_BITS), 1 << 46);
    assert_eq!(address_space_end(48), 1 << 48);
    assert_eq!(address_space_end(52), 1 << 48);
}

#[test]
fn with_1_gib_pages_the_map_reaches_maxphyaddr_with_pds_for_the_memory_map() {
    // The Hyper-V test VM: 2 GiB of memory map, 46 bits.
    let layout = MapLayout::new(2 * SIZE_1G, HYPER_V_BITS, true).unwrap();
    assert_eq!(layout.pd_gibs(), 2);
    assert_eq!(layout.pd_end(), 2 * SIZE_1G);
    assert_eq!(layout.end(), 1 << 46);
    assert_eq!(layout.huge_gibs(), (1 << 16) - 2);
    assert_eq!(layout.pdpts(), 128);
    // 52 bits: the 4-level walk stops at 48, 512 PDPTs.
    let layout = MapLayout::new(2 * SIZE_1G, 52, true).unwrap();
    assert_eq!((layout.end(), layout.pdpts()), (1 << 48, 512));
    // 36 bits, the smallest: one PDPT, partly filled.
    let layout = MapLayout::new(2 * SIZE_1G, 36, true).unwrap();
    assert_eq!((layout.end(), layout.pdpts(), layout.huge_gibs()), (1 << 36, 1, 62));
}

#[test]
fn without_1_gib_pages_the_pds_reach_the_memory_map_or_4_gib_and_no_further() {
    let layout = MapLayout::new(2 * SIZE_1G, HYPER_V_BITS, false).unwrap();
    assert_eq!(
        (layout.pd_gibs(), layout.end(), layout.huge_gibs()),
        (4, 4 * SIZE_1G, 0)
    );
    let layout = MapLayout::new(8 * SIZE_1G, HYPER_V_BITS, false).unwrap();
    assert_eq!((layout.pd_gibs(), layout.end()), (8, 8 * SIZE_1G));
    assert_eq!(layout.pdpts(), 1);
}

#[test]
fn a_memory_map_past_the_address_space_is_refused() {
    for huge_pages in [false, true] {
        assert_eq!(
            MapLayout::new(1 << 37, 36, huge_pages),
            Err(MapError::RangeTooLarge {
                end: 1 << 37,
                limit: 1 << 36
            })
        );
        assert_eq!(
            MapLayout::new(u64::MAX, HYPER_V_BITS, huge_pages),
            Err(MapError::RangeTooLarge {
                end: u64::MAX,
                limit: 1 << 46
            })
        );
    }
}

#[test]
fn spare_pts_cover_each_variable_range_and_the_fixed_ranges() {
    assert_eq!(spare_pts(MtrrCapabilities::NONE), 0);
    // Hyper-V: eight variable ranges and the fixed ones.
    assert_eq!(spare_pts(MtrrCapabilities::new((1 << 8) | 8, 46).unwrap()), 9);
    assert_eq!(spare_pts(MtrrCapabilities::new(10, 46).unwrap()), 10);
}

#[test]
fn the_free_stack_hands_out_the_spare_pts_in_address_order() {
    let mut stack = [0; 5];
    let mut len = fill_free_stack(&mut stack, 2);
    assert_eq!(len, 3);
    let mut free = FreePts {
        stack: &mut stack,
        len: &mut len,
    };
    assert_eq!(
        [free.take(), free.take(), free.take(), free.take()],
        [Some(2), Some(3), Some(4), None]
    );
    free.give_back(3);
    assert_eq!(free.take(), Some(3));
}

#[test]
fn only_memory_the_firmware_or_an_os_uses_as_memory_is_ram() {
    use boot::MemoryType as Type;
    for ram in [
        Type::LOADER_CODE,
        Type::LOADER_DATA,
        Type::BOOT_SERVICES_CODE,
        Type::BOOT_SERVICES_DATA,
        Type::RUNTIME_SERVICES_CODE,
        Type::RUNTIME_SERVICES_DATA,
        Type::CONVENTIONAL,
        Type::ACPI_RECLAIM,
        Type::ACPI_NON_VOLATILE,
        Type::PERSISTENT_MEMORY,
    ] {
        assert!(is_ram(ram), "{ram:?}");
    }
    for other in [
        Type::RESERVED,
        Type::UNUSABLE,
        Type::MMIO,
        Type::MMIO_PORT_SPACE,
        Type::PAL_CODE,
        Type::UNACCEPTED,
        Type::custom(0x8000_0000),
    ] {
        assert!(!is_ram(other), "{other:?}");
    }
}

#[test]
fn the_ram_map_sorts_and_merges_the_ram_regions_and_drops_the_rest() {
    use boot::MemoryType as Type;
    let regions = [
        (0x10_0000, 0x100, Type::BOOT_SERVICES_DATA),
        (0, 0xA0, Type::CONVENTIONAL),
        // Touches the first region: one range with it.
        (0x20_0000, 0x200, Type::LOADER_DATA),
        (0xFEC0_0000, 1, Type::MMIO),
        (0x5000_0000, 0x10, Type::RESERVED),
        // Empty: ignored.
        (0x30_0000, 0, Type::CONVENTIONAL),
        // Overlaps the range the first and third make: merged into it.
        (0x38_0000, 0x100, Type::ACPI_NON_VOLATILE),
    ];
    let mut ranges = [(0, 0); 7];
    let ram = RamMap::collect(regions.into_iter(), &mut ranges);
    assert_eq!(ram.ranges(), [(0, 0xA_0000), (0x10_0000, 0x48_0000)]);
}

#[test]
fn the_ram_map_says_how_much_of_a_block_is_ram() {
    let ram = RamMap {
        ranges: &[(0, 0xA_0000), (0x10_0000, 0x7FF0_0000)],
    };
    assert!(ram.contains(0x9_F000));
    assert!(!ram.contains(0xA_0000));
    assert!(ram.contains(0x10_0000));
    assert!(!ram.contains(0x7FF0_0000));
    assert_eq!(ram.coverage(0, SIZE_2M), RamCoverage::Part);
    assert_eq!(ram.coverage(SIZE_2M, SIZE_2M), RamCoverage::All);
    assert_eq!(ram.coverage(0x7FC0_0000, SIZE_2M), RamCoverage::All);
    assert_eq!(ram.coverage(0x7FE0_0000, SIZE_2M), RamCoverage::Part);
    assert_eq!(ram.coverage(0x8000_0000, SIZE_2M), RamCoverage::None);
    // A block inside the gap between two ranges.
    assert_eq!(ram.coverage(0xA_0000, 0x1000), RamCoverage::None);
}
