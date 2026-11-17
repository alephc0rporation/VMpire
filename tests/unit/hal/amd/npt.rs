//! Unit tests for `src/hal/amd/npt/mod.rs`.
//!
//! Compiled only by `cargo test-host`. That file includes this one with
//! `#[path]` as its child module `tests`, so `use super::*` reaches its
//! private items. They mirror `tests/unit/vmx/ept.rs`: the same 2 GiB map,
//! the same MTRRs, the same protected page, on NPT entries.

use super::*;
use crate::hal::amd::vmcb::Vmcb;
use crate::x86::mtrr::MemoryType::{
    Uncacheable as UC, WriteBack as WB, WriteCombining as WC, WriteProtected as WP, WriteThrough as WT,
};
use crate::x86::mtrr::tests::{enabled, range};

/// A CPU like the Hyper-V test VM's, on AMD: MAXPHYADDR 46, 1 GiB pages,
/// and the host running with EFER.NXE.
const RULES: EntryRules = EntryRules {
    physical_address_bits: 46,
    huge_pages: true,
    nx: true,
};

/// [`RULES`] without EFER.NXE: execution cannot be taken away.
const NO_NX: EntryRules = EntryRules { nx: false, ..RULES };

#[test]
fn the_host_pat_has_an_entry_for_every_mtrr_type() {
    for memory_type in [UC, WC, WT, WP, WB] {
        assert_eq!(pat_type(pat_index(memory_type)), Some(memory_type), "{memory_type:?}");
    }
    // Entries 0-3 are the power-on PAT's: WB, WT, UC-, UC.
    assert_eq!(HOST_PAT & 0xFFFF_FFFF, 0x0007_0406);
    assert_eq!(pat_type(2), None);
    assert_eq!(pat_type(6), None);
}

#[test]
fn a_table_entry_is_present_writable_user_and_accessed() {
    let entry = NptEntry::<Pml4>::table(0x7eaa_9000);
    assert_eq!(entry.raw(), 0x7eaa_9000 | 0b10_0111);
    assert_eq!(entry_access(entry.raw()), NPT_RWX);
}

#[test]
fn a_large_page_carries_its_type_in_pwt_pcd_and_bit_12() {
    // WB is PAT entry 0: no type bits.
    let wb = NptEntry::<Pd>::large_page(0x7ec0_0000, WB).raw();
    assert_eq!(wb, 0x7ec0_0000 | 0b1110_0111);
    // UC is entry 3: PWT and PCD. WC is entry 4: bit 12 alone.
    assert_eq!(
        NptEntry::<Pd>::large_page(0, UC).raw() & (PWT | PCD | PAT_LARGE),
        PWT | PCD
    );
    assert_eq!(
        NptEntry::<Pdpt>::huge_page(0, WC).raw() & (PWT | PCD | PAT_LARGE),
        PAT_LARGE
    );
    for memory_type in [UC, WC, WT, WP, WB] {
        assert_eq!(
            NptEntry::<Pd>::large_page(0, memory_type).mapping(),
            RegionMapping::LargePage(memory_type)
        );
        assert_eq!(
            NptEntry::<Pdpt>::huge_page(0, memory_type).mapping(),
            RegionMapping::HugePage(memory_type)
        );
    }
}

#[test]
fn a_pt_entry_carries_its_type_in_pwt_pcd_and_bit_7() {
    let wp = NptEntry::<Pt>::page(0x1000, NPT_RWX, WP, RULES);
    // WP is entry 5: PWT and bit 7.
    assert_eq!(wp.raw() & (PWT | PCD | PAT_4K), PWT | PAT_4K);
    assert_eq!(wp.memory_type(), Some(WP));
    assert_eq!(wp.with_memory_type(WT).memory_type(), Some(WT));
    assert_eq!(wp.with_memory_type(WT).access(), NPT_RWX);
}

#[test]
fn access_maps_to_present_writable_and_nx() {
    let page = |access, rules| NptEntry::<Pt>::page(0x1000, access, WB, rules);
    assert_eq!(
        page(NPT_RWX, RULES).raw() & (PRESENT | WRITABLE | USER | NO_EXECUTE),
        PRESENT | WRITABLE | USER
    );
    assert_eq!(page(NPT_READ, RULES).raw() & (WRITABLE | NO_EXECUTE), NO_EXECUTE);
    assert_eq!(page(NPT_READ | NPT_EXECUTE, RULES).access(), NPT_READ | NPT_EXECUTE);
    assert_eq!(page(NPT_READ | NPT_WRITE, RULES).access(), NPT_READ | NPT_WRITE);
    // No read is not present: nothing else counts.
    assert_eq!(page(NPT_WRITE | NPT_EXECUTE, RULES).raw() & (PRESENT | WRITABLE), 0);
    // Without NX, a page that may be read may be executed.
    assert_eq!(page(NPT_READ, NO_NX).access(), NPT_READ | NPT_EXECUTE);
    assert_eq!(page(NPT_READ, NO_NX).raw() & NO_EXECUTE, 0);
}

#[test]
fn with_access_keeps_the_address_and_the_type() {
    let page = NptEntry::<Pt>::page(0x7eb1_d000, NPT_RWX, UC, RULES);
    let read_only = page.with_access(NPT_READ, RULES);
    assert_eq!(read_only.raw() & ADDRESS_MASK, 0x7eb1_d000);
    assert_eq!(read_only.memory_type(), Some(UC));
    assert_eq!(read_only.with_access(NPT_RWX, RULES), page);
}

#[test]
fn ncr3_is_the_pml4_with_wb_type_bits() {
    let ncr3 = Ncr3::new(0x7eaa_9fff);
    assert_eq!(ncr3.raw(), 0x7eaa_9000);
    assert_eq!(ncr3.pml4(), 0x7eaa_9000);
}

#[test]
fn support_needs_nested_paging_and_flushes_by_asid_when_it_can() {
    assert_eq!(
        check_support(false, true, 46, true, true),
        Err(NptError::NestedPagingUnsupported)
    );
    let support = check_support(true, true, 46, true, true).unwrap();
    assert_eq!(support.flush, FlushType::Asid);
    assert_eq!(support.rules, RULES);
    assert_eq!(
        check_support(true, false, 46, true, true).unwrap().flush,
        FlushType::All
    );
    assert_eq!(FlushType::Asid.tlb_control(), 7);
    assert_eq!(FlushType::All.tlb_control(), 1);
}

#[test]
fn a_map_error_becomes_its_npt_error() {
    assert_eq!(NptError::from(MapError::EmptyMemoryMap), NptError::EmptyMemoryMap);
    assert_eq!(
        NptError::from(MapError::RangeTooLarge { end: 2, limit: 1 }),
        NptError::RangeTooLarge { end: 2, limit: 1 }
    );
}

/// MTRRs of a 2 GiB machine, as EPT's tests: WB up to 2 GiB, UC above, a
/// 1 MiB UC hole at 0x7FF0_0000 and a UC 2 MiB region at 0x4000_0000.
fn test_mtrrs() -> Mtrrs {
    Mtrrs::decode(
        enabled(UC),
        None,
        &[
            range(0, 2 * SIZE_1G, WB),
            range(0x7FF0_0000, 0x10_0000, UC),
            range(0x4000_0000, SIZE_2M, UC),
        ],
    )
    .unwrap()
}

fn all_write_back() -> Mtrrs {
    Mtrrs::decode(enabled(UC), None, &[range(0, 2 * SIZE_1G, WB)]).unwrap()
}

fn write_back_with(more: &[(u64, u64)]) -> Mtrrs {
    let mut variable = vec![range(0, 2 * SIZE_1G, WB)];
    variable.extend_from_slice(more);
    Mtrrs::decode(enabled(UC), None, &variable).unwrap()
}

/// All of memory RAM: the memory types are the MTRRs' alone.
const ALL_RAM: RamMap<'static> = RamMap {
    ranges: &[(0, u64::MAX)],
};

/// Where [`TypeMap`] puts its PTs.
const PTS_ADDRESS: u64 = 0x10_0000;
/// The protected page: in region 0x3F5 (0x7EA0_0000), index 0x11D of its PT.
const PROTECTED: u64 = 0x7eb1_d000;
/// A shadow HPA for the hiding tests, outside the 2 GiB map.
const SHADOW: u64 = 0xDEAD_0000;

/// A 2 GiB map built by [`fill_identity_map`], with what [`retype`] and
/// [`hide_page`] need, in vectors.
struct TypeMap {
    pml4: Vec<NptEntry<Pml4>>,
    pdpts: Vec<NptEntry<Pdpt>>,
    pds: Vec<NptEntry<Pd>>,
    /// Every PT, the spare ones last.
    pts: Vec<NptEntry<Pt>>,
    free: Vec<u32>,
    free_len: usize,
    ram: RamMap<'static>,
    rules: EntryRules,
}

impl TypeMap {
    /// The map `mtrrs` give, all of it RAM, with [`PROTECTED`] read-only and
    /// `spare` PTs set aside.
    fn new(mtrrs: &Mtrrs, spare: usize) -> Self {
        Self::with(mtrrs, spare, ALL_RAM, RULES)
    }

    fn with(mtrrs: &Mtrrs, spare: usize, ram: RamMap<'static>, rules: EntryRules) -> Self {
        let types = MapTypes { mtrrs, ram };
        let split = split_regions(&types, 2, &[PROTECTED]);
        let mut pml4 = vec![NptEntry::EMPTY; ENTRIES];
        let mut pdpts = vec![NptEntry::EMPTY; ENTRIES];
        let mut pds = vec![NptEntry::EMPTY; 2 * ENTRIES];
        let mut pts = vec![NptEntry::EMPTY; (split + spare) * ENTRIES];
        fill_identity_map(
            Tables {
                pml4: &mut pml4,
                pdpts: &mut pdpts,
                pds: &mut pds,
                pts: &mut pts[..split * ENTRIES],
                pdpts_address: 0x1000,
                pds_address: 0x2000,
                pts_address: PTS_ADDRESS,
            },
            MapLayout { pd_gibs: 2, gibs: 2 },
            &types,
            &[PROTECTED],
            rules,
        );
        let mut free = vec![0; split + spare];
        let free_len = fill_free_stack(&mut free, split);
        Self {
            pml4,
            pdpts,
            pds,
            pts,
            free,
            free_len,
            ram,
            rules,
        }
    }

    fn retype(&mut self, mtrrs: &Mtrrs) -> Retyped {
        let (pts, _) = self.pts.as_chunks_mut::<ENTRIES>();
        let mut tables = Retype {
            pds: &mut self.pds,
            pts,
            pts_address: PTS_ADDRESS,
            free: FreePts {
                stack: &mut self.free,
                len: &mut self.free_len,
            },
            protected_regions: [(PROTECTED / SIZE_2M) as usize, usize::MAX],
            rules: self.rules,
        };
        retype(&mut tables, &MapTypes { mtrrs, ram: self.ram })
    }

    fn hide(&mut self, page: u64, code: bool) -> Hidden {
        let mtrrs = all_write_back();
        let (pts, _) = self.pts.as_chunks_mut::<ENTRIES>();
        let mut tables = Hiding {
            pds: &mut self.pds,
            pts,
            free: FreePts {
                stack: &mut self.free,
                len: &mut self.free_len,
            },
            pts_address: PTS_ADDRESS,
            shadow: SHADOW,
            types: MapTypes {
                mtrrs: &mtrrs,
                ram: self.ram,
            },
            rules: self.rules,
        };
        let mut hidden = Hidden::default();
        hide_page(&mut tables, page, code, &mut hidden).unwrap();
        hidden
    }

    /// PT number `index`.
    fn pt(&self, index: usize) -> &[NptEntry<Pt>] {
        &self.pts[index * ENTRIES..][..ENTRIES]
    }
}

#[test]
fn identity_map_links_the_levels() {
    let map = TypeMap::new(&test_mtrrs(), 0);
    assert_eq!(map.pml4[0], NptEntry::table(0x1000));
    assert_eq!(map.pml4[1], NptEntry::EMPTY);
    assert_eq!(map.pdpts[0], NptEntry::table(0x2000));
    assert_eq!(map.pdpts[1], NptEntry::table(0x3000));
    assert_eq!(map.pdpts[2], NptEntry::EMPTY);
}

#[test]
fn regions_of_one_memory_type_are_2m_pages_of_that_type() {
    let map = TypeMap::new(&test_mtrrs(), 0);
    assert_eq!(map.pds[0], NptEntry::large_page(0, WB));
    assert_eq!(map.pds[0x200], NptEntry::large_page(0x4000_0000, UC));
    assert_eq!(map.pds[0x201], NptEntry::large_page(0x4020_0000, WB));
}

#[test]
fn the_protected_region_and_mixed_regions_get_a_pt_each_and_only_the_page_is_read_only() {
    let map = TypeMap::new(&test_mtrrs(), 0);
    assert_eq!(map.pds[0x3F5], NptEntry::table(PTS_ADDRESS));
    assert_eq!(map.pds[0x3FF], NptEntry::table(PTS_ADDRESS + 0x1000));
    assert_eq!(map.pt(0)[0x11D], NptEntry::page(PROTECTED, NPT_READ, WB, RULES));
    assert_eq!(map.pt(0)[0x11C], NptEntry::page(0x7eb1_c000, NPT_RWX, WB, RULES));
    // The mixed region: WB, then the UC hole from 0x7FF0_0000.
    assert_eq!(map.pt(1)[0], NptEntry::page(0x7FE0_0000, NPT_RWX, WB, RULES));
    assert_eq!(map.pt(1)[0x100], NptEntry::page(0x7FF0_0000, NPT_RWX, UC, RULES));
}

#[test]
fn memory_that_is_not_ram_is_uc_whatever_the_mtrrs_say() {
    let ram = RamMap {
        ranges: &[(0, 0xA_0000), (0x10_0000, 0x6000_0000)],
    };
    let map = TypeMap::with(&all_write_back(), 0, ram, RULES);
    assert_eq!(map.pds[0], NptEntry::table(PTS_ADDRESS));
    assert_eq!(map.pt(0)[0x9F], NptEntry::page(0x9_F000, NPT_RWX, WB, RULES));
    assert_eq!(map.pt(0)[0xA0], NptEntry::page(0xA_0000, NPT_RWX, UC, RULES));
    assert_eq!(map.pds[0x2FF], NptEntry::large_page(0x5FE0_0000, WB));
    assert_eq!(map.pds[0x300], NptEntry::large_page(0x6000_0000, UC));
}

#[test]
fn the_same_types_change_nothing() {
    let mut map = TypeMap::new(&test_mtrrs(), 1);
    let (pds, pts) = (map.pds.clone(), map.pts.clone());
    assert_eq!(map.retype(&test_mtrrs()), Retyped::default());
    assert_eq!((map.pds, map.pts, map.free_len), (pds, pts, 1));
}

#[test]
fn a_region_of_one_type_changes_only_its_type_bits() {
    let mut map = TypeMap::new(&all_write_back(), 1);
    let before = map.pds.clone();
    let retyped = map.retype(&write_back_with(&[range(SIZE_2M, SIZE_2M, UC)]));
    assert_eq!((retyped.regions, retyped.span), (1, Some((SIZE_2M, 2 * SIZE_2M))));
    assert_eq!(map.pds[1], NptEntry::large_page(SIZE_2M, UC));
    // WB is PAT entry 0, UC entry 3: only PWT and PCD differ.
    assert_eq!(map.pds[1].raw() ^ before[1].raw(), PWT | PCD);
}

#[test]
fn a_region_that_comes_to_mix_types_takes_a_spare_pt_and_gives_it_back() {
    let mut map = TypeMap::new(&all_write_back(), 2);
    let retyped = map.retype(&write_back_with(&[range(SIZE_2M, 0x1000, UC)]));
    assert_eq!((retyped.regions, retyped.split, retyped.merged), (1, 1, 0));
    assert_eq!(map.pds[1], NptEntry::table(PTS_ADDRESS + 0x1000));
    assert_eq!(map.pt(1)[0], NptEntry::page(SIZE_2M, NPT_RWX, UC, RULES));
    assert_eq!(map.pt(1)[1], NptEntry::page(SIZE_2M + 0x1000, NPT_RWX, WB, RULES));
    assert_eq!(map.free_len, 1);
    let retyped = map.retype(&all_write_back());
    assert_eq!((retyped.split, retyped.merged), (0, 1));
    assert_eq!(map.pds[1], NptEntry::large_page(SIZE_2M, WB));
    assert_eq!(map.free_len, 2);
}

#[test]
fn without_a_spare_pt_a_mixed_region_is_one_uc_page() {
    let mut map = TypeMap::new(&all_write_back(), 0);
    let retyped = map.retype(&write_back_with(&[range(SIZE_2M, 0x1000, UC)]));
    assert_eq!((retyped.split, retyped.uncached), (0, 1));
    assert_eq!(map.pds[1], NptEntry::large_page(SIZE_2M, UC));
}

#[test]
fn the_protected_region_keeps_its_pt_and_its_access() {
    let mut map = TypeMap::new(&all_write_back(), 0);
    let region = PROTECTED & !(SIZE_2M - 1);
    map.retype(&write_back_with(&[range(region, SIZE_2M, UC)]));
    assert_eq!(map.pds[0x3F5], NptEntry::table(PTS_ADDRESS));
    assert_eq!(map.pt(0)[0x11D], NptEntry::page(PROTECTED, NPT_READ, UC, RULES));
    assert_eq!(map.pt(0)[0], NptEntry::page(region, NPT_RWX, UC, RULES));
}

#[test]
fn hiding_a_page_splits_its_region_and_redirects_it_to_the_shadow() {
    let mut map = TypeMap::new(&all_write_back(), 1);
    let region = 2 * SIZE_2M;
    let page = region + 5 * 0x1000;
    let hidden = map.hide(page, false);
    assert_eq!((hidden.data_pages, hidden.regions_split), (1, 1));
    assert_eq!(map.pds[2], NptEntry::table(PTS_ADDRESS + PAGE_SIZE as u64));
    assert_eq!(map.pt(1)[5], NptEntry::page(SHADOW, NPT_READ | NPT_WRITE, WB, RULES));
    assert_eq!(map.pt(1)[0], NptEntry::page(region, NPT_RWX, WB, RULES));
    assert_eq!(map.free_len, 0);
}

#[test]
fn a_hidden_code_page_is_kept_read_only_and_executable() {
    let mut map = TypeMap::new(&all_write_back(), 1);
    let page = 2 * SIZE_2M + 5 * 0x1000;
    let hidden = map.hide(page, true);
    assert_eq!(hidden.code_pages, 1);
    assert_eq!(map.pt(1)[5], NptEntry::page(page, NPT_READ | NPT_EXECUTE, WB, RULES));
}

#[test]
fn a_hidden_region_is_never_collapsed_by_retype() {
    for rules in [RULES, NO_NX] {
        let mut map = TypeMap::with(&all_write_back(), 1, ALL_RAM, rules);
        map.hide(2 * SIZE_2M + 5 * 0x1000, false);
        let retyped = map.retype(&all_write_back());
        assert_eq!(retyped.merged, 0, "{rules:?}");
        assert_eq!(map.pds[2], NptEntry::table(PTS_ADDRESS + PAGE_SIZE as u64));
    }
}

#[test]
fn granting_the_protected_write_makes_it_writable_and_asks_for_a_flush() {
    let mut map = TypeMap::new(&all_write_back(), 0);
    let page = ProtectedPage {
        gpa: PROTECTED,
        entry: &raw mut map.pts[0x11D],
        flush: FlushType::Asid,
        rules: RULES,
    };
    let mut vmcb = Box::new(Vmcb::zeroed());
    // SAFETY: `entry` points into `map.pts`, which outlives the call.
    let flush = unsafe { page.grant_write(&mut vmcb.control) };
    assert_eq!(flush, FlushType::Asid);
    assert_eq!(vmcb.control.tlb_control, tlb_control::FLUSH_ASID);
    assert_eq!(
        map.pts[0x11D],
        NptEntry::page(PROTECTED, NPT_READ | NPT_WRITE, WB, RULES)
    );
}

#[test]
fn pd_and_pdpt_entries_read_back_how_they_map() {
    assert_eq!(NptEntry::<Pd>::table(0x1000).mapping(), RegionMapping::Pt);
    assert_eq!(NptEntry::<Pd>::EMPTY.mapping(), RegionMapping::Invalid);
    // A PDPT entry pointing to a PD is no 1 GiB page.
    assert_eq!(NptEntry::<Pdpt>::table(0x1000).mapping(), RegionMapping::Invalid);
    // A UC- type (PAT entry 2: PCD alone) is not one Aleph0 makes.
    let uc_minus = NptEntry::<Pd>::from_raw(NptEntry::<Pd>::large_page(0, WB).raw() | PCD);
    assert_eq!(uc_minus.mapping(), RegionMapping::Invalid);
}

#[test]
fn region_runs_merge_pd_entries_that_map_alike() {
    let map = TypeMap::new(&test_mtrrs(), 0);
    let runs: Vec<_> = region_runs(&map.pds, 0).collect();
    assert_eq!(
        runs[..3],
        [
            RegionRun {
                start: 0,
                end: 0x4000_0000,
                mapping: RegionMapping::LargePage(WB)
            },
            RegionRun {
                start: 0x4000_0000,
                end: 0x4020_0000,
                mapping: RegionMapping::LargePage(UC)
            },
            RegionRun {
                start: 0x4020_0000,
                end: 0x7EA0_0000,
                mapping: RegionMapping::LargePage(WB)
            },
        ]
    );
}
