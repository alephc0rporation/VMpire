//! What the guest's identity map looks like, whatever the nested page
//! tables are made of: EPT on Intel (`hal::intel::ept`), NPT on AMD
//! (`arch::amd::npt`). Only the entries differ between the two; how far the
//! map reaches, which memory is RAM, the memory type each page gets, how
//! many PTs to set aside, and the bookkeeping of free PTs are the same, and
//! live here. The backend code names below (`Ept::build`, `retype`, ...) are
//! EPT's; NPT has the same functions.

#![allow(
    dead_code,
    reason = "only the AMD backend uses this so far; Intel's EPT keeps its own"
)]

use core::fmt;

use uefi::boot;

use crate::x86::mtrr::{MemoryType, MtrrCapabilities, Mtrrs};
use crate::x86::page::PAGE_SIZE;
use crate::x86::page_table::{ENTRIES, SIZE_1G, SIZE_2M};

/// Why the map's extent could not be worked out.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MapError {
    /// The memory map has no memory in it.
    EmptyMemoryMap,
    /// The memory map reaches `end`, past `limit`, the end of the
    /// guest-physical address space ([`address_space_end`]).
    RangeTooLarge { end: u64, limit: u64 },
}

/// Address bits a 4-level nested walk (EPT or NPT) translates: 9 per level and 12 of page
/// offset. A GPA needs no more, whatever MAXPHYADDR says.
pub const WALK_ADDRESS_BITS: u8 = 48;

/// End of the 32-bit address space, below which the Local APIC, the IOAPIC
/// and 32-bit PCI BARs live. Without 1 GiB pages the map reaches at least
/// this far ([`MapLayout::new`]).
pub const LOW_DEVICE_END: u64 = 4 * SIZE_1G;

/// How many pages `Ept::build` can map read-only and publish for the exit
/// handler at once: the guest's data page and its NMI-delivery stack page.
pub const MAX_PROTECTED: usize = 2;

/// How a PD entry maps its 2 MiB region, or a PDPT entry its GiB: see the
/// two `mapping` methods of each backend's entry type (`hal::intel::ept::EptEntry`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RegionMapping {
    /// One 2 MiB page of this memory type: a PD entry.
    LargePage(MemoryType),
    /// One 1 GiB page of this memory type: a PDPT entry.
    HugePage(MemoryType),
    /// A PT of 4 KiB pages, each with its own type and access: a PD entry.
    Pt,
    /// Neither: not present, a page whose type bits hold a reserved value,
    /// or, read as a 1 GiB page, a PDPT entry pointing to a PD. `Ept::build`
    /// makes no such entry where it is read this way.
    Invalid,
}

impl fmt::Display for RegionMapping {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::LargePage(memory_type) => write!(f, "2 MiB pages, {}", memory_type.name()),
            Self::HugePage(memory_type) => write!(f, "1 GiB pages, {}", memory_type.name()),
            Self::Pt => f.write_str("4 KiB pages (a PT each)"),
            Self::Invalid => f.write_str("invalid (not present or reserved type)"),
        }
    }
}

/// Consecutive parts of the map mapped the same way: see
/// `TypeTables::region_runs`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RegionRun {
    pub start: u64,
    pub end: u64,
    pub mapping: RegionMapping,
}

/// The `mappings` of consecutive blocks of `size` bytes, from block number
/// `first`, with neighbours mapped alike merged into one run.
pub fn runs(mappings: impl Iterator<Item = RegionMapping>, first: usize, size: u64) -> impl Iterator<Item = RegionRun> {
    let mut blocks = mappings
        .enumerate()
        .map(move |(i, mapping)| (first + i, mapping))
        .peekable();
    core::iter::from_fn(move || {
        let (first, mapping) = blocks.next()?;
        let mut last = first;
        while let Some((next, _)) = blocks.next_if(|&(_, next_mapping)| next_mapping == mapping) {
            last = next;
        }
        Some(RegionRun {
            start: first as u64 * size,
            end: (last as u64 + 1) * size,
            mapping,
        })
    })
}

/// End of memory-map regions `(start, pages)`: the highest address any
/// region reaches, rounded up to 1 GiB, so every PD of [`MapLayout`] is
/// full. Every region counts, whatever its type, so memory-mapped I/O the
/// firmware reports is in it too, UC ([`MapTypes`]).
///
/// A region reaching past the top of the 64-bit address space gives
/// `u64::MAX`, past any address space, which [`MapLayout::new`] refuses.
pub fn memory_map_end(regions: impl Iterator<Item = (u64, u64)>) -> Result<u64, MapError> {
    let highest = regions
        .map(|(start, pages)| start.saturating_add(pages.saturating_mul(PAGE_SIZE as u64)))
        .max()
        .filter(|&end| end > 0)
        .ok_or(MapError::EmptyMemoryMap)?;
    Ok(highest.checked_next_multiple_of(SIZE_1G).unwrap_or(u64::MAX))
}

/// End of the guest-physical address space EPT maps, for a CPU whose
/// physical addresses have `physical_address_bits` bits (MAXPHYADDR): every
/// address the guest can reach, as far as a 4-level walk translates.
pub fn address_space_end(physical_address_bits: u8) -> u64 {
    1 << physical_address_bits.min(WALK_ADDRESS_BITS)
}

/// How `Ept::build` lays out the identity map, GiB by GiB, from GPA 0:
///
/// ```text
/// 0 ── PDs (2 MiB and 4 KiB pages) ── pd_end ── 1 GiB pages, UC ── end
///      the GiBs the UEFI memory map        no RAM above the memory map
///      reaches: RAM, retyped when the
///      guest writes its MTRRs
/// ```
///
/// Every GiB the memory map reaches keeps a PD, even one of a single type,
/// so the guest's MTRR writes can retype its RAM 2 MiB or 4 KiB at a time
/// (`retype`) without splitting a 1 GiB page. Above the memory map there
/// is no RAM, so whatever the MTRRs say a GiB there is UC ([`MapTypes`]),
/// and never retyped: one 1 GiB page each.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MapLayout {
    /// GiBs `[0, pd_gibs)` are mapped through one PD each.
    pub pd_gibs: usize,
    /// GiBs `[pd_gibs, gibs)` are 1 GiB pages; the map ends at GiB `gibs`.
    pub gibs: usize,
}

impl MapLayout {
    /// The layout for a memory map ending at `memory_map_end`
    /// ([`memory_map_end`]) on a CPU with `physical_address_bits` (MAXPHYADDR)
    /// and 1 GiB nested pages or not (`huge_pages`): up to the end of the
    /// address space ([`address_space_end`]). Without 1 GiB pages, a PD per
    /// GiB of a 46-bit address space would take 256 MiB of tables, so the
    /// PDs then stop at `max(memory_map_end, 4 GiB)`: the 32-bit device
    /// window, with the Local APIC and the IOAPIC, is mapped, and an access
    /// above it is a nested page fault (an EPT violation on Intel).
    pub fn new(memory_map_end: u64, physical_address_bits: u8, huge_pages: bool) -> Result<Self, MapError> {
        let limit = address_space_end(physical_address_bits);
        if memory_map_end > limit {
            return Err(MapError::RangeTooLarge {
                end: memory_map_end,
                limit,
            });
        }
        let gibs_to = |end: u64| (end / SIZE_1G) as usize;
        Ok(if huge_pages {
            Self {
                pd_gibs: gibs_to(memory_map_end),
                gibs: gibs_to(limit),
            }
        } else {
            let pd_gibs = gibs_to(memory_map_end.max(LOW_DEVICE_END).min(limit));
            Self { pd_gibs, gibs: pd_gibs }
        })
    }

    /// End of the map: GPAs `[0, end)` are mapped.
    pub fn end(self) -> u64 {
        self.gibs as u64 * SIZE_1G
    }

    /// End of the part mapped through PDs.
    pub fn pd_end(self) -> u64 {
        self.pd_gibs as u64 * SIZE_1G
    }

    /// How many GiBs are mapped through a PD each.
    pub fn pd_gibs(self) -> usize {
        self.pd_gibs
    }

    /// How many GiBs are 1 GiB pages.
    pub fn huge_gibs(self) -> usize {
        self.gibs - self.pd_gibs
    }

    /// How many PDPTs the map takes: one per 512 GiB, each in a PML4 entry.
    pub fn pdpts(self) -> usize {
        self.gibs.div_ceil(ENTRIES)
    }
}

/// Whether memory of UEFI type `memory_type` is RAM: memory the firmware or
/// an OS uses as memory, which EPT gives the type the MTRRs give it. Any
/// other type (memory-mapped I/O, reserved or unusable memory, a type this
/// list does not know) may hold a device's registers, and EPT maps it UC
/// ([`MapTypes`]).
pub fn is_ram(memory_type: boot::MemoryType) -> bool {
    use boot::MemoryType as Type;
    [
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
    ]
    .contains(&memory_type)
}

/// The RAM of the UEFI memory map ([`is_ram`]): `[start, end)` ranges in
/// address order, apart from each other, so that memory RAM covers without
/// a gap lies in one range.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RamMap<'a> {
    pub ranges: &'a [(u64, u64)],
}

/// How much of a block of memory is RAM: see [`RamMap::coverage`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RamCoverage {
    All,
    Part,
    None,
}

impl<'a> RamMap<'a> {
    /// The RAM of the memory-map regions `(start, pages, type)`, gathered in
    /// `ranges`, which has room for one range per region: sorted, and with
    /// ranges that overlap or touch merged into one.
    pub fn collect(regions: impl Iterator<Item = (u64, u64, boot::MemoryType)>, ranges: &'a mut [(u64, u64)]) -> Self {
        let mut count = 0;
        for (start, pages, _) in regions.filter(|&(_, pages, memory_type)| pages > 0 && is_ram(memory_type)) {
            let Some(slot) = ranges.get_mut(count) else {
                break;
            };
            *slot = (start, start.saturating_add(pages.saturating_mul(PAGE_SIZE as u64)));
            count += 1;
        }
        let ranges = &mut ranges[..count];
        ranges.sort_unstable_by_key(|&(start, _)| start);
        // Merged in place: range `i` goes on the last merged one if it
        // overlaps or touches it, else after it. `merged` <= `i` throughout.
        let mut merged = 0;
        for i in 0..ranges.len() {
            let (start, end) = ranges[i];
            if merged > 0 && start <= ranges[merged - 1].1 {
                ranges[merged - 1].1 = ranges[merged - 1].1.max(end);
            } else {
                ranges[merged] = (start, end);
                merged += 1;
            }
        }
        let ranges: &'a [(u64, u64)] = ranges;
        Self {
            ranges: &ranges[..merged],
        }
    }

    /// The ranges, in address order.
    pub fn ranges(&self) -> &'a [(u64, u64)] {
        self.ranges
    }

    /// Whether `address` is RAM.
    pub fn contains(&self, address: u64) -> bool {
        // The first range ending past `address` is the only one that can
        // hold it.
        let i = self.ranges.partition_point(|&(_, end)| end <= address);
        self.ranges.get(i).is_some_and(|&(start, _)| start <= address)
    }

    /// How much of the `size` bytes at `base` is RAM.
    pub fn coverage(&self, base: u64, size: u64) -> RamCoverage {
        let end = base + size;
        // The first range ending past `base`: all of the block lies in it,
        // or some of it, or none of the block is RAM, since every later
        // range starts later still.
        let i = self.ranges.partition_point(|&(_, range_end)| range_end <= base);
        match self.ranges.get(i) {
            Some(&(start, range_end)) if start <= base && end <= range_end => RamCoverage::All,
            Some(&(start, _)) if start < end => RamCoverage::Part,
            _ => RamCoverage::None,
        }
    }
}

/// The memory type EPT gives each guest page: the one the MTRRs give it
/// when it is RAM, UC when it is not.
///
/// Without EPT the MTRRs alone type every access, and firmware programs
/// them so that device memory is UC. EPT replaces them for guest accesses,
/// and they are now the guest's, which can make any range WB. Some VMMs
/// also leave device memory to their own EPT: Hyper-V's MTRRs make all of
/// the first 4 GiB WB, the device window below 4 GiB included (`docs/MTRR.md`
/// section 6). WB on a device's registers lets the CPU read them
/// speculatively and keep them in its caches, which the device does not
/// expect, and can end in a machine check, for the whole machine. So
/// whatever the MTRRs say, what the UEFI memory map does not call RAM is
/// UC. The guest's PAT can still make it WC, for a framebuffer: UC from
/// EPT with WC from the PAT gives WC (Intel SDM Vol. 3A, "Selecting Memory
/// Types for Pentium III and More Recent Processor Families"). KVM does the
/// same (`vmx_get_mt_mask` gives MMIO UC).
#[derive(Debug, Clone, Copy)]
pub struct MapTypes<'a> {
    pub mtrrs: &'a Mtrrs,
    pub ram: RamMap<'a>,
}

impl MapTypes<'_> {
    /// The EPT memory type of the 4 KiB page at `address`.
    pub fn page_type(&self, address: u64) -> MemoryType {
        if self.ram.contains(address) {
            self.mtrrs.memory_type(address)
        } else {
            MemoryType::Uncacheable
        }
    }

    /// The EPT memory type of the 2 MiB region at `region` when one 2 MiB
    /// page can carry it, or `None` when it needs a PT: its pages differ in
    /// type, or only some of them are RAM.
    ///
    /// A region only partly RAM always gets a PT, whatever the types now.
    /// The guest's MTRRs can make its RAM WB at any time while the rest
    /// stays UC, and that PT must not have to come from the spare ones
    /// ([`spare_pts`]), which only cover what MTRR writes can split.
    pub fn region_type(&self, region: u64) -> Option<MemoryType> {
        match self.ram.coverage(region, SIZE_2M) {
            RamCoverage::All => self.mtrrs.block_type(region, SIZE_2M),
            RamCoverage::Part => None,
            RamCoverage::None => Some(MemoryType::Uncacheable),
        }
    }
}

/// The memory type of the 2 MiB region at `region` when one 2 MiB page can
/// map it, or `None` when it needs a PT of 4 KiB pages: its pages differ in
/// memory type, only some of them are RAM, or it holds one of
/// `protected_pages`. An EPT entry has one memory type, so a 2 MiB page over
/// pages of mixed types would give some of them the wrong one.
pub fn large_page_type(types: &MapTypes<'_>, region: u64, protected_pages: &[u64]) -> Option<MemoryType> {
    if protected_pages.iter().any(|&page| region == region_of(page)) {
        return None;
    }
    types.region_type(region)
}

/// The start of the 2 MiB region holding `address`.
pub fn region_of(address: u64) -> u64 {
    address & !(SIZE_2M - 1)
}

/// How many PTs `Ept::build` sets aside, beyond those the map needs, for
/// the 2 MiB regions the guest's MTRR writes split later: one per variable
/// range, and one for the first 2 MiB, which the fixed ranges can split.
///
/// With contiguous masks, as the SDM describes them, that is always enough.
/// A variable range of 2 MiB or more is aligned to its size, so it covers
/// whole regions. A smaller one lies inside one region. So at most one
/// region per variable range, plus the first, has pages of different
/// types, and a region that is one type again gives its PT back. Only a
/// non-contiguous mask, one that tests an address bit below 21 and leaves a
/// higher one free, can split more: `retype` then maps the regions left
/// without a PT UC.
pub fn spare_pts(mtrr: MtrrCapabilities) -> usize {
    mtrr.variable_count() + usize::from(mtrr.fixed())
}

/// How many PTs `Ept::build` sets aside for hiding Aleph0's own pages
/// (`TypeTables::hide`): one per distinct 2 MiB region that holds a page
/// Aleph0 keeps and is not already mapped through a PT. The test VM makes
/// about 27 allocations in a handful of clusters, so this is generous; if a
/// build ever needs more, `hide` refuses the launch with
/// `EptError::NoHidePt` rather than leave a page exposed.
pub const HIDE_PTS: usize = 64;

/// How many 2 MiB regions of the first `pd_gibs` GiB need a PT
/// ([`large_page_type`] is `None`): how many PTs `fill_identity_map`
/// takes. At least one, for the region holding a protected page (two
/// protected pages in the same region still share one PT).
pub fn split_regions(types: &MapTypes<'_>, pd_gibs: usize, protected_pages: &[u64]) -> usize {
    (0..pd_gibs * ENTRIES)
        .filter(|&i| large_page_type(types, i as u64 * SIZE_2M, protected_pages).is_none())
        .count()
}

/// What `TypeTables::hide` hid.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Hidden {
    /// Data pages of Aleph0 redirected to the shadow page.
    pub data_pages: usize,
    /// Code pages of Aleph0 (the image copy, the guest's pages) kept
    /// read-only and executable instead of redirected.
    pub code_pages: usize,
    /// 2 MiB regions split into a PT to hide a page in them, taking one of
    /// the [`HIDE_PTS`].
    pub regions_split: usize,
}

/// Puts the PTs from `first_spare` to the end of `stack`'s length on the
/// free stack `stack`, so the first one comes off first, and returns how
/// many there are.
pub fn fill_free_stack(stack: &mut [u32], first_spare: usize) -> usize {
    let spare = first_spare..stack.len();
    let count = spare.len();
    for (slot, index) in stack.iter_mut().zip(spare.rev()) {
        *slot = index as u32;
    }
    count
}

/// The free PTs: a stack of their indices, the first `len` of `stack`.
pub struct FreePts<'a> {
    pub stack: &'a mut [u32],
    pub len: &'a mut usize,
}

impl FreePts<'_> {
    /// The index of a free PT, now in use, or `None` when none is left.
    pub fn take(&mut self) -> Option<usize> {
        *self.len = self.len.checked_sub(1)?;
        self.stack.get(*self.len).map(|&index| index as usize)
    }

    /// Puts the PT at `index` back: no PD entry points at it any more.
    pub fn give_back(&mut self, index: usize) {
        // The stack has room for every PT, and each is either in use or on
        // it once, so the slot is always there. A full stack means a PT was
        // given back twice: a bug in `retype`, not something a guest can do.
        let slot = self.stack.get_mut(*self.len);
        debug_assert!(slot.is_some(), "PT {index} given back to a full free stack");
        if let Some(slot) = slot {
            *slot = index as u32;
            *self.len += 1;
        }
    }
}

/// What `retype` changed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Retyped {
    /// 2 MiB regions whose entries changed.
    pub regions: usize,
    /// From the start of the first of them to the end of the last; `None`
    /// when none changed.
    pub span: Option<(u64, u64)>,
    /// Regions now mapped through a spare PT: their pages came to differ in
    /// type.
    pub split: usize,
    /// Regions now one 2 MiB page, whose PT went back to the spare ones:
    /// their pages came to share one type.
    pub merged: usize,
    /// Regions whose pages differ in type, mapped UC as one 2 MiB page for
    /// lack of a spare PT. Counted whether or not this call made them so.
    pub uncached: usize,
}

impl Retyped {
    /// Whether any entry changed.
    pub fn changed(&self) -> bool {
        self.regions > 0
    }

    /// Counts the 2 MiB region number `region` as changed. `retype`'s
    /// second pass can count a region below one the first pass counted.
    pub fn count(&mut self, region: usize) {
        let start = region as u64 * SIZE_2M;
        let end = start + SIZE_2M;
        self.regions += 1;
        self.span = Some(match self.span {
            Some((first, last)) => (first.min(start), last.max(end)),
            None => (start, end),
        });
    }
}

#[cfg(test)]
#[path = "../../tests/unit/x86/identity_map.rs"]
mod tests;
