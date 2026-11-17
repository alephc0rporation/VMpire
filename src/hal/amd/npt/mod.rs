//! Nested page tables (NPT), the AMD counterpart of `hal::intel::ept`.
//!
//! With nested paging on (`control::NESTED_CTL_NP_ENABLE`), every guest
//! memory access is translated twice, as with EPT:
//!
//! ```text
//! guest linear ──(guest page tables, guest CR3)──▶ guest-physical (GPA)
//! GPA          ──(NPT tables, built here)───────▶ host-physical  (HPA)
//! ```
//!
//! An access the NPT entries do not allow is a #VMEXIT with exit code NPF
//! (0x400, `exit::code::NPF`), the GPA in EXITINFO2 and a page-fault error
//! code in EXITINFO1.
//!
//! The map is the one EPT builds (`x86::identity_map`): an identity map of
//! the whole guest-physical address space, PDs for the GiBs the UEFI memory
//! map reaches and 1 GiB pages above, each page of RAM with the type the
//! MTRRs give it and everything else UC, the protected pages read-only, and
//! Aleph0's own pages hidden. Only the entries differ from EPT's.
//!
//! NPT entries are ordinary long-mode page-table entries (AMD64 APM Vol. 2,
//! 5.3.3, and 15.25 for nested paging):
//!
//! ```text
//!  63  62    52 51               12 11  9  8  7    6  5  4   3   2   1  0
//! ┌──┬────────┬───────────────────┬─────┬──┬─────┬──┬──┬───┬───┬───┬───┬──┐
//! │NX│ avail. │ address of next   │avail│G │PS/  │D │A │PCD│PWT│U/S│R/W│P │
//! │  │        │ table or the page │     │  │PAT  │  │  │   │   │   │   │  │
//! └──┴────────┴───────────────────┴─────┴──┴─────┴──┴──┴───┴───┴───┴───┴──┘
//! ```
//!
//! What differs from EPT, all of it from the hardware:
//!
//! - **Access.** P is "read", R/W "write", and NX (with host EFER.NXE) takes
//!   execution away; there is no write-without-read or execute-only entry.
//!   Every nested walk is a user access (APM 15.25.5), so U/S is set at
//!   every level, or the walk faults.
//! - **Memory type.** No type field: PWT, PCD and PAT (bit 7 in a PT entry,
//!   bit 12 in a large page's) index the host's PAT, the IA32_PAT MSR as it
//!   is while the guest runs (APM 15.25.8; the guest's own PAT is the VMCB's
//!   G_PAT). Aleph0 loads [`HOST_PAT`] there, which has an entry for each of
//!   the five MTRR types. The CPU then combines that type with the real
//!   MTRRs', which stay the firmware's: UC from either side gives UC, so the
//!   types here, like EPT's, decide what the guest's memory is.
//! - **Accessed and dirty.** The CPU sets A and D in nested entries as it
//!   walks; they are set in advance, so it never writes the tables.
//! - **Flushing.** There is no INVEPT. A changed entry is dropped from the
//!   TLB by the next VMRUN, through the VMCB's TLB_CONTROL ([`FlushType`]).
//! - **Faults.** There is no separate misconfiguration exit: an entry with a
//!   reserved bit set is a nested page fault too, with the RSVD bit in its
//!   error code (`walk.rs`).

use core::fmt;
use core::marker::PhantomData;
use core::sync::atomic::{AtomicPtr, AtomicU8, AtomicU64, AtomicUsize, Ordering};

use uefi::boot;
use uefi::mem::memory_map::MemoryMap;

use super::control::{ControlArea, tlb_control};
use crate::x86::identity_map::{
    FreePts, HIDE_PTS, Hidden, MAX_PROTECTED, MapError, MapLayout, MapTypes, RamCoverage, RamMap, RegionMapping,
    RegionRun, Retyped, fill_free_stack, large_page_type, memory_map_end, region_of, runs, split_regions,
};
use crate::x86::mtrr::{MemoryType, Mtrrs};
use crate::x86::page::{Allocation, PAGE_SIZE, allocate_zeroed_pages};
use crate::x86::page_table::level::{Level, Pd, Pdpt, Pml4, Pt, TableLevel};
use crate::x86::page_table::{ENTRIES, PAGE_SIZE_BIT, SIZE_1G, SIZE_2M};

pub mod walk;

pub use walk::{NptWalk, walk};

/// Entry bit 0 (P): present. In NPT, also what lets the guest read.
const PRESENT: u64 = 1 << 0;
/// Entry bit 1 (R/W): writes allowed.
const WRITABLE: u64 = 1 << 1;
/// Entry bit 2 (U/S): user access allowed. Every nested walk is a user
/// access, so every entry has it.
const USER: u64 = 1 << 2;
/// Entry bit 3 (PWT): bit 0 of the PAT index.
const PWT: u64 = 1 << 3;
/// Entry bit 4 (PCD): bit 1 of the PAT index.
const PCD: u64 = 1 << 4;
/// Entry bit 5 (A): accessed. Set in advance.
const ACCESSED: u64 = 1 << 5;
/// Leaf bit 6 (D): dirty. Set in advance.
const DIRTY: u64 = 1 << 6;
/// PT entry bit 7: bit 2 of the PAT index (in a PDPT or PD entry bit 7 is
/// PS, [`PAGE_SIZE_BIT`]).
const PAT_4K: u64 = 1 << 7;
/// Large-page bit 12: bit 2 of the PAT index of a 1 GiB or 2 MiB page.
const PAT_LARGE: u64 = 1 << 12;
/// Entry bit 63 (NX): no instruction fetch. A reserved bit unless the host
/// runs with EFER.NXE ([`EntryRules::nx`]).
const NO_EXECUTE: u64 = 1 << 63;
/// Bits 51:12: the physical address in an entry.
const ADDRESS_MASK: u64 = 0x000F_FFFF_FFFF_F000;

/// Access bit: reads allowed (P). The three access bits are EPT's
/// (`hal::intel::ept::EPT_READ` and the others), so reports read alike; [`access_bits`]
/// turns them into NPT bits.
pub const NPT_READ: u64 = 1 << 0;
/// Access bit: writes allowed (R/W).
pub const NPT_WRITE: u64 = 1 << 1;
/// Access bit: instruction fetches allowed (NX clear).
pub const NPT_EXECUTE: u64 = 1 << 2;
/// Every access allowed.
pub const NPT_RWX: u64 = NPT_READ | NPT_WRITE | NPT_EXECUTE;

/// The IA32_PAT value Aleph0 runs the guest with, which nested walks read
/// (see the module docs): entries 0-3 are the power-on PAT's (WB, WT, UC-,
/// UC), so Aleph0's own page tables, which use entry 0, keep WB, and 4-7
/// add WC and WP, which the power-on PAT lacks.
///
/// ```text
/// index  0   1   2    3   4   5   6    7
/// type   WB  WT  UC-  UC  WC  WP  UC-  UC
/// ```
pub const HOST_PAT: u64 = 0x0007_0501_0007_0406;

/// The [`HOST_PAT`] index of `memory_type`.
const fn pat_index(memory_type: MemoryType) -> u64 {
    match memory_type {
        MemoryType::WriteBack => 0,
        MemoryType::WriteThrough => 1,
        MemoryType::Uncacheable => 3,
        MemoryType::WriteCombining => 4,
        MemoryType::WriteProtected => 5,
    }
}

/// The memory type [`HOST_PAT`] gives `index`, or `None` for UC- (2 and 6),
/// which Aleph0 never uses and which has no MTRR type.
pub const fn pat_type(index: u64) -> Option<MemoryType> {
    match HOST_PAT >> ((index & 0b111) * 8) & 0xFF {
        0 => Some(MemoryType::Uncacheable),
        1 => Some(MemoryType::WriteCombining),
        4 => Some(MemoryType::WriteThrough),
        5 => Some(MemoryType::WriteProtected),
        6 => Some(MemoryType::WriteBack),
        _ => None,
    }
}

/// The PWT, PCD and PAT bits of a leaf entry of `memory_type`: `pat` is the
/// entry's PAT bit, bit 7 in a PT entry and bit 12 in a large page's.
const fn type_bits(memory_type: MemoryType, pat: u64) -> u64 {
    let index = pat_index(memory_type);
    (if index & 0b001 != 0 { PWT } else { 0 })
        | (if index & 0b010 != 0 { PCD } else { 0 })
        | (if index & 0b100 != 0 { pat } else { 0 })
}

/// The [`HOST_PAT`] index a leaf entry `raw` holds: its PAT bit is `pat`.
const fn leaf_pat_index(raw: u64, pat: u64) -> u64 {
    (if raw & PWT != 0 { 0b001 } else { 0 })
        | (if raw & PCD != 0 { 0b010 } else { 0 })
        | (if raw & pat != 0 { 0b100 } else { 0 })
}

/// The bits an entry needs for `access` (a mix of [`NPT_READ`],
/// [`NPT_WRITE`], [`NPT_EXECUTE`]): none without read, since an entry that
/// is not present allows nothing. Without NX (`rules.nx` false) execution
/// cannot be taken away, so an entry that may be read may be executed.
fn access_bits(access: u64, rules: EntryRules) -> u64 {
    if access & NPT_READ == 0 {
        return 0;
    }
    PRESENT
        | USER
        | (if access & NPT_WRITE != 0 { WRITABLE } else { 0 })
        | (if access & NPT_EXECUTE == 0 && rules.nx {
            NO_EXECUTE
        } else {
            0
        })
}

/// The access an entry `raw` allows, as [`NPT_READ`], [`NPT_WRITE`] and
/// [`NPT_EXECUTE`] bits: 0 when it is not present.
pub fn entry_access(raw: u64) -> u64 {
    if raw & PRESENT == 0 {
        return 0;
    }
    NPT_READ | (if raw & WRITABLE != 0 { NPT_WRITE } else { 0 }) | (if raw & NO_EXECUTE == 0 { NPT_EXECUTE } else { 0 })
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NptError {
    /// The CPU has no nested paging (CPUID 0x8000000A:EDX.NP).
    NestedPagingUnsupported,
    /// UEFI could not return its memory map.
    MemoryMapUnavailable,
    /// The memory map has no memory in it.
    EmptyMemoryMap,
    /// The memory map reaches `end`, past `limit`, the end of the
    /// guest-physical address space.
    RangeTooLarge { end: u64, limit: u64 },
    /// The page to protect lies outside the identity map.
    ProtectedPageOutsideMap,
    /// A page Aleph0 keeps, to hide from the guest ([`TypeTables::hide`]),
    /// lies outside the part of the map mapped through PDs, where it cannot
    /// be redirected a 4 KiB page at a time.
    HiddenPageOutsideMap { gpa: u64 },
    /// Hiding Aleph0's pages ran out of the PTs [`Npt::build`] set aside for
    /// it ([`HIDE_PTS`]); the launch is refused rather than leaving a page of
    /// Aleph0 readable by the guest.
    NoHidePt,
    /// UEFI could not provide the table pages.
    AllocationFailed,
    /// An NPT was already built and its protected pages published; the exit
    /// handler keeps one published set.
    AlreadyBuilt,
}

impl From<MapError> for NptError {
    fn from(error: MapError) -> Self {
        match error {
            MapError::EmptyMemoryMap => Self::EmptyMemoryMap,
            MapError::RangeTooLarge { end, limit } => Self::RangeTooLarge { end, limit },
        }
    }
}

/// What the CPU allows in an NPT entry beyond what every one allows: what the
/// map may use ([`MapLayout`]), and what the CPU refuses in an entry.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EntryRules {
    /// MAXPHYADDR: an entry's address bits from here up to 51 are reserved.
    pub physical_address_bits: u8,
    /// Whether a PDPT entry may map a 1 GiB page (CPUID 0x80000001:EDX
    /// bit 26, `x86::cpuid::has_1g_pages`).
    pub huge_pages: bool,
    /// Whether NX is a permission (host EFER.NXE): without it bit 63 is
    /// reserved, and execution cannot be taken away.
    pub nx: bool,
}

/// How the next VMRUN drops the guest's cached translations after an NPT
/// change: the TLB_CONTROL value it writes. The AMD counterpart of
/// `hal::intel::ept::InveptType`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FlushType {
    /// Only this guest's ASID (needs flush-by-ASID, CPUID 0x8000000A:EDX
    /// bit 6).
    Asid,
    /// The whole TLB, every ASID.
    All,
}

impl FlushType {
    /// The TLB_CONTROL value.
    pub fn tlb_control(self) -> u32 {
        match self {
            Self::Asid => tlb_control::FLUSH_ASID,
            Self::All => tlb_control::FLUSH_ALL,
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Self::Asid => "this ASID",
            Self::All => "all ASIDs",
        }
    }
}

/// What this CPU's NPT offers Aleph0: see [`check_support`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NptSupport {
    pub flush: FlushType,
    pub rules: EntryRules,
}

/// Checks that this CPU has nested paging and picks how to flush: by ASID
/// when it can (`flush_by_asid`), the whole TLB otherwise. `nested_paging`
/// and `flush_by_asid` come from `svm::SvmSupport`;
/// `physical_address_bits` is MAXPHYADDR, `huge_pages` whether the CPU has
/// 1 GiB pages, and `nx` whether the host runs with EFER.NXE. 2 MiB pages
/// and a 4-level walk are in every long-mode CPU, so unlike EPT nothing
/// else needs checking.
pub fn check_support(
    nested_paging: bool,
    flush_by_asid: bool,
    physical_address_bits: u8,
    huge_pages: bool,
    nx: bool,
) -> Result<NptSupport, NptError> {
    if !nested_paging {
        return Err(NptError::NestedPagingUnsupported);
    }
    Ok(NptSupport {
        flush: if flush_by_asid { FlushType::Asid } else { FlushType::All },
        rules: EntryRules {
            physical_address_bits,
            huge_pages,
            nx,
        },
    })
}

/// One entry of a level-`L` table, laid out as in the module docs.
/// `repr(transparent)`, so a table page is 512 of these back to back.
///
/// | Level | Constructors |
/// |---|---|
/// | [`Pml4`] | [`table`](Self::table) |
/// | [`Pdpt`] | [`table`](Self::table), [`huge_page`](Self::huge_page) |
/// | [`Pd`] | [`table`](Self::table), [`large_page`](Self::large_page) |
/// | [`Pt`] | [`page`](Self::page), [`with_access`](Self::with_access) |
#[repr(transparent)]
pub struct NptEntry<L: Level> {
    raw: u64,
    level: PhantomData<L>,
}

impl<L: Level> NptEntry<L> {
    /// Not present.
    pub const EMPTY: Self = Self::from_raw(0);

    const fn from_raw(raw: u64) -> Self {
        Self {
            raw,
            level: PhantomData,
        }
    }

    /// The entry's bits.
    #[cfg_attr(
        not(any(test, feature = "ept-misconfig-test")),
        allow(dead_code, reason = "used by the unit tests and `misconfigure`")
    )]
    pub fn raw(self) -> u64 {
        self.raw
    }
}

impl<L: TableLevel> NptEntry<L> {
    /// An entry pointing to the level-`L::Next` table at `address`, allowing
    /// every access: present, writable, user, executable. A non-leaf entry
    /// only limits what lies below it, so the leaf entries decide.
    pub fn table(address: u64) -> Self {
        Self::from_raw((address & ADDRESS_MASK) | PRESENT | WRITABLE | USER | ACCESSED)
    }
}

/// The bits of an RWX large page at `address` of `memory_type`: a 1 GiB page
/// in a PDPT entry, a 2 MiB one in a PD entry. Never NX: every large page is
/// full RWX.
fn large_page_bits(address: u64, memory_type: MemoryType) -> u64 {
    (address & ADDRESS_MASK)
        | PRESENT
        | WRITABLE
        | USER
        | ACCESSED
        | DIRTY
        | PAGE_SIZE_BIT
        | type_bits(memory_type, PAT_LARGE)
}

/// How a large-page entry `raw` maps its block, read back: present, a page,
/// full RWX and of a type the PAT names; [`RegionMapping::Invalid`]
/// otherwise. `page` makes the type a [`RegionMapping`].
fn large_mapping(raw: u64, page: fn(MemoryType) -> RegionMapping) -> RegionMapping {
    let full = PRESENT | WRITABLE | USER;
    if raw & full != full || raw & NO_EXECUTE != 0 || raw & PAGE_SIZE_BIT == 0 {
        return RegionMapping::Invalid;
    }
    pat_type(leaf_pat_index(raw, PAT_LARGE)).map_or(RegionMapping::Invalid, page)
}

impl NptEntry<Pdpt> {
    /// An entry mapping the 1 GiB page at `address`: RWX, of `memory_type`.
    /// Only on a CPU with 1 GiB pages ([`EntryRules::huge_pages`]).
    pub fn huge_page(address: u64, memory_type: MemoryType) -> Self {
        Self::from_raw(large_page_bits(address, memory_type))
    }

    /// How the entry maps its GiB as one 1 GiB page:
    /// [`RegionMapping::HugePage`], or [`RegionMapping::Invalid`] for
    /// anything else, an entry pointing to a PD included.
    pub fn mapping(self) -> RegionMapping {
        large_mapping(self.raw, RegionMapping::HugePage)
    }
}

impl NptEntry<Pd> {
    /// An entry mapping the 2 MiB page at `address`: RWX, of `memory_type`.
    pub fn large_page(address: u64, memory_type: MemoryType) -> Self {
        Self::from_raw(large_page_bits(address, memory_type))
    }

    /// How the entry maps its 2 MiB region, read back from its bits.
    pub fn mapping(self) -> RegionMapping {
        if self.raw & PRESENT == 0 {
            return RegionMapping::Invalid;
        }
        if self.raw & PAGE_SIZE_BIT == 0 {
            return RegionMapping::Pt;
        }
        large_mapping(self.raw, RegionMapping::LargePage)
    }
}

/// The regions `pds` map, as [`runs`], from the one of region number
/// `first` (GPA `first` * 2 MiB).
fn region_runs(pds: &[NptEntry<Pd>], first: usize) -> impl Iterator<Item = RegionRun> + '_ {
    runs(pds.iter().map(|entry| entry.mapping()), first, SIZE_2M)
}

/// The GiBs `pdpts` map as 1 GiB pages, as [`runs`], from the entry of GiB
/// number `first`.
fn gib_runs(pdpts: &[NptEntry<Pdpt>], first: usize) -> impl Iterator<Item = RegionRun> + '_ {
    runs(pdpts.iter().map(|entry| entry.mapping()), first, SIZE_1G)
}

impl NptEntry<Pt> {
    /// An entry mapping the 4 KiB page at `address` with `access` (a mix of
    /// [`NPT_READ`], [`NPT_WRITE`], [`NPT_EXECUTE`]), of `memory_type`, on a
    /// CPU with `rules`.
    pub fn page(address: u64, access: u64, memory_type: MemoryType, rules: EntryRules) -> Self {
        Self::from_raw((address & ADDRESS_MASK) | ACCESSED | DIRTY | type_bits(memory_type, PAT_4K))
            .with_access(access, rules)
    }

    /// The same entry with its access replaced by `access`: P, R/W, U/S and
    /// NX. Its address and memory type stay.
    pub fn with_access(self, access: u64, rules: EntryRules) -> Self {
        let access_mask = PRESENT | WRITABLE | USER | NO_EXECUTE;
        Self::from_raw((self.raw & !access_mask) | access_bits(access, rules))
    }

    /// The same entry with its memory type replaced by `memory_type`; its
    /// address and access stay.
    pub fn with_memory_type(self, memory_type: MemoryType) -> Self {
        let type_mask = PWT | PCD | PAT_4K;
        Self::from_raw((self.raw & !type_mask) | type_bits(memory_type, PAT_4K))
    }

    /// The access the entry allows ([`entry_access`]).
    pub fn access(self) -> u64 {
        entry_access(self.raw)
    }

    /// The memory type the entry gives its page, or `None` for UC-.
    #[cfg_attr(not(test), allow(dead_code, reason = "used by the unit tests"))]
    pub fn memory_type(self) -> Option<MemoryType> {
        pat_type(leaf_pat_index(self.raw, PAT_4K))
    }
}

// Written by hand rather than derived: a derive would require `L` itself to
// implement each trait, and the level types are never instantiated.
impl<L: Level> Clone for NptEntry<L> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<L: Level> Copy for NptEntry<L> {}

impl<L: Level> PartialEq for NptEntry<L> {
    fn eq(&self, other: &Self) -> bool {
        self.raw == other.raw
    }
}

impl<L: Level> Eq for NptEntry<L> {}

impl<L: Level> Default for NptEntry<L> {
    fn default() -> Self {
        Self::EMPTY
    }
}

impl<L: Level> fmt::Debug for NptEntry<L> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "NptEntry<{}>({:#x})", L::NAME, self.raw)
    }
}

/// The nested CR3: the VMCB's N_CR3 field, the PML4's physical address. Its
/// PWT and PCD (bits 3 and 4) are 0, so the PML4 itself is read as
/// [`HOST_PAT`] entry 0 gives it: WB. The AMD counterpart of
/// `hal::intel::ept::Eptp`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Ncr3(u64);

impl Ncr3 {
    /// The nested CR3 for the PML4 table at `pml4`.
    pub fn new(pml4: u64) -> Self {
        Self(pml4 & ADDRESS_MASK)
    }

    /// The value as the CPU reads it.
    pub fn raw(self) -> u64 {
        self.0
    }

    /// The physical address of the PML4 table it names.
    pub fn pml4(self) -> u64 {
        self.0 & ADDRESS_MASK
    }
}

/// The tables of an identity map, as slices, with the physical address each
/// one has. Under UEFI's identity mapping a table's physical address is its
/// virtual one; on the host tests these are made-up numbers.
struct Tables<'a> {
    pml4: &'a mut [NptEntry<Pml4>],
    /// One PDPT per 512 GiB of the map, back to back, so entry `g` maps
    /// GiB `g`.
    pdpts: &'a mut [NptEntry<Pdpt>],
    /// One PD per GiB mapped through a PD ([`MapLayout`]), back to back.
    pds: &'a mut [NptEntry<Pd>],
    /// One PT per split region ([`split_regions`]), back to back.
    pts: &'a mut [NptEntry<Pt>],
    pdpts_address: u64,
    pds_address: u64,
    pts_address: u64,
}

/// Fills `tables` with an identity map laid out as `layout` says, each page
/// with the memory type `types` gives it, and RWX except `protected_pages`,
/// which are read-only (and not executable where NX is available, as EPT's
/// R-only entry). As `hal::intel::ept`'s: a 2 MiB region is
/// one 2 MiB page when it can be ([`large_page_type`]), otherwise it takes
/// the next PT in `tables.pts`. Returns the index in `tables.pts` of each
/// protected page's entry, in the order given.
fn fill_identity_map(
    tables: Tables<'_>,
    layout: MapLayout,
    types: &MapTypes<'_>,
    protected_pages: &[u64],
    rules: EntryRules,
) -> [usize; MAX_PROTECTED] {
    debug_assert!(tables.pdpts.len() == layout.pdpts() * ENTRIES);
    debug_assert!(tables.pds.len() == layout.pd_gibs * ENTRIES);
    debug_assert!(protected_pages.len() <= MAX_PROTECTED);
    for (number, entry) in tables.pml4.iter_mut().take(layout.pdpts()).enumerate() {
        *entry = NptEntry::table(tables.pdpts_address + (number * PAGE_SIZE) as u64);
    }
    for (gib, entry) in tables.pdpts.iter_mut().take(layout.gibs).enumerate() {
        let base = gib as u64 * SIZE_1G;
        *entry = if gib < layout.pd_gibs {
            NptEntry::table(tables.pds_address + (gib * PAGE_SIZE) as u64)
        } else {
            // Above the memory map, so no RAM: UC whatever the MTRRs say.
            debug_assert!(types.ram.coverage(base, SIZE_1G) == RamCoverage::None);
            NptEntry::huge_page(base, MemoryType::Uncacheable)
        };
    }

    let (pts, rest) = tables.pts.as_chunks_mut::<ENTRIES>();
    debug_assert!(rest.is_empty(), "the PTs are whole tables");
    let mut pts = pts.iter_mut().enumerate();
    let mut protected = [None; MAX_PROTECTED];
    for (i, entry) in tables.pds.iter_mut().enumerate() {
        let region = i as u64 * SIZE_2M;
        if let Some(memory_type) = large_page_type(types, region, protected_pages) {
            *entry = NptEntry::large_page(region, memory_type);
            continue;
        }
        let (number, pt) = pts.next().expect("split_regions counted a PT for this region");
        *entry = NptEntry::table(tables.pts_address + (number * PAGE_SIZE) as u64);
        fill_pt(pt, region, types, rules);
        for (slot, &protected_page) in protected_pages.iter().enumerate() {
            if region == region_of(protected_page) {
                let index = Pt::index(protected_page);
                pt[index] = pt[index].with_access(NPT_READ, rules);
                protected[slot] = Some(number * ENTRIES + index);
            }
        }
    }
    debug_assert!(pts.next().is_none(), "every PT is used");
    let mut indices = [0; MAX_PROTECTED];
    for slot in 0..protected_pages.len() {
        indices[slot] = protected[slot].expect("the protected page is inside the map");
    }
    indices
}

/// Fills `pt` with the 512 pages of the 2 MiB region at `region`: RWX,
/// each of the memory type `types` gives it.
fn fill_pt(pt: &mut [NptEntry<Pt>; ENTRIES], region: u64, types: &MapTypes<'_>, rules: EntryRules) {
    for (j, page) in pt.iter_mut().enumerate() {
        let address = region + (j * PAGE_SIZE) as u64;
        *page = NptEntry::page(address, NPT_RWX, types.page_type(address), rules);
    }
}

/// The tables [`TypeTables::hide`] redirects pages in, as [`Retype`] is for
/// retyping.
struct Hiding<'a> {
    pds: &'a mut [NptEntry<Pd>],
    pts: &'a mut [[NptEntry<Pt>; ENTRIES]],
    free: FreePts<'a>,
    pts_address: u64,
    shadow: u64,
    types: MapTypes<'a>,
    rules: EntryRules,
}

/// Hides the 4 KiB page at GPA `page` from the guest, as `hal::intel::ept`'s
/// `hide_page`: a data page (`code` false) is redirected to the shadow page,
/// read-write, not executable when NX is available; a code page (`code`
/// true) is kept at its own address, read-only and executable. Splits the
/// region into a PT first if it still maps as one 2 MiB page.
fn hide_page(tables: &mut Hiding<'_>, page: u64, code: bool, hidden: &mut Hidden) -> Result<(), NptError> {
    let region_number = (page / SIZE_2M) as usize;
    let region = region_number as u64 * SIZE_2M;
    let pd = tables
        .pds
        .get_mut(region_number)
        .ok_or(NptError::HiddenPageOutsideMap { gpa: page })?;
    let index = match pd.mapping() {
        RegionMapping::Pt => {
            pt_index(*pd, tables.pts_address, tables.pts.len()).ok_or(NptError::HiddenPageOutsideMap { gpa: page })?
        }
        RegionMapping::LargePage(_) => {
            let index = tables.free.take().ok_or(NptError::NoHidePt)?;
            fill_pt(&mut tables.pts[index], region, &tables.types, tables.rules);
            *pd = NptEntry::table(tables.pts_address + (index * PAGE_SIZE) as u64);
            hidden.regions_split += 1;
            index
        }
        RegionMapping::HugePage(_) | RegionMapping::Invalid => {
            return Err(NptError::HiddenPageOutsideMap { gpa: page });
        }
    };
    let hidden_entry = if code {
        NptEntry::page(page, NPT_READ | NPT_EXECUTE, tables.types.page_type(page), tables.rules)
    } else {
        NptEntry::page(tables.shadow, NPT_READ | NPT_WRITE, MemoryType::WriteBack, tables.rules)
    };
    let entry = &mut tables.pts[index][Pt::index(page)];
    if *entry != hidden_entry {
        *entry = hidden_entry;
        if code {
            hidden.code_pages += 1;
        } else {
            hidden.data_pages += 1;
        }
    }
    Ok(())
}

/// Whether `pt`, which maps the 2 MiB region at `region`, holds a page that
/// is not what [`fill_pt`] makes there, so its region must stay a PT: a
/// hidden page, or a protected one. As `hal::intel::ept`'s `pt_pinned`, which
/// looks at the access alone: an EPT redirect to the shadow is R+W, never
/// full RWX. Here, on a CPU without NX, the redirect is full access, so a
/// page that maps another address than its own pins the region too.
/// Collapsing it to one 2 MiB page would re-expose the hidden page.
fn pt_pinned(pt: &[NptEntry<Pt>; ENTRIES], region: u64, rules: EntryRules) -> bool {
    pt.iter().enumerate().any(|(j, entry)| {
        let own_address = region + (j * PAGE_SIZE) as u64;
        entry.raw & ADDRESS_MASK != own_address || *entry != entry.with_access(NPT_RWX, rules)
    })
}

/// One NPT identity map. Its table pages are never freed: the CPU walks them
/// on every guest access for as long as the VMCB points at them.
pub struct Npt {
    ncr3: Ncr3,
    layout: MapLayout,
    memory_map_end: u64,
    split_regions: usize,
    spare_pts: usize,
    table_pages: usize,
}

impl Npt {
    /// Builds the identity map from the UEFI memory map, as `hal::intel::ept`'s
    /// `Ept::build`: laid out for the CPU's `support` ([`MapLayout`]), with
    /// the memory types of `mtrrs` for its RAM and UC for everything else,
    /// each of `protected_pages` read-only, `spare_pts` more PTs set aside
    /// for the guest's MTRR writes and [`HIDE_PTS`] more for hiding Aleph0's
    /// own pages, and a shadow page for them. Publishes the
    /// [`ProtectedPage`]s the exit handler needs later, and returns, with the
    /// map, the [`TypeTables`] it needs to change memory types and to hide
    /// Aleph0's pages.
    ///
    /// # Safety
    ///
    /// Boot services must still be active, and physical memory must be
    /// identity-mapped (as UEFI does), since the tables are filled through
    /// their physical addresses.
    pub unsafe fn build(
        protected_pages: &[u64],
        support: NptSupport,
        mtrrs: &Mtrrs,
        spare_pts: usize,
    ) -> Result<(Self, TypeTables), NptError> {
        debug_assert!(protected_pages.len() <= MAX_PROTECTED);
        let rules = support.rules;
        let map = boot::memory_map(boot::MemoryType::LOADER_DATA).map_err(|_| NptError::MemoryMapUnavailable)?;
        let memory_map_end = memory_map_end(map.entries().map(|region| (region.phys_start, region.page_count)))?;
        let layout = MapLayout::new(memory_map_end, rules.physical_address_bits, rules.huge_pages)?;
        for &protected_page in protected_pages {
            if protected_page >= layout.pd_end() || !protected_page.is_multiple_of(PAGE_SIZE as u64) {
                return Err(NptError::ProtectedPageOutsideMap);
            }
        }

        // The RAM, kept with the tables: `retype` needs it in exit context,
        // which cannot ask UEFI. One range per memory-map region at most.
        let regions = map.entries().count();
        let ram_pages = (regions * size_of::<(u64, u64)>()).div_ceil(PAGE_SIZE);
        let ram_ranges = allocate_zeroed_pages(ram_pages, NptError::AllocationFailed)?.as_ptr() as *mut (u64, u64);
        // SAFETY: a zeroed, page-aligned allocation of at least `regions`
        // ranges, which nothing else refers to; never freed.
        let ram_ranges = unsafe { core::slice::from_raw_parts_mut(ram_ranges, regions) };
        let ram = RamMap::collect(
            map.entries()
                .map(|region| (region.phys_start, region.page_count, region.ty)),
            ram_ranges,
        );
        let types = MapTypes { mtrrs, ram };

        let split_regions = split_regions(&types, layout.pd_gibs, protected_pages);
        let pt_count = split_regions + spare_pts + HIDE_PTS;
        let shadow = allocate_zeroed_pages(1, NptError::AllocationFailed)?.as_ptr() as u64;
        let pml4 = allocate_zeroed_pages(1, NptError::AllocationFailed)?.as_ptr() as u64;
        let pdpts = allocate_zeroed_pages(layout.pdpts(), NptError::AllocationFailed)?.as_ptr() as u64;
        let pds = allocate_zeroed_pages(layout.pd_gibs, NptError::AllocationFailed)?.as_ptr() as u64;
        let pts = allocate_zeroed_pages(pt_count, NptError::AllocationFailed)?.as_ptr() as u64;
        let free_pages = (pt_count * size_of::<u32>()).div_ceil(PAGE_SIZE);
        let free = allocate_zeroed_pages(free_pages, NptError::AllocationFailed)?.as_ptr() as *mut u32;

        /// The `pages` table pages at `address` as one slice of entries.
        ///
        /// # Safety
        ///
        /// `address` is a page-aligned allocation of `pages` zeroed pages,
        /// and nothing else refers to it for the returned lifetime.
        unsafe fn table<'a, L: Level>(address: u64, pages: usize) -> &'a mut [NptEntry<L>] {
            // SAFETY: the caller guarantees the memory is ours and as long as
            // the slice; an all-zero `NptEntry` is a valid (empty) entry.
            unsafe { core::slice::from_raw_parts_mut(address as *mut NptEntry<L>, pages * ENTRIES) }
        }
        // SAFETY: each allocation above is page-aligned, zeroed and at least
        // as long as its slice, and nothing else refers to it yet.
        let tables = unsafe {
            Tables {
                pml4: table(pml4, 1),
                pdpts: table(pdpts, layout.pdpts()),
                pds: table(pds, layout.pd_gibs),
                pts: table(pts, split_regions),
                pdpts_address: pdpts,
                pds_address: pds,
                pts_address: pts,
            }
        };
        let protected = fill_identity_map(tables, layout, &types, protected_pages, rules);
        let mut protected_regions = [usize::MAX; MAX_PROTECTED];
        for (slot, &protected_page) in protected_pages.iter().enumerate() {
            protected_regions[slot] = (protected_page / SIZE_2M) as usize;
        }
        // SAFETY: `free` is a zeroed allocation of at least `pt_count` u32s,
        // which nothing else refers to yet.
        let free_len = fill_free_stack(
            unsafe { core::slice::from_raw_parts_mut(free, pt_count) },
            split_regions,
        );

        let npt = Self {
            ncr3: Ncr3::new(pml4),
            layout,
            memory_map_end,
            split_regions,
            spare_pts,
            table_pages: 1 + layout.pdpts() + layout.pd_gibs + pt_count,
        };
        let tables = TypeTables {
            pml4,
            pdpts: pdpts as *mut NptEntry<Pdpt>,
            layout,
            pds: pds as *mut NptEntry<Pd>,
            regions: layout.pd_gibs * ENTRIES,
            pts: pts as *mut NptEntry<Pt>,
            pt_count,
            pts_address: pts,
            free,
            free_len,
            protected_regions,
            shadow,
            ram: ram.ranges().as_ptr(),
            ram_len: ram.ranges().len(),
            flush: support.flush,
            rules,
        };
        let mut pages = [ProtectedPage::ABSENT; MAX_PROTECTED];
        for (slot, &gpa) in protected_pages.iter().enumerate() {
            pages[slot] = ProtectedPage {
                gpa,
                entry: (pts as *mut NptEntry<Pt>).wrapping_add(protected[slot]),
                flush: support.flush,
                rules,
            };
        }
        publish(&pages[..protected_pages.len()])?;
        Ok((npt, tables))
    }

    /// The value for the VMCB's N_CR3 field.
    pub fn ncr3(&self) -> Ncr3 {
        self.ncr3
    }

    /// How the map is laid out: where it ends, and which GiBs have a PD.
    pub fn layout(&self) -> MapLayout {
        self.layout
    }

    /// End of the identity map: GPAs `[0, end)` are mapped.
    pub fn end(&self) -> u64 {
        self.layout.end()
    }

    /// End of the UEFI memory map, rounded up to 1 GiB.
    pub fn memory_map_end(&self) -> u64 {
        self.memory_map_end
    }

    /// How many 2 MiB regions are mapped in 4 KiB pages, through a PT each.
    pub fn split_regions(&self) -> usize {
        self.split_regions
    }

    /// How many more PTs are set aside for regions the guest's MTRR writes
    /// split.
    pub fn spare_pts(&self) -> usize {
        self.spare_pts
    }

    /// Pages holding the tables, spare PTs included.
    pub fn table_pages(&self) -> usize {
        self.table_pages
    }
}

/// The tables [`Npt::build`] filled, as the exit handler needs them once
/// the guest runs: to change memory types, to hide Aleph0's pages, and to
/// read the map back. As `hal::intel::ept`'s `TypeTables`.
pub struct TypeTables {
    /// The PML4's physical address.
    pml4: u64,
    /// The PDPTs, back to back: entry g maps GiB g, `layout.gibs` of them.
    pdpts: *mut NptEntry<Pdpt>,
    layout: MapLayout,
    /// The PDs, back to back: entry i maps the 2 MiB region at i * 2 MiB.
    pds: *mut NptEntry<Pd>,
    regions: usize,
    /// Every PT, in use or spare, back to back; `pts_address` is the
    /// physical address of the first.
    pts: *mut NptEntry<Pt>,
    pt_count: usize,
    pts_address: u64,
    /// The free PTs' indices, a stack: its first `free_len` entries.
    free: *mut u32,
    free_len: usize,
    /// The 2 MiB regions holding the protected pages, which keep their PTs.
    protected_regions: [usize; MAX_PROTECTED],
    /// The shared zero page Aleph0's data pages are redirected to.
    shadow: u64,
    /// The ranges of the [`RamMap`] `build` gathered: `ram_len` of them.
    ram: *const (u64, u64),
    ram_len: usize,
    flush: FlushType,
    /// What the CPU refuses in an entry, for [`walk`], and whether NX is a
    /// permission.
    rules: EntryRules,
}

impl TypeTables {
    /// The RAM of the UEFI memory map.
    pub fn ram(&self) -> RamMap<'_> {
        // SAFETY: the ranges `build` gathered, `ram_len` of them, in an
        // allocation never freed and never written after `build`.
        let ranges = unsafe { core::slice::from_raw_parts(self.ram, self.ram_len) };
        RamMap { ranges }
    }

    /// Gives every page the memory type `mtrrs` gives it, or UC if it is not
    /// RAM ([`retype`]), and, if an entry changed, has the next VMRUN flush
    /// the guest's translations through `control`'s TLB_CONTROL: the AMD
    /// counterpart of `hal::intel::ept`'s INVEPT. Returns what changed.
    ///
    /// # Safety
    ///
    /// [`Npt::build`]'s tables still in place and the guest not running:
    /// exit context.
    pub unsafe fn apply(&mut self, mtrrs: &Mtrrs, control: &mut ControlArea) -> Retyped {
        // SAFETY: the PDs, PTs, free stack and RAM ranges `build` allocated,
        // as long as these slices and never freed. Nothing else refers to
        // them while `retype` runs: the guest is not running.
        let retyped = unsafe {
            let ram = RamMap {
                ranges: core::slice::from_raw_parts(self.ram, self.ram_len),
            };
            let mut tables = Retype {
                pds: core::slice::from_raw_parts_mut(self.pds, self.regions),
                pts: core::slice::from_raw_parts_mut(self.pts.cast::<[NptEntry<Pt>; ENTRIES]>(), self.pt_count),
                pts_address: self.pts_address,
                free: FreePts {
                    stack: core::slice::from_raw_parts_mut(self.free, self.pt_count),
                    len: &mut self.free_len,
                },
                protected_regions: self.protected_regions,
                rules: self.rules,
            };
            retype(&mut tables, &MapTypes { mtrrs, ram })
        };
        if retyped.changed() {
            control.tlb_control = self.flush.tlb_control();
        }
        retyped
    }

    /// Hides every page in `allocations` that Aleph0 keeps, but the shadow
    /// page and the pages in `visible`, as `hal::intel::ept`'s `TypeTables::hide`.
    /// Runs once, before the first VMRUN, so it needs no flush.
    ///
    /// # Safety
    ///
    /// Single-threaded context with [`Npt::build`]'s tables in place and the
    /// guest not yet run, so these slices are not aliased and no translation
    /// is stale.
    pub unsafe fn hide(
        &mut self,
        allocations: impl Iterator<Item = Allocation>,
        mtrrs: &Mtrrs,
        visible: &[Allocation],
    ) -> Result<Hidden, NptError> {
        debug_assert!(self.shadow != 0, "the shadow page was not allocated");
        // SAFETY: as `apply`: the PDs, PTs, free stack and RAM ranges
        // `build` allocated, exclusive here while the guest is stopped.
        let mut tables = unsafe {
            Hiding {
                pds: core::slice::from_raw_parts_mut(self.pds, self.regions),
                pts: core::slice::from_raw_parts_mut(self.pts.cast::<[NptEntry<Pt>; ENTRIES]>(), self.pt_count),
                free: FreePts {
                    stack: core::slice::from_raw_parts_mut(self.free, self.pt_count),
                    len: &mut self.free_len,
                },
                pts_address: self.pts_address,
                shadow: self.shadow,
                types: MapTypes {
                    mtrrs,
                    ram: RamMap {
                        ranges: core::slice::from_raw_parts(self.ram, self.ram_len),
                    },
                },
                rules: self.rules,
            }
        };
        let mut hidden = Hidden::default();
        for allocation in allocations {
            for i in 0..allocation.pages {
                let page = allocation.address + (i * PAGE_SIZE) as u64;
                if page == self.shadow || visible.iter().any(|v| v.contains(page)) {
                    continue;
                }
                hide_page(&mut tables, page, allocation.code, &mut hidden)?;
            }
        }
        Ok(hidden)
    }

    /// How [`apply`](Self::apply) has the guest's translations flushed.
    pub fn flush(&self) -> FlushType {
        self.flush
    }

    /// The N_CR3 of these tables: their PML4.
    pub fn ncr3(&self) -> Ncr3 {
        Ncr3::new(self.pml4)
    }

    /// Test only: makes the PDPT entry of `gpa`'s GiB one the CPU refuses,
    /// so the guest's access there is a nested page fault with RSVD set: the
    /// NPT counterpart of `hal::intel::ept`'s `misconfigure`. Address bit 51
    /// is reserved below a 52-bit MAXPHYADDR; above it, NX without host
    /// EFER.NXE; with both, the entry loses U/S, which every nested walk
    /// needs (a fault without RSVD).
    ///
    /// # Safety
    ///
    /// The CPU must not be using the tables: before the first VMRUN.
    #[cfg(feature = "ept-misconfig-test")]
    pub unsafe fn misconfigure(&self, gpa: u64) {
        let gib = (gpa / SIZE_1G) as usize;
        assert!(gib < self.layout.gibs, "GPA {gpa:#x} is outside the map");
        // SAFETY: entry `gib` of the PDPTs `build` filled, `layout.gibs` of
        // them, never freed; the caller guarantees the CPU is not using them.
        unsafe {
            let entry = self.pdpts.add(gib);
            let raw = entry.read_volatile().raw();
            let broken = if self.rules.physical_address_bits < 52 {
                raw | (1 << 51)
            } else if !self.rules.nx {
                raw | NO_EXECUTE
            } else {
                raw & !USER
            };
            entry.write_volatile(NptEntry::from_raw(broken));
        }
    }

    /// Walks `gpa` through the tables `ncr3` names ([`walk`]), reading only
    /// the table pages `build` allocated: a pointer to anything else is
    /// reported, never followed. For the exit handler's report of a nested
    /// page fault.
    pub fn walk(&self, ncr3: Ncr3, gpa: u64) -> NptWalk {
        walk(ncr3, gpa, self.rules, |address| {
            self.contains_entry(address).then(|| {
                // SAFETY: `address` is an aligned entry inside one of the
                // table pages `build` allocated, which are never freed and
                // identity-mapped. Only `apply`, `hide` and `grant_write`
                // write them, never during a walk.
                unsafe { (address as *const u64).read_volatile() }
            })
        })
    }

    /// Whether `address` is an 8-byte aligned entry inside one of the
    /// table pages `build` allocated.
    fn contains_entry(&self, address: u64) -> bool {
        let tables = [
            (self.pml4, 1),
            (self.pdpts as u64, self.layout.pdpts()),
            (self.pds as u64, self.layout.pd_gibs),
            (self.pts_address, self.pt_count),
        ];
        address.is_multiple_of(size_of::<u64>() as u64)
            && tables.iter().any(|&(start, pages)| {
                let len = (pages as u64).saturating_mul(PAGE_SIZE as u64);
                (start..start.saturating_add(len)).contains(&address)
            })
    }

    /// How many PTs are free, set aside or given back.
    pub fn free_pts(&self) -> usize {
        self.free_len
    }

    /// The map from `start` to `end` as its entries hold it: runs of 2 MiB
    /// regions mapped the same way by the PDs, then runs of 1 GiB pages
    /// above them.
    pub fn region_runs(&self, start: u64, end: u64) -> impl Iterator<Item = RegionRun> + '_ {
        let first = ((start / SIZE_2M) as usize).min(self.regions);
        let last = (end.div_ceil(SIZE_2M) as usize).clamp(first, self.regions);
        let first_gib = ((start / SIZE_1G) as usize).clamp(self.layout.pd_gibs, self.layout.gibs);
        let last_gib = (end.div_ceil(SIZE_1G) as usize).clamp(first_gib, self.layout.gibs);
        // SAFETY: the PDs and PDPTs `build` filled, never freed. Only
        // `apply` writes them, through `&mut self`, so never while this
        // iterator borrows `self`.
        let (pds, pdpts) = unsafe {
            (
                core::slice::from_raw_parts(self.pds, self.regions),
                core::slice::from_raw_parts(self.pdpts, self.layout.gibs),
            )
        };
        region_runs(&pds[first..last], first).chain(gib_runs(&pdpts[first_gib..last_gib], first_gib))
    }
}

/// The parts of an identity map [`retype`] changes.
struct Retype<'a> {
    /// One entry per 2 MiB region, from GPA 0.
    pds: &'a mut [NptEntry<Pd>],
    /// Every PT, in use or free; `pts_address` is the first one's physical
    /// address.
    pts: &'a mut [[NptEntry<Pt>; ENTRIES]],
    pts_address: u64,
    free: FreePts<'a>,
    /// The 2 MiB regions holding the protected pages, which keep their PTs.
    protected_regions: [usize; MAX_PROTECTED],
    rules: EntryRules,
}

/// Gives every region of `tables` the memory types `types` give its pages,
/// changing only the entries that differ, and returns what changed: the
/// algorithm of `hal::intel::ept`'s `retype`, on NPT entries. A region whose pages
/// share one type is one 2 MiB page; any other, or a protected page's, a PT.
/// The regions that need a PT come last, in a second pass, so that every PT
/// this call gives back is free by then; with none left a region stays one
/// UC 2 MiB page.
fn retype(tables: &mut Retype<'_>, types: &MapTypes<'_>) -> Retyped {
    let protected_regions = tables.protected_regions;
    let rules = tables.rules;
    let wanted = |i: usize| {
        if protected_regions.contains(&i) {
            None
        } else {
            types.region_type(i as u64 * SIZE_2M)
        }
    };
    let mut retyped = Retyped::default();
    let mut needing_pts: Option<(usize, usize)> = None;
    for (i, entry) in tables.pds.iter_mut().enumerate() {
        let region = i as u64 * SIZE_2M;
        let pt = pt_index(*entry, tables.pts_address, tables.pts.len());
        let want = if pt
            .and_then(|index| tables.pts.get(index))
            .is_some_and(|pt| pt_pinned(pt, region, rules))
        {
            None
        } else {
            wanted(i)
        };
        let changed = match (entry.mapping(), want) {
            (RegionMapping::LargePage(current), Some(memory_type)) => {
                let change = current != memory_type;
                if change {
                    *entry = NptEntry::large_page(region, memory_type);
                }
                change
            }
            (RegionMapping::Pt, Some(memory_type)) => {
                *entry = NptEntry::large_page(region, memory_type);
                if let Some(index) = pt {
                    tables.free.give_back(index);
                }
                retyped.merged += 1;
                true
            }
            (RegionMapping::Pt, None) => pt
                .and_then(|index| tables.pts.get_mut(index))
                .is_some_and(|pt| retype_pt(pt, region, types)),
            (RegionMapping::LargePage(_), None) => {
                needing_pts = Some(needing_pts.map_or((i, i), |(first, _)| (first, i)));
                false
            }
            (RegionMapping::Invalid | RegionMapping::HugePage(_), _) => false,
        };
        if changed {
            retyped.count(i);
        }
    }

    let Some((first, last)) = needing_pts else {
        return retyped;
    };
    for (i, entry) in tables.pds.iter_mut().enumerate().take(last + 1).skip(first) {
        let RegionMapping::LargePage(current) = entry.mapping() else {
            continue;
        };
        if wanted(i).is_some() {
            continue;
        }
        let region = i as u64 * SIZE_2M;
        let changed = match tables.free.take().filter(|&index| index < tables.pts.len()) {
            Some(index) => {
                fill_pt(&mut tables.pts[index], region, types, rules);
                *entry = NptEntry::table(tables.pts_address + (index * PAGE_SIZE) as u64);
                retyped.split += 1;
                true
            }
            None => {
                retyped.uncached += 1;
                let change = current != MemoryType::Uncacheable;
                if change {
                    *entry = NptEntry::large_page(region, MemoryType::Uncacheable);
                }
                change
            }
        };
        if changed {
            retyped.count(i);
        }
    }
    retyped
}

/// The index, among `count` PTs from `pts_address`, of the PT the PD entry
/// `entry` points to, or `None` when it points to none of them.
fn pt_index(entry: NptEntry<Pd>, pts_address: u64, count: usize) -> Option<usize> {
    let offset = (entry.raw & ADDRESS_MASK).checked_sub(pts_address)?;
    let index = (offset / PAGE_SIZE as u64) as usize;
    (index < count).then_some(index)
}

/// Gives each page of `pt`, which maps the 2 MiB region at `region`, the
/// memory type `types` gives it, keeping its access. Returns whether any
/// entry changed.
fn retype_pt(pt: &mut [NptEntry<Pt>; ENTRIES], region: u64, types: &MapTypes<'_>) -> bool {
    let mut changed = false;
    for (j, page) in pt.iter_mut().enumerate() {
        let retyped = page.with_memory_type(types.page_type(region + (j * PAGE_SIZE) as u64));
        changed |= retyped != *page;
        *page = retyped;
    }
    changed
}

/// The page [`Npt::build`] maps read-only, with what making it writable
/// needs: its PT entry, and how to flush. As `hal::intel::ept`'s `ProtectedPage`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ProtectedPage {
    /// Guest-physical address of the page.
    pub gpa: u64,
    /// Its entry in the PT [`Npt::build`] filled for its region.
    entry: *mut NptEntry<Pt>,
    flush: FlushType,
    rules: EntryRules,
}

impl ProtectedPage {
    /// A placeholder for the unused slots of a fixed-size array. Never
    /// published.
    const ABSENT: Self = Self {
        gpa: 0,
        entry: core::ptr::null_mut(),
        flush: FlushType::All,
        rules: EntryRules {
            physical_address_bits: 0,
            huge_pages: false,
            nx: false,
        },
    };

    /// Makes the page writable, and has the next VMRUN flush the guest's
    /// translations through `control`'s TLB_CONTROL. Returns the flush used,
    /// for the report. A nested page fault drops the faulting translation
    /// already; the flush is the architectural way to publish the change.
    ///
    /// # Safety
    ///
    /// [`Npt::build`]'s tables still in place. Called from the exit handler,
    /// while the guest is not running.
    pub unsafe fn grant_write(&self, control: &mut ControlArea) -> FlushType {
        // SAFETY: `entry` points into the PT `Npt::build` filled, which is
        // never freed, and the guest is not running to race with the write.
        unsafe {
            let entry = self.entry.read_volatile();
            self.entry
                .write_volatile(entry.with_access(entry.access() | NPT_WRITE, self.rules));
        }
        control.tlb_control = self.flush.tlb_control();
        self.flush
    }
}

// What `Npt::build` publishes for the exit handler, as `hal::intel::ept` does: up to
// [`MAX_PROTECTED`] read-only pages, all with one flush type and the same
// rules. Written once before the first VMRUN; read only in exit context.
static PROTECTED_GPAS: [AtomicU64; MAX_PROTECTED] = [const { AtomicU64::new(0) }; MAX_PROTECTED];
static PROTECTED_ENTRIES: [AtomicPtr<NptEntry<Pt>>; MAX_PROTECTED] =
    [const { AtomicPtr::new(core::ptr::null_mut()) }; MAX_PROTECTED];
static PROTECTED_COUNT: AtomicUsize = AtomicUsize::new(0);
/// The rules, packed: MAXPHYADDR in bits 7:0, 1 GiB pages in bit 8, NX in
/// bit 9.
static PUBLISHED_RULES: AtomicU64 = AtomicU64::new(0);
/// The [`FlushType`] (1 = ASID, 2 = all), and the "published" flag: 0 until
/// [`publish`] has stored everything else. Written last (Release), read
/// first (Acquire).
static FLUSH_TYPE: AtomicU8 = AtomicU8::new(0);

/// The flush type stored as `value`, or `None` for anything [`publish`]
/// never stores.
fn decode_flush(value: u8) -> Option<FlushType> {
    match value {
        1 => Some(FlushType::Asid),
        2 => Some(FlushType::All),
        _ => None,
    }
}

fn encode_flush(flush: FlushType) -> u8 {
    match flush {
        FlushType::Asid => 1,
        FlushType::All => 2,
    }
}

fn encode_rules(rules: EntryRules) -> u64 {
    u64::from(rules.physical_address_bits) | (u64::from(rules.huge_pages) << 8) | (u64::from(rules.nx) << 9)
}

fn decode_rules(value: u64) -> EntryRules {
    EntryRules {
        physical_address_bits: value as u8,
        huge_pages: value & (1 << 8) != 0,
        nx: value & (1 << 9) != 0,
    }
}

/// Publishes `pages` for [`protected_page_for`]. Refuses a second
/// publication.
fn publish(pages: &[ProtectedPage]) -> Result<(), NptError> {
    if decode_flush(FLUSH_TYPE.load(Ordering::Acquire)).is_some() {
        return Err(NptError::AlreadyBuilt);
    }
    debug_assert!(pages.len() <= MAX_PROTECTED);
    for (slot, page) in pages.iter().enumerate() {
        PROTECTED_GPAS[slot].store(page.gpa, Ordering::Relaxed);
        PROTECTED_ENTRIES[slot].store(page.entry, Ordering::Relaxed);
    }
    PROTECTED_COUNT.store(pages.len(), Ordering::Relaxed);
    if let Some(first) = pages.first() {
        PUBLISHED_RULES.store(encode_rules(first.rules), Ordering::Relaxed);
        FLUSH_TYPE.store(encode_flush(first.flush), Ordering::Release);
    }
    Ok(())
}

/// The published read-only page whose 4 KiB frame holds `gpa`, or `None`
/// when no page was published yet or `gpa` is in none of them. The exit
/// handler calls it with a nested page fault's GPA (EXITINFO2).
pub fn protected_page_for(gpa: u64) -> Option<ProtectedPage> {
    let flush = decode_flush(FLUSH_TYPE.load(Ordering::Acquire))?;
    let rules = decode_rules(PUBLISHED_RULES.load(Ordering::Relaxed));
    let count = PROTECTED_COUNT.load(Ordering::Relaxed);
    let frame = gpa & !(PAGE_SIZE as u64 - 1);
    let slot = PROTECTED_GPAS[..count]
        .iter()
        .position(|slot| slot.load(Ordering::Relaxed) == frame)?;
    Some(ProtectedPage {
        gpa: PROTECTED_GPAS[slot].load(Ordering::Relaxed),
        entry: PROTECTED_ENTRIES[slot].load(Ordering::Relaxed),
        flush,
        rules,
    })
}

/// GPA of the second published read-only page, the guest's stack page, as
/// `hal::intel::ept::nmi_stack_page`. 0 before [`Npt::build`], or when fewer than
/// two pages were published.
pub fn nmi_stack_page() -> u64 {
    if decode_flush(FLUSH_TYPE.load(Ordering::Acquire)).is_none() {
        return 0;
    }
    const STACK_SLOT: usize = 1;
    if PROTECTED_COUNT.load(Ordering::Relaxed) > STACK_SLOT {
        PROTECTED_GPAS[STACK_SLOT].load(Ordering::Relaxed)
    } else {
        0
    }
}

#[cfg(test)]
#[path = "../../../../tests/unit/hal/amd/npt.rs"]
mod tests;
