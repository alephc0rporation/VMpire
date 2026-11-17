//! Reading the NPT tables back the way the CPU walks them, as `hal::intel::ept`'s
//! `walk.rs` does for EPT: what the CPU refuses in an entry
//! ([`check_entry`]), and the path of one GPA from the nested CR3 down
//! ([`walk`]).
//!
//! A nested page fault (#VMEXIT NPF) is either a walk that found an entry
//! not present, one that does not allow the access (a write to a read-only
//! page, a fetch from an NX one, or an entry without U/S, since every nested
//! walk is a user access), or one with a reserved bit set: there is no
//! separate misconfiguration exit as on Intel, only the RSVD bit of the
//! fault's error code. Aleph0's map allows every access but those to the
//! protected pages, so any other nested page fault means the map is wrong,
//! and the exit handler prints the walk.
//!
//! The reserved bits are the long-mode page-table formats (AMD64 APM Vol. 2,
//! 5.3.3 and 5.4.1), as KVM's `__reset_rsvds_bits_mask` lists them for an
//! AMD CPU. Both functions are pure. [`walk`] reads entries through a
//! function the caller gives it, which returns `None` for an address outside
//! the tables, so a corrupt table pointer is reported, never followed.

use core::fmt;

use super::{
    ADDRESS_MASK, EntryRules, NO_EXECUTE, NPT_EXECUTE, NPT_READ, NPT_WRITE, Ncr3, PAT_4K, PAT_LARGE, PRESENT, USER,
    WRITABLE, leaf_pat_index, pat_type,
};
use crate::x86::identity_map::WALK_ADDRESS_BITS;
use crate::x86::mtrr::MemoryType;
use crate::x86::page_table::{PAGE_SIZE_BIT, WalkLevel, bits};

/// Bit 8 of a PML4 entry: reserved on AMD in an entry pointing to a table
/// (KVM's `nonleaf_bit8_rsvd`; it would be G in a leaf).
const NONLEAF_BIT_8: u64 = 1 << 8;

/// The bits an entry at `level` must keep 0: one that points to a table,
/// or with `leaf` one that maps a page.
fn reserved_bits(level: WalkLevel, leaf: bool, rules: EntryRules) -> u64 {
    // Address bits from MAXPHYADDR up to 51, at every level.
    let address = match rules.physical_address_bits {
        bits_used @ 0..=51 => bits(u32::from(bits_used), 51),
        _ => 0,
    };
    // NX is a permission only with host EFER.NXE.
    let no_execute = if rules.nx { 0 } else { NO_EXECUTE };
    address
        | no_execute
        | match (level, leaf) {
            // No page size in a PML4 entry, and bit 8 reserved on AMD.
            (WalkLevel::Pml4, _) => PAGE_SIZE_BIT | NONLEAF_BIT_8,
            (WalkLevel::Pdpt | WalkLevel::Pd, false) => 0,
            // A 1 GiB page's address bits below its alignment, bit 12 being
            // its PAT bit; and the page size bit itself on a CPU without
            // 1 GiB pages.
            (WalkLevel::Pdpt, true) => bits(13, 29) | if rules.huge_pages { 0 } else { PAGE_SIZE_BIT },
            // A 2 MiB page's address bits below its alignment, bit 12 being
            // its PAT bit.
            (WalkLevel::Pd, true) => bits(13, 20),
            (WalkLevel::Pt, _) => 0,
        }
}

/// What the CPU refuses in one NPT entry ([`check_entry`]): a walk that
/// reaches it is a nested page fault with RSVD set.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct EntryFaults {
    /// The bits set that the entry's level reserves.
    pub reserved_bits: u64,
}

impl EntryFaults {
    /// Whether the CPU uses the entry as it is.
    pub fn is_empty(&self) -> bool {
        *self == Self::default()
    }
}

impl fmt::Display for EntryFaults {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_empty() {
            f.write_str("ok")
        } else {
            write!(f, "reserved bits {:#x} set", self.reserved_bits)
        }
    }
}

/// What the CPU refuses in `raw`, an entry at `level`, on a CPU with
/// `rules`. Nothing for an entry that is not present: a walk reaching it is
/// a not-present fault, whatever else it holds.
pub fn check_entry(raw: u64, level: WalkLevel, rules: EntryRules) -> EntryFaults {
    if raw & PRESENT == 0 {
        return EntryFaults::default();
    }
    EntryFaults {
        reserved_bits: raw & reserved_bits(level, level.is_leaf(raw), rules),
    }
}

/// One entry a [`walk`] read.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WalkedEntry {
    pub level: WalkLevel,
    /// Its index in its table.
    pub index: usize,
    /// Its physical address.
    pub address: u64,
    pub raw: u64,
    /// What the CPU refuses in it.
    pub faults: EntryFaults,
}

/// How a [`walk`] ended.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WalkEnd {
    /// A page of `size` bytes maps the GPA to `hpa`, allowing `access`
    /// ([`NPT_READ`], [`NPT_WRITE`], [`NPT_EXECUTE`]: what every level on the
    /// path allows together), of `memory_type` (`None` for UC-, a PAT entry
    /// Aleph0 never uses).
    Mapped {
        hpa: u64,
        size: u64,
        access: u64,
        memory_type: Option<MemoryType>,
    },
    /// The last entry is not present: nothing maps the GPA.
    NotPresent,
    /// The last entry has a reserved bit set: a fault with RSVD.
    Reserved,
    /// An entry on the path lacks U/S: every nested walk is a user access,
    /// so the walk faults there.
    Supervisor,
    /// The last entry points to `table` (or the nested CR3 does, with no
    /// entry read), which is not one of the tables: not followed.
    OutsideTables { table: u64 },
    /// The GPA has a bit set above the 48 a 4-level walk translates.
    BeyondWalk,
}

impl fmt::Display for WalkEnd {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match *self {
            Self::Mapped {
                hpa,
                size,
                access,
                memory_type,
            } => {
                let size = match size {
                    0x1000 => "4 KiB",
                    0x20_0000 => "2 MiB",
                    _ => "1 GiB",
                };
                let memory_type = memory_type.map_or("UC-", MemoryType::name);
                write!(
                    f,
                    "mapped to HPA {hpa:#x} by a {size} page, {memory_type}, R={} W={} X={}",
                    access & NPT_READ,
                    (access & NPT_WRITE) >> 1,
                    (access & NPT_EXECUTE) >> 2,
                )
            }
            Self::NotPresent => f.write_str("not present: nothing maps the GPA"),
            Self::Reserved => f.write_str("the last entry has a reserved bit set: a fault with RSVD"),
            Self::Supervisor => f.write_str("an entry lacks U/S, which every nested walk needs"),
            Self::OutsideTables { table } => {
                write!(f, "points to {table:#x}, none of Aleph0's NPT tables: not followed")
            }
            Self::BeyondWalk => f.write_str("beyond the 48 bits a 4-level walk translates: nothing maps the GPA"),
        }
    }
}

/// The path of one GPA through the tables ([`walk`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NptWalk {
    pub gpa: u64,
    entries: [Option<WalkedEntry>; 4],
    pub end: WalkEnd,
}

impl NptWalk {
    /// The entries the walk read, from the PML4 down.
    pub fn entries(&self) -> impl Iterator<Item = &WalkedEntry> {
        self.entries.iter().flatten()
    }
}

/// Walks `gpa` through the tables `ncr3` names, as the CPU does: one entry
/// per level from the PML4 down, until one maps a page, is not present, has
/// a reserved bit set ([`check_entry`]) or lacks U/S. The access a page
/// allows is what every level allows together: R/W and U/S are ANDed down
/// the path, NX ORed. `read` returns the 8-byte entry at a physical address,
/// or `None` for an address outside the tables.
pub fn walk(ncr3: Ncr3, gpa: u64, rules: EntryRules, read: impl Fn(u64) -> Option<u64>) -> NptWalk {
    let mut entries = [None; 4];
    if gpa >> WALK_ADDRESS_BITS != 0 {
        return NptWalk {
            gpa,
            entries,
            end: WalkEnd::BeyondWalk,
        };
    }
    let mut table = ncr3.pml4();
    let mut level = WalkLevel::Pml4;
    let mut writable = true;
    let mut executable = true;
    for slot in &mut entries {
        let index = level.index(gpa);
        let address = table + (index * size_of::<u64>()) as u64;
        let Some(raw) = read(address) else {
            return NptWalk {
                gpa,
                entries,
                end: WalkEnd::OutsideTables { table },
            };
        };
        let faults = check_entry(raw, level, rules);
        *slot = Some(WalkedEntry {
            level,
            index,
            address,
            raw,
            faults,
        });
        writable &= raw & WRITABLE != 0;
        executable &= raw & NO_EXECUTE == 0;
        let end = if raw & PRESENT == 0 {
            Some(WalkEnd::NotPresent)
        } else if !faults.is_empty() {
            Some(WalkEnd::Reserved)
        } else if raw & USER == 0 {
            Some(WalkEnd::Supervisor)
        } else if level.is_leaf(raw) {
            let size = level.page_size();
            let pat = if level == WalkLevel::Pt { PAT_4K } else { PAT_LARGE };
            Some(WalkEnd::Mapped {
                hpa: (raw & ADDRESS_MASK & !(size - 1)) | (gpa & (size - 1)),
                size,
                access: NPT_READ | (if writable { NPT_WRITE } else { 0 }) | (if executable { NPT_EXECUTE } else { 0 }),
                memory_type: pat_type(leaf_pat_index(raw, pat)),
            })
        } else {
            None
        };
        if let Some(end) = end {
            return NptWalk { gpa, entries, end };
        }
        table = raw & ADDRESS_MASK;
        level = level.below().expect("a PT entry always maps a page");
    }
    unreachable!("a PT entry always ends the walk")
}

#[cfg(test)]
#[path = "../../../../tests/unit/hal/amd/npt/walk.rs"]
mod tests;
