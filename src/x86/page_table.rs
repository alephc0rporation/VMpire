//! The shape x86-64 4-level page tables share with EPT and NPT: four levels
//! of 512 eight-byte entries per 4 KiB table, the same 9-bit index per
//! level, and the page sizes a level maps. Only the entry bits differ, and
//! those stay with each table kind (`host`, `hal::intel::ept`, `hal::amd::npt`).

// Used by the AMD backend only so far (`hal::amd::npt`), which is not run
// until bring-up picks it.
#![allow(dead_code, reason = "only the AMD backend uses this so far")]

pub use super::paging::{SIZE_1G, SIZE_2M};

/// Entries per table: 4096 bytes / 8 bytes per entry.
pub const ENTRIES: usize = 512;
/// Bit 7 of a PDPT or PD entry (PS): the entry maps a 1 GiB or 2 MiB page
/// itself instead of pointing to a table.
pub const PAGE_SIZE_BIT: u64 = 1 << 7;

pub use super::paging::level;

use self::level::{Level, Pd, Pdpt, Pml4, Pt};

/// A table level of the walk, from the root down.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WalkLevel {
    Pml4,
    Pdpt,
    Pd,
    Pt,
}

impl WalkLevel {
    /// The level's name: "PML4", "PDPT", "PD" or "PT".
    pub fn name(self) -> &'static str {
        match self {
            Self::Pml4 => Pml4::NAME,
            Self::Pdpt => Pdpt::NAME,
            Self::Pd => Pd::NAME,
            Self::Pt => Pt::NAME,
        }
    }

    /// Index of `gpa`'s entry in a table of this level.
    pub fn index(self, gpa: u64) -> usize {
        match self {
            Self::Pml4 => Pml4::index(gpa),
            Self::Pdpt => Pdpt::index(gpa),
            Self::Pd => Pd::index(gpa),
            Self::Pt => Pt::index(gpa),
        }
    }

    /// The level of the table an entry here points to; `None` for a PT,
    /// whose entries always map a page.
    pub fn below(self) -> Option<Self> {
        match self {
            Self::Pml4 => Some(Self::Pdpt),
            Self::Pdpt => Some(Self::Pd),
            Self::Pd => Some(Self::Pt),
            Self::Pt => None,
        }
    }

    /// Whether the entry `raw` at this level maps a page itself: every PT
    /// entry, and a PDPT or PD entry with bit 7 set. A PML4 entry never
    /// does: bit 7 is reserved there. Bit 7 means this in EPT, NPT and the
    /// ordinary page tables alike.
    pub fn is_leaf(self, raw: u64) -> bool {
        match self {
            Self::Pml4 => false,
            Self::Pdpt | Self::Pd => raw & PAGE_SIZE_BIT != 0,
            Self::Pt => true,
        }
    }

    /// Bytes a page this level's entry maps: 4 KiB at a PT, 2 MiB at a PD,
    /// 1 GiB at a PDPT.
    pub fn page_size(self) -> u64 {
        let shift = match self {
            Self::Pml4 => Pml4::SHIFT,
            Self::Pdpt => Pdpt::SHIFT,
            Self::Pd => Pd::SHIFT,
            Self::Pt => Pt::SHIFT,
        };
        1 << shift
    }
}

/// Bits `low..=high` set, as KVM's `rsvd_bits`.
pub const fn bits(low: u32, high: u32) -> u64 {
    (u64::MAX >> (63 - high)) & (u64::MAX << low)
}
