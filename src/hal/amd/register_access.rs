//! The guest's registers, as the shared logic sees them in exit context.
//!
//! [`GuestRegister`] lists every register the trait covers, and
//! [`GuestRegister::ALL`] lets a caller loop over them. The list holds the
//! registers both Intel and AMD give the hypervisor after an exit: the 16
//! general-purpose registers, RIP, RFLAGS, CR0, CR3, CR4, DR7, EFER, PAT and
//! the FS and GS bases. Where each one lives (a VMCS field, a VMCB field, the
//! frame the exit stub saved, or still the MSR) is the backend's business.

// Nothing outside the host tests calls these yet: the exit handler moves
// over to them once the backends are split (HAL plan, phase 3).
#![allow(dead_code, reason = "not wired into the exit handler yet")]

use core::fmt;

/// A guest register the hypervisor can read or write after an exit.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum GuestRegister {
    Rax,
    Rbx,
    Rcx,
    Rdx,
    Rsi,
    Rdi,
    Rbp,
    Rsp,
    R8,
    R9,
    R10,
    R11,
    R12,
    R13,
    R14,
    R15,
    Rip,
    Rflags,
    Cr0,
    Cr3,
    Cr4,
    Dr7,
    Efer,
    Pat,
    FsBase,
    GsBase,
}

impl GuestRegister {
    /// How many registers [`ALL`](Self::ALL) holds.
    pub const COUNT: usize = 26;

    /// The 16 general-purpose registers come first, in their encoding order
    /// apart from RSP's place, then the others. Each register's position is
    /// its [`index`](Self::index).
    pub const ALL: [Self; Self::COUNT] = [
        Self::Rax,
        Self::Rbx,
        Self::Rcx,
        Self::Rdx,
        Self::Rsi,
        Self::Rdi,
        Self::Rbp,
        Self::Rsp,
        Self::R8,
        Self::R9,
        Self::R10,
        Self::R11,
        Self::R12,
        Self::R13,
        Self::R14,
        Self::R15,
        Self::Rip,
        Self::Rflags,
        Self::Cr0,
        Self::Cr3,
        Self::Cr4,
        Self::Dr7,
        Self::Efer,
        Self::Pat,
        Self::FsBase,
        Self::GsBase,
    ];

    /// Its position in [`ALL`](Self::ALL) and in a [`RegisterSnapshot`].
    pub const fn index(self) -> usize {
        self as usize
    }

    /// Whether it is one of the 16 general-purpose registers.
    pub const fn is_general_purpose(self) -> bool {
        self.index() < 16
    }

    /// Its lowercase name, as the reports print it.
    pub const fn name(self) -> &'static str {
        match self {
            Self::Rax => "rax",
            Self::Rbx => "rbx",
            Self::Rcx => "rcx",
            Self::Rdx => "rdx",
            Self::Rsi => "rsi",
            Self::Rdi => "rdi",
            Self::Rbp => "rbp",
            Self::Rsp => "rsp",
            Self::R8 => "r8",
            Self::R9 => "r9",
            Self::R10 => "r10",
            Self::R11 => "r11",
            Self::R12 => "r12",
            Self::R13 => "r13",
            Self::R14 => "r14",
            Self::R15 => "r15",
            Self::Rip => "rip",
            Self::Rflags => "rflags",
            Self::Cr0 => "cr0",
            Self::Cr3 => "cr3",
            Self::Cr4 => "cr4",
            Self::Dr7 => "dr7",
            Self::Efer => "efer",
            Self::Pat => "pat",
            Self::FsBase => "fs_base",
            Self::GsBase => "gs_base",
        }
    }
}

impl fmt::Display for GuestRegister {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

/// Reads and writes the guest's registers after an exit.
///
/// A write takes effect when the guest is next entered. Each backend says in
/// its `Error` why a register cannot be read or written, for example a VMCS
/// field that cannot be read, or a register the exit did not save.
pub trait GuestRegisters {
    type Error: Copy + fmt::Debug;

    /// The guest's value of `register`.
    fn read(&self, register: GuestRegister) -> Result<u64, Self::Error>;

    /// Gives the guest `value` in `register`.
    fn write(&mut self, register: GuestRegister, value: u64) -> Result<(), Self::Error>;

    /// Reads every register in [`GuestRegister::ALL`], in that order. A
    /// register that cannot be read keeps its error, and the loop goes on to
    /// the next one.
    fn read_all(&self) -> RegisterSnapshot<Self::Error> {
        RegisterSnapshot {
            values: core::array::from_fn(|index| self.read(GuestRegister::ALL[index])),
        }
    }
}

/// The result of [`GuestRegisters::read_all`]: one value or error for each
/// register, indexed by [`GuestRegister::index`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RegisterSnapshot<E> {
    values: [Result<u64, E>; GuestRegister::COUNT],
}

impl<E: Copy> RegisterSnapshot<E> {
    /// What was read for `register`.
    pub fn get(&self, register: GuestRegister) -> Result<u64, E> {
        self.values[register.index()]
    }

    /// Every register with what was read for it, in [`GuestRegister::ALL`]
    /// order.
    pub fn iter(&self) -> impl Iterator<Item = (GuestRegister, Result<u64, E>)> + '_ {
        GuestRegister::ALL
            .iter()
            .map(|&register| (register, self.get(register)))
    }
}

#[cfg(test)]
#[path = "../../../tests/unit/hal/amd/register_access.rs"]
mod tests;
