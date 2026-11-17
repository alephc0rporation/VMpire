//! Unit tests for `src/hal/amd/register_access.rs`.
//!
//! Compiled only by `cargo test-host`. That file includes this one with
//! `#[path]` as its child module `tests`, so `use super::*` reaches its
//! private items.

use super::*;
use std::collections::HashSet;

/// A backend over a plain array, for testing the trait's own methods.
/// Registers in `failing` cannot be read.
struct ArrayRegisters {
    values: [u64; GuestRegister::COUNT],
    failing: &'static [GuestRegister],
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Unreadable;

impl GuestRegisters for ArrayRegisters {
    type Error = Unreadable;

    fn read(&self, register: GuestRegister) -> Result<u64, Unreadable> {
        if self.failing.contains(&register) {
            Err(Unreadable)
        } else {
            Ok(self.values[register.index()])
        }
    }

    fn write(&mut self, register: GuestRegister, value: u64) -> Result<(), Unreadable> {
        self.values[register.index()] = value;
        Ok(())
    }
}

/// A distinct value per register: 0x1000 plus its index.
fn distinct_values() -> [u64; GuestRegister::COUNT] {
    core::array::from_fn(|index| 0x1000 + index as u64)
}

#[test]
fn all_lists_every_register_once_at_its_index() {
    assert_eq!(GuestRegister::ALL.len(), GuestRegister::COUNT);
    for (position, register) in GuestRegister::ALL.iter().enumerate() {
        assert_eq!(register.index(), position, "{register}");
    }
    let unique: HashSet<GuestRegister> = GuestRegister::ALL.iter().copied().collect();
    assert_eq!(unique.len(), GuestRegister::COUNT);
}

#[test]
fn the_first_sixteen_are_the_general_purpose_registers() {
    let general: Vec<&str> = GuestRegister::ALL
        .iter()
        .filter(|register| register.is_general_purpose())
        .map(|register| register.name())
        .collect();
    assert_eq!(
        general,
        [
            "rax", "rbx", "rcx", "rdx", "rsi", "rdi", "rbp", "rsp", "r8", "r9", "r10", "r11", "r12", "r13", "r14",
            "r15"
        ]
    );
    assert!(
        GuestRegister::ALL[..16]
            .iter()
            .all(|register| register.is_general_purpose())
    );
    assert!(
        !GuestRegister::ALL[16..]
            .iter()
            .any(|register| register.is_general_purpose())
    );
}

#[test]
fn names_are_unique_and_lowercase() {
    let names: HashSet<&str> = GuestRegister::ALL.iter().map(|register| register.name()).collect();
    assert_eq!(names.len(), GuestRegister::COUNT);
    for name in names {
        assert_eq!(name, name.to_lowercase());
    }
    assert_eq!(GuestRegister::FsBase.to_string(), "fs_base");
}

#[test]
fn read_all_reads_every_register_in_order() {
    let registers = ArrayRegisters {
        values: distinct_values(),
        failing: &[],
    };
    let snapshot = registers.read_all();
    for (register, value) in snapshot.iter() {
        assert_eq!(value, Ok(0x1000 + register.index() as u64), "{register}");
    }
    let order: Vec<GuestRegister> = snapshot.iter().map(|(register, _)| register).collect();
    assert_eq!(order, GuestRegister::ALL);
}

#[test]
fn read_all_keeps_an_error_and_goes_on() {
    let registers = ArrayRegisters {
        values: distinct_values(),
        failing: &[GuestRegister::Dr7, GuestRegister::Efer],
    };
    let snapshot = registers.read_all();
    assert_eq!(snapshot.get(GuestRegister::Dr7), Err(Unreadable));
    assert_eq!(snapshot.get(GuestRegister::Efer), Err(Unreadable));
    // The registers after the failing ones were still read.
    assert_eq!(
        snapshot.get(GuestRegister::Pat),
        Ok(0x1000 + GuestRegister::Pat.index() as u64)
    );
    assert_eq!(snapshot.iter().filter(|(_, value)| value.is_err()).count(), 2);
}

#[test]
fn a_write_shows_in_the_next_read_all() {
    let mut registers = ArrayRegisters {
        values: [0; GuestRegister::COUNT],
        failing: &[],
    };
    registers.write(GuestRegister::Rip, 0x7ead_4009).unwrap();
    let snapshot = registers.read_all();
    assert_eq!(snapshot.get(GuestRegister::Rip), Ok(0x7ead_4009));
    assert_eq!(snapshot.iter().filter(|&(_, value)| value != Ok(0)).count(), 1);
}
