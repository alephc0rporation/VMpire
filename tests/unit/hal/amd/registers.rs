//! Unit tests for `src/hal/amd/registers.rs`.
//!
//! Compiled only by `cargo test-host`. That file includes this one with
//! `#[path]` as its child module `tests`, so `use super::*` reaches its
//! private items.
//!
//! The VMCB is plain memory, so unlike VMCS access these tests exercise the
//! backend exactly as the run loop will.

use super::*;
use crate::hal::amd::vmcb::Vmcb;

/// Where each register must live: `Some(offset)` from the start of the VMCB
/// (APM Vol. 2, Table B-2, plus 0x400), or `None` for the frame.
fn expected_vmcb_offset(register: GuestRegister) -> Option<usize> {
    match register {
        GuestRegister::Rax => Some(0x5F8),
        GuestRegister::Rsp => Some(0x5D8),
        GuestRegister::Rip => Some(0x578),
        GuestRegister::Rflags => Some(0x570),
        GuestRegister::Cr0 => Some(0x558),
        GuestRegister::Cr3 => Some(0x550),
        GuestRegister::Cr4 => Some(0x548),
        GuestRegister::Dr7 => Some(0x560),
        GuestRegister::Efer => Some(0x4D0),
        GuestRegister::Pat => Some(0x668),
        GuestRegister::FsBase => Some(0x448),
        GuestRegister::GsBase => Some(0x458),
        _ => None,
    }
}

/// The little-endian u64 at `offset` in the VMCB.
fn u64_at(vmcb: &Vmcb, offset: usize) -> u64 {
    // SAFETY: `Vmcb` is 4096 bytes of integers with every reserved range an
    // explicit byte array, so all of its bytes are initialized.
    let bytes = unsafe { core::slice::from_raw_parts((vmcb as *const Vmcb).cast::<u8>(), size_of::<Vmcb>()) };
    u64::from_le_bytes(bytes[offset..offset + 8].try_into().unwrap())
}

/// A distinct, recognizable value per register.
fn value_for(register: GuestRegister) -> u64 {
    0xA0A0_0000 + register.index() as u64
}

#[test]
fn every_register_reads_back_what_was_written() {
    let mut vmcb = Box::new(Vmcb::zeroed());
    let mut gprs = GprFrame::default();
    let mut registers = SvmGuestRegisters::new(&mut vmcb, &mut gprs);
    for register in GuestRegister::ALL {
        registers.write(register, value_for(register)).unwrap();
    }
    // Read only after every write, so two registers sharing a slot would
    // show as the later value overwriting the earlier one.
    for register in GuestRegister::ALL {
        assert_eq!(registers.read(register), Ok(value_for(register)), "{register}");
    }
}

#[test]
fn vmcb_registers_sit_at_the_apm_offsets() {
    for register in GuestRegister::ALL {
        let Some(offset) = expected_vmcb_offset(register) else {
            continue;
        };
        let mut vmcb = Box::new(Vmcb::zeroed());
        let mut gprs = GprFrame::default();
        SvmGuestRegisters::new(&mut vmcb, &mut gprs)
            .write(register, value_for(register))
            .unwrap();
        assert_eq!(u64_at(&vmcb, offset), value_for(register), "{register} at {offset:#x}");
        assert_eq!(gprs, GprFrame::default(), "{register} must not touch the frame");
    }
}

#[test]
fn the_other_general_purpose_registers_live_in_the_frame() {
    let zero = Box::new(Vmcb::zeroed());
    for register in GuestRegister::ALL {
        if expected_vmcb_offset(register).is_some() {
            continue;
        }
        assert!(register.is_general_purpose(), "{register} has no place");
        let mut vmcb = Box::new(Vmcb::zeroed());
        let mut gprs = GprFrame::default();
        SvmGuestRegisters::new(&mut vmcb, &mut gprs)
            .write(register, value_for(register))
            .unwrap();
        assert_ne!(gprs, GprFrame::default(), "{register} must land in the frame");
        assert!(vmcb.save == zero.save, "{register} must not touch the VMCB");
    }
}

#[test]
fn rax_and_rsp_are_the_only_general_purpose_registers_in_the_vmcb() {
    let in_vmcb: Vec<GuestRegister> = GuestRegister::ALL
        .iter()
        .copied()
        .filter(|&register| register.is_general_purpose() && expected_vmcb_offset(register).is_some())
        .collect();
    assert_eq!(in_vmcb, [GuestRegister::Rax, GuestRegister::Rsp]);
}

#[test]
fn read_all_returns_what_the_exit_left() {
    // As #VMEXIT and the run loop would leave them for a guest that ran
    // CPUID: RAX in the VMCB, RBX/RCX/RDX in the frame.
    let mut vmcb = Box::new(Vmcb::zeroed());
    vmcb.save.rax = 1;
    vmcb.save.rip = 0x7ead_4009;
    vmcb.save.cr0 = 0x8001_0033;
    let mut gprs = GprFrame {
        rcx: 0,
        rbx: 0x0002_0800,
        ..GprFrame::default()
    };
    let registers = SvmGuestRegisters::new(&mut vmcb, &mut gprs);
    let snapshot = registers.read_all();
    assert_eq!(snapshot.get(GuestRegister::Rax), Ok(1));
    assert_eq!(snapshot.get(GuestRegister::Rbx), Ok(0x0002_0800));
    assert_eq!(snapshot.get(GuestRegister::Rip), Ok(0x7ead_4009));
    assert_eq!(snapshot.get(GuestRegister::Cr0), Ok(0x8001_0033));
    for (register, value) in snapshot.iter() {
        assert_eq!(value, registers.read(register), "{register}");
    }
}
