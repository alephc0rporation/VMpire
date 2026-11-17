//! Unit tests for `src/hal/amd/vmcb.rs`.
//!
//! Compiled only by `cargo test-host`. That file includes this one with
//! `#[path]` as its child module `tests`, so `use super::*` reaches its
//! private items.
//!
//! The field offsets are checked at compile time in that file; these tests
//! check what the CPU sees at those offsets in memory.

use super::*;

/// The VMCB's bytes, as the CPU reads them.
fn bytes(vmcb: &Vmcb) -> &[u8] {
    // SAFETY: `Vmcb` is 4096 bytes of integers with no padding the compiler
    // adds (every reserved range is an explicit byte array), so all of its
    // bytes are initialized.
    unsafe { core::slice::from_raw_parts((vmcb as *const Vmcb).cast::<u8>(), size_of::<Vmcb>()) }
}

/// The little-endian u64 at `offset` in the VMCB.
fn u64_at(vmcb: &Vmcb, offset: usize) -> u64 {
    u64::from_le_bytes(bytes(vmcb)[offset..offset + 8].try_into().unwrap())
}

#[test]
fn a_zeroed_vmcb_is_one_zero_page() {
    let vmcb = Box::new(Vmcb::zeroed());
    assert_eq!(bytes(&vmcb).len(), 4096);
    assert!(bytes(&vmcb).iter().all(|&byte| byte == 0));
    assert_eq!(
        (&*vmcb as *const Vmcb as usize) % 4096,
        0,
        "VMRUN needs the VMCB page-aligned"
    );
}

#[test]
fn save_area_fields_sit_at_their_vmcb_offsets() {
    let mut vmcb = Box::new(Vmcb::zeroed());
    // Offsets from the start of the VMCB: 0x400 plus the save-area offset
    // in APM Vol. 2, Table B-2.
    vmcb.save.efer = 0x0d01;
    vmcb.save.cr4 = 0xc4;
    vmcb.save.cr3 = 0xc3;
    vmcb.save.cr0 = 0xc0;
    vmcb.save.dr7 = 0xd7;
    vmcb.save.rflags = 0xf1;
    vmcb.save.rip = 0x1b;
    vmcb.save.rsp = 0x5b;
    vmcb.save.rax = 0xa0;
    vmcb.save.g_pat = 0x9a7;
    vmcb.save.fs.base = 0xf5;
    vmcb.save.gs.base = 0x65;
    assert_eq!(u64_at(&vmcb, 0x4D0), 0x0d01, "EFER");
    assert_eq!(u64_at(&vmcb, 0x548), 0xc4, "CR4");
    assert_eq!(u64_at(&vmcb, 0x550), 0xc3, "CR3");
    assert_eq!(u64_at(&vmcb, 0x558), 0xc0, "CR0");
    assert_eq!(u64_at(&vmcb, 0x560), 0xd7, "DR7");
    assert_eq!(u64_at(&vmcb, 0x570), 0xf1, "RFLAGS");
    assert_eq!(u64_at(&vmcb, 0x578), 0x1b, "RIP");
    assert_eq!(u64_at(&vmcb, 0x5D8), 0x5b, "RSP");
    assert_eq!(u64_at(&vmcb, 0x5F8), 0xa0, "RAX");
    assert_eq!(u64_at(&vmcb, 0x668), 0x9a7, "G_PAT");
    assert_eq!(u64_at(&vmcb, 0x448), 0xf5, "FS base");
    assert_eq!(u64_at(&vmcb, 0x458), 0x65, "GS base");
}

#[test]
fn the_control_area_is_left_alone_by_save_area_writes() {
    let mut vmcb = Box::new(Vmcb::zeroed());
    vmcb.save = SaveArea {
        rip: u64::MAX,
        ..vmcb.save
    };
    vmcb.save.es.selector = u16::MAX;
    assert!(bytes(&vmcb)[..SAVE_AREA_OFFSET].iter().all(|&byte| byte == 0));
}
