//! Unit tests for `src/hal/amd/guest.rs`.
//!
//! Compiled only by `cargo test-host`. These check the segment-attribute
//! packing and the save-area assembly against the APM.

use super::*;

/// A flat 64-bit code segment descriptor: base 0, limit 0xFFFFF, Type 0xB
/// (execute/read, accessed), S=1, DPL=0, P=1, L=1, G=1, D=0.
/// Access byte (bits 47:40) = 0x9B, high nibble (bits 55:52) = 0b1010 (L, G).
const FLAT_CODE: u64 = 0x00AF_9B00_0000_FFFF;

#[test]
fn segment_attrib_packs_the_access_byte_and_the_high_nibble() {
    // 0x9B access byte in bits 7:0, 0xA high nibble in bits 11:8.
    assert_eq!(segment_attrib(FLAT_CODE), 0x0A9B);
    // The limit nibble (bits 51:48 of the descriptor) must not leak in.
    let with_limit_nibble = FLAT_CODE | 0x000F_0000_0000_0000;
    assert_eq!(segment_attrib(with_limit_nibble), 0x0A9B);
}

#[test]
fn the_present_bit_decides_usability() {
    assert!(segment_usable(segment_attrib(FLAT_CODE)));
    // Clear P (bit 7): the segment is unusable.
    let not_present = FLAT_CODE & !(1 << 47);
    assert!(!segment_usable(segment_attrib(not_present)));
}

#[test]
fn a_segment_from_a_descriptor_keeps_the_passed_base() {
    let segment = segment_from_descriptor(0x0008, FLAT_CODE, 0xDEAD_0000);
    assert_eq!(segment.selector, 0x0008);
    assert_eq!(segment.attrib, 0x0A9B);
    assert_eq!(segment.limit, 0xFFFF_FFFF);
    // The 64-bit base is the one passed in, not the descriptor's 32-bit base.
    assert_eq!(segment.base, 0xDEAD_0000);
}

/// A minimal guest init with recognizable values.
fn sample_init() -> GuestInit {
    let code = segment_from_descriptor(0x0008, FLAT_CODE, 0);
    let data = segment_from_descriptor(0x0010, FLAT_CODE & !(1 << 43), 0);
    GuestInit {
        cr0: 0x8001_0033,
        cr3: 0x0440_1000,
        cr4: 0x2668,
        efer: 0xD00 | (1 << 12),
        pat: 0x0007_0406_0007_0406,
        rflags: 0x2,
        rip: 0x7EAD_4009,
        rsp: 0x7EAD_4000,
        rax: 0xA1E1,
        cs: code,
        ss: data,
        ds: data,
        es: data,
        fs: data,
        gs: data,
        gdtr: Segment {
            selector: 0,
            attrib: 0,
            limit: 0x27,
            base: 0x1000,
        },
        idtr: Segment {
            selector: 0,
            attrib: 0,
            limit: 0xFFF,
            base: 0x2000,
        },
        ldtr: Segment::default(),
        tr: segment_from_descriptor(0x0040, 0x0000_8B00_0000_0067, 0x3000),
        cpl: 0,
    }
}

#[test]
fn into_save_area_places_the_control_registers_and_rip() {
    let save = sample_init().into_save_area();
    assert_eq!(save.cr0, 0x8001_0033);
    assert_eq!(save.cr3, 0x0440_1000);
    assert_eq!(save.cr4, 0x2668);
    assert_eq!(save.efer, 0xD00 | (1 << 12));
    assert_eq!(save.g_pat, 0x0007_0406_0007_0406);
    assert_eq!(save.rip, 0x7EAD_4009);
    assert_eq!(save.rsp, 0x7EAD_4000);
    assert_eq!(save.rax, 0xA1E1);
    assert_eq!(save.rflags, 0x2);
    assert_eq!(save.cpl, 0);
}

#[test]
fn into_save_area_copies_every_segment() {
    let init = sample_init();
    let save = init.into_save_area();
    assert_eq!(save.cs, init.cs);
    assert_eq!(save.ss, init.ss);
    assert_eq!(save.tr, init.tr);
    assert_eq!(save.gdtr.base, 0x1000);
    assert_eq!(save.idtr.limit, 0xFFF);
    // Fields no guest-init value fills stay zero.
    assert_eq!(save.dr7, 0);
    assert_eq!(save.sysenter_cs, 0);
}

#[test]
fn the_guest_efer_keeps_svme_set() {
    // VMRUN rejects a guest EFER without SVME (bit 12).
    let save = sample_init().into_save_area();
    assert_ne!(save.efer & (1 << 12), 0);
}
