//! The VMCB control area and which events exit.
//!
//! The AMD counterpart of the VMCS execution controls plus the MSR and I/O
//! bitmaps. Intel chooses control bits the capability MSRs allow and writes
//! them with VMWRITE; AMD writes plain intercept bit vectors into the VMCB
//! control area (offsets 0x000-0x3FF) and points two fields at an MSR
//! permission map (MSRPM) and an I/O permission map (IOPM).
//!
//! The control-area layout follows AMD64 APM Vol. 2, Appendix B, Table B-1.
//! The bitmap index math is pure and host-tested, like Intel's
//! `exit_bitmaps::msr_bit` and `io_bit`.

/// Intercept word 3 (offset 0x00C). A set bit makes that event #VMEXIT.
#[allow(dead_code, reason = "the APM's intercept bits, listed whole")]
pub mod intercept3 {
    pub const INTR: u32 = 1 << 0;
    pub const NMI: u32 = 1 << 1;
    pub const SMI: u32 = 1 << 2;
    pub const INIT: u32 = 1 << 3;
    pub const CPUID: u32 = 1 << 18;
    pub const HLT: u32 = 1 << 24;
    pub const IOIO_PROT: u32 = 1 << 27;
    pub const MSR_PROT: u32 = 1 << 28;
    pub const SHUTDOWN: u32 = 1 << 31;
}

/// Intercept word 4 (offset 0x010).
#[allow(dead_code, reason = "the APM's intercept bits, listed whole")]
pub mod intercept4 {
    pub const VMRUN: u32 = 1 << 0;
    pub const VMMCALL: u32 = 1 << 1;
    pub const VMLOAD: u32 = 1 << 2;
    pub const VMSAVE: u32 = 1 << 3;
    pub const STGI: u32 = 1 << 4;
    pub const CLGI: u32 = 1 << 5;
    pub const SKINIT: u32 = 1 << 6;
    pub const XSETBV: u32 = 1 << 13;
}

/// `nested_ctl` bit 0 (offset 0x090): enable nested paging (NPT).
pub const NESTED_CTL_NP_ENABLE: u64 = 1 << 0;

/// `tlb_control` values (offset 0x05C).
pub mod tlb_control {
    /// Do not flush.
    pub const NONE: u32 = 0;
    /// Flush the whole TLB on the next VMRUN.
    pub const FLUSH_ALL: u32 = 1;
    /// Flush this guest's ASID (needs the flush-by-ASID feature).
    pub const FLUSH_ASID: u32 = 7;
}

/// The VMCB control area (offsets 0x000-0x3FF), with the fields bring-up and
/// the run loop set. Reserved ranges are explicit padding so every field
/// lands at its APM offset.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ControlArea {
    pub intercept_cr: u32,
    pub intercept_dr: u32,
    pub intercept_exceptions: u32,
    pub intercept_word3: u32,
    pub intercept_word4: u32,
    pub intercept_word5: u32,
    _reserved_018: [u8; 0x03C - 0x018],
    pub pause_filter_thresh: u16,
    pub pause_filter_count: u16,
    pub iopm_base_pa: u64,
    pub msrpm_base_pa: u64,
    pub tsc_offset: u64,
    pub asid: u32,
    pub tlb_control: u32,
    pub int_ctl: u32,
    pub int_vector: u32,
    pub int_state: u32,
    _reserved_06c: u32,
    pub exit_code: u32,
    pub exit_code_hi: u32,
    pub exit_info_1: u64,
    pub exit_info_2: u64,
    pub exit_int_info: u32,
    pub exit_int_info_err: u32,
    pub nested_ctl: u64,
    _reserved_098: [u8; 0x0A8 - 0x098],
    pub event_inj: u64,
    pub nested_cr3: u64,
    _reserved_0b8: [u8; 0x0C8 - 0x0B8],
    /// The guest RIP of the instruction after the one that exited, when the
    /// CPU has the NRIPS feature (`svm::SvmSupport::next_rip_save`). 0 when it
    /// does not, and the handler decodes the length instead.
    pub next_rip: u64,
    _reserved_0d0: [u8; 0x400 - 0x0D0],
}

impl ControlArea {
    /// An all-zero control area: no intercepts, no nested paging.
    #[cfg_attr(not(test), allow(dead_code, reason = "used by the unit tests"))]
    pub const fn zeroed() -> Self {
        // SAFETY: every field is an integer or a byte array, for which
        // all-zero is a valid value.
        unsafe { core::mem::zeroed() }
    }
}

// Tie the layout to the APM's offsets (Table B-1), so a wrong padding size
// breaks the build.
const _: () = {
    use core::mem::offset_of;
    assert!(offset_of!(ControlArea, intercept_word3) == 0x00C);
    assert!(offset_of!(ControlArea, intercept_word4) == 0x010);
    assert!(offset_of!(ControlArea, iopm_base_pa) == 0x040);
    assert!(offset_of!(ControlArea, msrpm_base_pa) == 0x048);
    assert!(offset_of!(ControlArea, asid) == 0x058);
    assert!(offset_of!(ControlArea, tlb_control) == 0x05C);
    assert!(offset_of!(ControlArea, exit_code) == 0x070);
    assert!(offset_of!(ControlArea, exit_info_1) == 0x078);
    assert!(offset_of!(ControlArea, exit_info_2) == 0x080);
    assert!(offset_of!(ControlArea, exit_int_info) == 0x088);
    assert!(offset_of!(ControlArea, nested_ctl) == 0x090);
    assert!(offset_of!(ControlArea, event_inj) == 0x0A8);
    assert!(offset_of!(ControlArea, nested_cr3) == 0x0B0);
    assert!(offset_of!(ControlArea, next_rip) == 0x0C8);
    assert!(size_of::<ControlArea>() == 0x400);
};

/// The MSRPM covers three MSR ranges, two bits per MSR (read, write). An MSR
/// outside all three cannot be intercepted and always exits.
pub const MSRPM_RANGES: [core::ops::Range<u32>; 3] = [
    0x0000_0000..0x0000_2000,
    0xC000_0000..0xC000_2000,
    0xC001_0000..0xC001_2000,
];

/// The MSRPM is 8 KiB.
pub const MSRPM_SIZE: usize = 0x2000;
/// The IOPM is 12 KiB plus one bit; 3 pages are allocated.
pub const IOPM_SIZE: usize = 0x3000;

/// Which bit a port holds. One bit per port, in order from port 0.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MsrAccess {
    Read,
    Write,
}

/// The (byte, bit) of `msr`'s intercept pair in the MSRPM, or `None` when it
/// lies outside the three covered ranges. `access` picks the read or write
/// bit of the pair.
pub const fn msrpm_bit(msr: u32, access: MsrAccess) -> Option<(usize, u8)> {
    // Each range is 0x2000 MSRs; two bits per MSR, so 4 MSRs per byte.
    let mut base_bits = 0usize;
    let mut range = 0;
    while range < MSRPM_RANGES.len() {
        let start = MSRPM_RANGES[range].start;
        let end = MSRPM_RANGES[range].end;
        if msr >= start && msr < end {
            let index = (msr - start) as usize;
            let bit_in_range = index * 2
                + match access {
                    MsrAccess::Read => 0,
                    MsrAccess::Write => 1,
                };
            let total = base_bits + bit_in_range;
            return Some((total / 8, (total % 8) as u8));
        }
        base_bits += (end - start) as usize * 2;
        range += 1;
    }
    None
}

/// The (byte, bit) of `port`'s intercept bit in the IOPM. Every port fits,
/// so this never fails.
pub const fn iopm_bit(port: u16) -> (usize, u8) {
    let index = port as usize;
    (index / 8, (index % 8) as u8)
}

#[cfg(test)]
#[path = "../../../tests/unit/hal/amd/control.rs"]
mod tests;
