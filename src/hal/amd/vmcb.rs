//! The VMCB (virtual machine control block): the 4 KiB page that describes
//! an SVM guest (AMD64 APM Vol. 2, Appendix B).
//!
//! - Offsets `0x000..0x400`: the control area (intercepts, exit code and
//!   exit information, nested paging). Kept as raw bytes until the run loop
//!   needs its fields.
//! - Offset `0x400`: the state save area ([`SaveArea`]), the guest's
//!   registers. VMRUN loads the guest from it, and every #VMEXIT writes the
//!   guest's values back into it.
//!
//! VMRUN and #VMEXIT switch only part of the save area. FS, GS, TR, LDTR,
//! KernelGsBase, STAR, LSTAR, CSTAR, SFMASK and the SYSENTER MSRs are moved
//! by VMLOAD and VMSAVE instead, so the FS and GS bases read here are the
//! guest's only if the run loop executes VMSAVE on the guest's VMCB after
//! each exit (and VMLOAD before each VMRUN).

/// Where the state save area starts in the VMCB.
pub const SAVE_AREA_OFFSET: usize = 0x400;

/// A segment register in the save area: selector, attributes, limit, base.
#[repr(C)]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Segment {
    pub selector: u16,
    /// The descriptor's attribute bits, packed (APM Vol. 2, 15.5.1).
    pub attrib: u16,
    pub limit: u32,
    pub base: u64,
}

/// The state save area, up to G_PAT: the fields the hypervisor reads and
/// writes as the guest's registers, at the APM's offsets. Reserved ranges
/// are kept as padding so every field lands where the CPU expects it.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SaveArea {
    pub es: Segment,
    pub cs: Segment,
    pub ss: Segment,
    pub ds: Segment,
    pub fs: Segment,
    pub gs: Segment,
    pub gdtr: Segment,
    pub ldtr: Segment,
    pub idtr: Segment,
    pub tr: Segment,
    _reserved_0a0: [u8; 0x2A],
    pub vmpl: u8,
    pub cpl: u8,
    _reserved_0cc: [u8; 4],
    pub efer: u64,
    _reserved_0d8: [u8; 0x70],
    pub cr4: u64,
    pub cr3: u64,
    pub cr0: u64,
    pub dr7: u64,
    pub dr6: u64,
    pub rflags: u64,
    pub rip: u64,
    _reserved_180: [u8; 0x58],
    pub rsp: u64,
    pub s_cet: u64,
    pub ssp: u64,
    pub isst_addr: u64,
    pub rax: u64,
    pub star: u64,
    pub lstar: u64,
    pub cstar: u64,
    pub sfmask: u64,
    pub kernel_gs_base: u64,
    pub sysenter_cs: u64,
    pub sysenter_esp: u64,
    pub sysenter_eip: u64,
    pub cr2: u64,
    _reserved_248: [u8; 0x20],
    pub g_pat: u64,
}

/// A whole VMCB page. VMRUN takes its physical address in RAX and needs it
/// 4 KiB aligned.
#[repr(C, align(4096))]
#[derive(Clone, Copy)]
pub struct Vmcb {
    pub control: super::control::ControlArea,
    pub save: SaveArea,
    _rest: [u8; 4096 - SAVE_AREA_OFFSET - size_of::<SaveArea>()],
}

impl Vmcb {
    /// An all-zero VMCB.
    #[cfg_attr(not(test), allow(dead_code, reason = "used by the unit tests"))]
    pub const fn zeroed() -> Self {
        // SAFETY: every field is an integer or an array of integers, for
        // which all-zero bytes are a valid value.
        unsafe { core::mem::zeroed() }
    }
}

// Tie the layout to the APM's offsets (APM Vol. 2, Table B-2, offsets in the
// save area), so a wrong padding size breaks the build instead of putting a
// guest register in the wrong place.
const _: () = {
    use core::mem::offset_of;
    assert!(size_of::<Segment>() == 16);
    assert!(offset_of!(Segment, base) == 8);
    assert!(offset_of!(SaveArea, fs) == 0x040);
    assert!(offset_of!(SaveArea, gs) == 0x050);
    assert!(offset_of!(SaveArea, tr) == 0x090);
    assert!(offset_of!(SaveArea, cpl) == 0x0CB);
    assert!(offset_of!(SaveArea, efer) == 0x0D0);
    assert!(offset_of!(SaveArea, cr4) == 0x148);
    assert!(offset_of!(SaveArea, cr3) == 0x150);
    assert!(offset_of!(SaveArea, cr0) == 0x158);
    assert!(offset_of!(SaveArea, dr7) == 0x160);
    assert!(offset_of!(SaveArea, rflags) == 0x170);
    assert!(offset_of!(SaveArea, rip) == 0x178);
    assert!(offset_of!(SaveArea, rsp) == 0x1D8);
    assert!(offset_of!(SaveArea, rax) == 0x1F8);
    assert!(offset_of!(SaveArea, kernel_gs_base) == 0x220);
    assert!(offset_of!(SaveArea, cr2) == 0x240);
    assert!(offset_of!(SaveArea, g_pat) == 0x268);
    assert!(size_of::<Vmcb>() == 4096);
    assert!(align_of::<Vmcb>() == 4096);
    assert!(offset_of!(Vmcb, save) == SAVE_AREA_OFFSET);
};

#[cfg(test)]
#[path = "../../../tests/unit/hal/amd/vmcb.rs"]
mod tests;
