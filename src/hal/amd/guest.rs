//! Guest state setup.
//!
//! The AMD counterpart of `hal::intel::guest_state`. Intel copies the current CPU
//! state into the VMCS guest fields, converting each segment's access rights
//! into VMCS format. AMD fills the VMCB save area ([`super::vmcb::SaveArea`]) and
//! packs each segment's attributes into AMD's 12-bit format. AMD needs no
//! host-state fields: VMRUN saves the host into the host save area itself.
//!
//! The attribute packing and the save-area assembly are pure and host-tested.
//! Reading the live CPU state is the `unsafe` runtime step, done later by the
//! run loop.

use super::vmcb::{SaveArea, Segment};
use crate::x86::segment::SegmentDescriptor;

/// Packs a GDT descriptor's attributes into the VMCB's 12-bit segment
/// attribute format.
///
/// A descriptor's access byte (bits 47:40: Type, S, DPL, P) becomes attrib
/// bits 7:0, and the high attribute nibble (bits 55:52: AVL, L, D/B, G)
/// becomes attrib bits 11:8. The limit bits 51:48 between them are dropped,
/// as on Intel, but Intel leaves a 16-bit field with the high nibble at bits
/// 15:12, whereas AMD packs it down to bits 11:8.
pub const fn segment_attrib(descriptor: u64) -> u16 {
    ((descriptor >> 40) as u16 & 0x00FF) | ((descriptor >> 44) as u16 & 0x0F00)
}

/// Attribute Present bit (bit 7). A segment with it clear is unusable; this
/// is how AMD marks a null segment, in place of Intel's explicit unusable
/// bit.
#[cfg_attr(not(test), allow(dead_code, reason = "used by the unit tests"))]
pub const SEGMENT_PRESENT: u16 = 1 << 7;

/// Whether a packed attribute marks a usable segment.
#[cfg_attr(not(test), allow(dead_code, reason = "used by the unit tests"))]
pub const fn segment_usable(attrib: u16) -> bool {
    attrib & SEGMENT_PRESENT != 0
}

/// Builds a VMCB segment from a GDT descriptor. `base` is passed in because
/// FS, GS and TR take a 64-bit base from elsewhere (MSRs, or a 16-byte system
/// descriptor), not the 32-bit base the descriptor itself holds.
pub fn segment_from_descriptor(selector: u16, descriptor: u64, base: u64) -> Segment {
    Segment {
        selector,
        attrib: segment_attrib(descriptor),
        limit: SegmentDescriptor(descriptor).limit(),
        base,
    }
}

/// The processor state a guest starts from, as plain values. The run loop
/// reads these from the live CPU; keeping them in a struct makes
/// [`into_save_area`](Self::into_save_area) pure and testable.
#[cfg_attr(not(test), allow(dead_code, reason = "used by the unit tests"))]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GuestInit {
    pub cr0: u64,
    pub cr3: u64,
    pub cr4: u64,
    pub efer: u64,
    pub pat: u64,
    pub rflags: u64,
    pub rip: u64,
    pub rsp: u64,
    pub rax: u64,
    pub cs: Segment,
    pub ss: Segment,
    pub ds: Segment,
    pub es: Segment,
    pub fs: Segment,
    pub gs: Segment,
    pub gdtr: Segment,
    pub idtr: Segment,
    pub ldtr: Segment,
    pub tr: Segment,
    pub cpl: u8,
}

impl GuestInit {
    /// Assembles the VMCB save area this guest starts from. The remaining
    /// save-area fields (SYSENTER, STAR/LSTAR, DR6/DR7) stay zero until the
    /// run loop fills them; VMRUN requires EFER.SVME set in the guest EFER,
    /// which the caller includes in `efer`.
    #[cfg_attr(not(test), allow(dead_code, reason = "used by the unit tests"))]
    pub fn into_save_area(self) -> SaveArea {
        let mut save = SaveArea::zeroed();
        save.cr0 = self.cr0;
        save.cr3 = self.cr3;
        save.cr4 = self.cr4;
        save.efer = self.efer;
        save.g_pat = self.pat;
        save.rflags = self.rflags;
        save.rip = self.rip;
        save.rsp = self.rsp;
        save.rax = self.rax;
        save.cs = self.cs;
        save.ss = self.ss;
        save.ds = self.ds;
        save.es = self.es;
        save.fs = self.fs;
        save.gs = self.gs;
        save.gdtr = self.gdtr;
        save.idtr = self.idtr;
        save.ldtr = self.ldtr;
        save.tr = self.tr;
        save.cpl = self.cpl;
        save
    }
}

impl SaveArea {
    /// An all-zero save area.
    pub const fn zeroed() -> Self {
        // SAFETY: every field is an integer, a byte array, or a `Segment` of
        // integers; all-zero is valid for each.
        unsafe { core::mem::zeroed() }
    }
}

#[cfg(test)]
#[path = "../../../tests/unit/hal/amd/guest.rs"]
mod tests;
