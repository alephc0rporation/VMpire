//! What Aleph0 offers the guest, chosen once at bring-up ([`GuestFeatures`]).
//! The AMD counterpart of `hv::features`.
//!
//! Intel's version has two halves:
//!
//! - optional secondary controls (RDTSCP, INVPCID, XSAVES/XRSTORS, user wait
//!   and pause). In VMX non-root operation those instructions raise #UD
//!   unless their control is set, so a control the CPU does not allow means
//!   hiding its CPUID bits;
//! - XSETBV, which Aleph0 runs in root operation, needing CR4.OSXSAVE in the
//!   host's CR4.
//!
//! SVM has no counterpart of the first half: there are no enable controls. A
//! guest under SVM runs RDTSCP, INVPCID, XSAVES and the rest natively unless
//! the VMCB intercepts them (AMD64 APM Vol. 2, 15.9-15.13), so nothing #UDs
//! that would not #UD on bare metal, and no CPUID bit has to be hidden for it.
//!
//! The second half carries over when XSETBV is intercepted
//! (`control::intercept4::XSETBV`; unlike Intel, it exits only then). Aleph0
//! then runs XSETBV in the host, which needs CR4.OSXSAVE there. AMD has no
//! IA32_VMX_CR4_FIXED1: the host may set CR4.OSXSAVE whenever the CPU has
//! XSAVE. Without XSAVE, XSAVE is hidden from the guest, as Intel does.
//!
//! What is not offered is hidden from the guest's CPUID
//! ([`GuestFeatures::hidden`], applied by `exit::cpuid_policy`). The exit
//! handler reads the choice back with [`published`].

use core::arch::x86_64::{__cpuid, __cpuid_count};
use core::sync::atomic::{AtomicU64, Ordering};

use crate::hv::features::CPUID_1_ECX_XSAVE;
use crate::x86::cpuid::XSAVE_LEAF;
use crate::x86::cr::CR4_OSXSAVE;

/// CPUID bits, by register, the guest must not see. Same registers as
/// Intel's `HiddenCpuid`, so `cpuid_policy` applies them the same way.
///
/// On AMD only `leaf_1_ecx` is ever set (XSAVE, when the CPU has none). The
/// other four are always 0: on Intel they hold the bits of the optional
/// secondary controls (INVPCID, RDPID, WAITPKG, XSAVES, RDTSCP), whose
/// instructions #UD in the guest unless the control is on. SVM has no such
/// controls; the guest runs those instructions natively, so there is nothing
/// to hide for them.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HiddenCpuid {
    /// CPUID.1:ECX. XSAVE (bit 26) when the CPU has no XSAVE.
    pub leaf_1_ecx: u32,
    /// CPUID.(EAX=7,ECX=0):EBX. Always 0 on AMD (INVPCID on Intel).
    pub leaf_7_ebx: u32,
    /// CPUID.(EAX=7,ECX=0):ECX. Always 0 on AMD (RDPID, WAITPKG on Intel).
    pub leaf_7_ecx: u32,
    /// CPUID.(EAX=0xD,ECX=1):EAX. Always 0 on AMD (XSAVES on Intel).
    pub leaf_d_1_eax: u32,
    /// CPUID.80000001H:EDX. Always 0 on AMD (RDTSCP on Intel).
    pub extended_edx: u32,
}

impl HiddenCpuid {
    /// Nothing hidden.
    pub const NONE: Self = Self {
        leaf_1_ecx: 0,
        leaf_7_ebx: 0,
        leaf_7_ecx: 0,
        leaf_d_1_eax: 0,
        extended_edx: 0,
    };
}

/// What Aleph0 offers the guest, chosen at bring-up.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GuestFeatures {
    /// The XCR0 bits the guest's XSETBV may set: CPUID.(EAX=0xD,ECX=0):EDX:EAX,
    /// or 0 when the CPU has no XSAVE, so that every XSETBV gets #GP. Bit 0
    /// (x87) is always reported, so 0 means only that.
    pub xcr0: u64,
}

impl GuestFeatures {
    /// No XSETBV: what the exit handler reads before bring-up has published.
    pub const NONE: Self = Self { xcr0: 0 };

    /// The features for a CPU with CPUID.1:ECX `cpuid_1_ecx` and XCR0 bits
    /// `xcr0_supported` (CPUID.(0xD,0), meaningful only with XSAVE). XSETBV
    /// runs in the host only with XSAVE.
    pub fn new(cpuid_1_ecx: u32, xcr0_supported: u64) -> Self {
        let xsetbv = cpuid_1_ecx & CPUID_1_ECX_XSAVE != 0;
        Self {
            xcr0: if xsetbv { xcr0_supported } else { 0 },
        }
    }

    /// Reads what [`GuestFeatures::new`] needs from the CPU.
    pub fn read() -> Self {
        let cpuid_1_ecx = __cpuid(1).ecx;
        // Leaf 0xD exists whenever XSAVE does.
        let xcr0_supported = if cpuid_1_ecx & CPUID_1_ECX_XSAVE != 0 {
            let leaf = __cpuid_count(XSAVE_LEAF, 0);
            (u64::from(leaf.edx) << 32) | u64::from(leaf.eax)
        } else {
            0
        };
        Self::new(cpuid_1_ecx, xcr0_supported)
    }

    /// Whether Aleph0 runs the guest's XSETBV.
    pub fn xsetbv(self) -> bool {
        self.xcr0 != 0
    }

    /// The host CR4 for the CR4 bring-up runs with (`cr4`): CR4.OSXSAVE added
    /// when Aleph0 runs XSETBV, which raises #UD without it.
    pub fn host_cr4(self, cr4: u64) -> u64 {
        if self.xsetbv() { cr4 | CR4_OSXSAVE } else { cr4 }
    }

    /// The CPUID bits the guest must not see: XSAVE when Aleph0 cannot run
    /// XSETBV. There are no optional controls to add (see the module docs).
    pub fn hidden(self) -> HiddenCpuid {
        if self.xsetbv() {
            HiddenCpuid::NONE
        } else {
            HiddenCpuid {
                leaf_1_ecx: CPUID_1_ECX_XSAVE,
                ..HiddenCpuid::NONE
            }
        }
    }
}

// What `publish` recorded, for the exit handler.
static XCR0: AtomicU64 = AtomicU64::new(GuestFeatures::NONE.xcr0);

/// Records, before the first VMRUN, what the guest is offered.
pub fn publish(features: GuestFeatures) {
    XCR0.store(features.xcr0, Ordering::Relaxed);
}

/// What [`publish`] recorded.
pub fn published() -> GuestFeatures {
    GuestFeatures {
        xcr0: XCR0.load(Ordering::Relaxed),
    }
}

#[cfg(test)]
#[path = "../../../tests/unit/hal/amd/features.rs"]
mod tests;
