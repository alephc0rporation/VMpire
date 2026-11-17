//! Part 1: turning SVM on.
//!
//! The AMD counterpart of entering VMX root operation. Where Intel checks
//! `IA32_FEATURE_CONTROL`, sets CR4.VMXE and runs VMXON, AMD checks the SVM
//! CPUID leaves and the `VM_CR` MSR, sets EFER.SVME, and gives the CPU a host
//! save area through `VM_HSAVE_PA`. There is no VMXON instruction.
//!
//! The capability decode ([`SvmSupport`]) is pure and host-tested; the
//! enabling steps are `unsafe` and run only on a real AMD CPU.

use core::arch::x86_64::__cpuid;

use crate::x86::msr::{IA32_EFER, rdmsr, wrmsr};

/// CPUID 0x80000001: ECX bit 2 is SVM.
pub const CPUID_EXT_FEATURES: u32 = 0x8000_0001;
/// ECX bit 2 of [`CPUID_EXT_FEATURES`]: the CPU has SVM.
pub const CPUID_EXT_ECX_SVM: u32 = 1 << 2;

/// CPUID 0x8000000A: SVM revision and feature flags.
pub const CPUID_SVM_FEATURES: u32 = 0x8000_000A;
/// EDX bit 0 of [`CPUID_SVM_FEATURES`]: nested paging (NPT).
pub const SVM_FEATURE_NP: u32 = 1 << 0;
/// EDX bit 2: SVM lock (the `VM_CR.LOCK` bit is honored).
pub const SVM_FEATURE_SVML: u32 = 1 << 2;
/// EDX bit 3: next-RIP save (`NRIP` in the VMCB control area).
pub const SVM_FEATURE_NRIPS: u32 = 1 << 3;
/// EDX bit 6: flush-by-ASID TLB control.
pub const SVM_FEATURE_FLUSH_BY_ASID: u32 = 1 << 6;

/// `VM_CR`: SVM master control.
pub const VM_CR: u32 = 0xC001_0114;
/// `VM_CR` bit 3 (LOCK): SVMDIS cannot be cleared until reset.
pub const VM_CR_LOCK: u64 = 1 << 3;
/// `VM_CR` bit 4 (SVMDIS): SVM is disabled; EFER.SVME write faults.
pub const VM_CR_SVMDIS: u64 = 1 << 4;

/// `VM_HSAVE_PA`: physical address of the host state save area VMRUN uses.
pub const VM_HSAVE_PA: u32 = 0xC001_0117;

/// EFER bit 12 (SVME): SVM enable. Must be set before any SVM instruction.
pub const EFER_SVME: u64 = 1 << 12;

/// What the CPUID and `VM_CR` reads say about SVM.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SvmSupport {
    /// CPUID 0x80000001:ECX.SVM.
    pub svm: bool,
    /// CPUID 0x8000000A:EDX feature flags.
    pub features: u32,
    /// `VM_CR` (its SVMDIS and LOCK bits are what matter).
    pub vm_cr: u64,
    /// Number of available ASIDs, CPUID 0x8000000A:EBX. 0 means SVM is
    /// unusable even when advertised.
    pub asids: u32,
}

/// Why SVM cannot be used.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SvmUnusable {
    /// CPUID 0x80000001:ECX.SVM is clear.
    NotSupported,
    /// `VM_CR.SVMDIS` is set and `VM_CR.LOCK` holds it: BIOS disabled SVM.
    DisabledLocked,
    /// CPUID 0x8000000A reports no ASIDs.
    NoAsids,
}

impl SvmSupport {
    /// Whether SVM can be turned on, or why not. Mirrors the checks Intel
    /// makes before VMXON (`hal::intel::vmxon`).
    pub fn usable(self) -> Result<(), SvmUnusable> {
        if !self.svm {
            return Err(SvmUnusable::NotSupported);
        }
        // SVMDIS set with LOCK is BIOS-disabled and cannot be cleared. SVML
        // (feature bit 2) says the LOCK bit is honored; without it, SVMDIS is
        // not locking.
        let locked = self.features & SVM_FEATURE_SVML != 0 && self.vm_cr & VM_CR_LOCK != 0;
        if self.vm_cr & VM_CR_SVMDIS != 0 && locked {
            return Err(SvmUnusable::DisabledLocked);
        }
        if self.asids == 0 {
            return Err(SvmUnusable::NoAsids);
        }
        Ok(())
    }

    /// Whether the CPU has nested paging (NPT), the AMD equivalent of EPT.
    pub fn nested_paging(self) -> bool {
        self.features & SVM_FEATURE_NP != 0
    }

    /// Whether VMRUN saves the next RIP into the VMCB, so the exit handler
    /// need not decode the instruction length.
    pub fn next_rip_save(self) -> bool {
        self.features & SVM_FEATURE_NRIPS != 0
    }
}

/// Reads CPUID and `VM_CR` to describe SVM support.
///
/// # Safety
///
/// Must run on an x86-64 CPU (CPUID is always available there). Reading
/// `VM_CR` needs an AMD CPU that has the MSR; on an Intel CPU the `rdmsr`
/// would fault, so call [`has_svm`] first.
pub unsafe fn read_support() -> SvmSupport {
    let ext = __cpuid(CPUID_EXT_FEATURES);
    let svm = ext.ecx & CPUID_EXT_ECX_SVM != 0;
    let (features, asids) = if svm {
        let leaf = __cpuid(CPUID_SVM_FEATURES);
        (leaf.edx, leaf.ebx)
    } else {
        (0, 0)
    };
    let vm_cr = if svm {
        // SAFETY: the SVM CPUID bit guarantees an AMD CPU with VM_CR.
        unsafe { rdmsr(VM_CR) }
    } else {
        0
    };
    SvmSupport {
        svm,
        features,
        vm_cr,
        asids,
    }
}

/// Whether CPUID advertises SVM, without touching any MSR.
pub fn has_svm() -> bool {
    let ext = __cpuid(CPUID_EXT_FEATURES);
    ext.ecx & CPUID_EXT_ECX_SVM != 0
}

/// Sets EFER.SVME and points `VM_HSAVE_PA` at `host_save_area`.
///
/// After this the CPU accepts VMRUN, VMLOAD, VMSAVE, STGI and CLGI.
///
/// # Safety
///
/// [`SvmSupport::usable`] must have returned `Ok`, and `host_save_area` must
/// be the physical address of a zeroed, 4 KiB-aligned page that stays
/// allocated for as long as SVM is on.
pub unsafe fn enable(host_save_area: u64) {
    // SAFETY: the caller guarantees SVM is usable, so setting SVME does not
    // fault; VM_HSAVE_PA exists on any SVM-capable CPU.
    unsafe {
        let efer = rdmsr(IA32_EFER);
        wrmsr(IA32_EFER, efer | EFER_SVME);
        wrmsr(VM_HSAVE_PA, host_save_area);
    }
}

#[cfg(test)]
#[path = "../../../tests/unit/hal/amd/svm.rs"]
mod tests;
