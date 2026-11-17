//! `hal::HardwareVirtualization` for SVM: detection (`svm::read_support`),
//! turning SVM on and off (`svm::enable`, EFER.SVME), over the functions of
//! `svm`. The lines it prints are the ones the AMD bring-up printed.

use super::svm::{
    self, CPUID_EXT_ECX_SVM, CPUID_EXT_FEATURES, EFER_SVME, SVM_FEATURE_FLUSH_BY_ASID, SvmSupport, SvmUnusable,
};
use crate::hal::types::{Capabilities, CpuidBit, CpuidRegister, CrConstraints, OptionalFeatures};
use crate::x86::cr;
use crate::x86::msr::{self, IA32_EFER};
use crate::x86::page::allocate_zeroed_page;

/// SVM turned on: EFER.SVME set, with VM_HSAVE_PA at the host save area.
pub struct HardwareVirtualization {
    original_cr0: u64,
    original_cr4: u64,
}

/// Why turning SVM on failed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VirtualizationError {
    /// UEFI could not provide the VM_HSAVE_PA page.
    AllocationFailed,
}

/// Whether the CPU can flush one ASID's translations
/// (CPUID 0x8000000A:EDX bit 6).
pub(super) fn flush_by_asid(support: SvmSupport) -> bool {
    support.features & SVM_FEATURE_FLUSH_BY_ASID != 0
}

impl crate::hal::HardwareVirtualization for HardwareVirtualization {
    type ControlCapabilities = SvmSupport;
    type Error = VirtualizationError;

    const NAME: &'static str = "SVM";
    /// CPUID.80000001H:ECX bit 2: SVM. A guest that saw it would try VMRUN,
    /// which always exits, and Aleph0 does not nest SVM.
    const CPUID_FEATURE: CpuidBit = CpuidBit {
        leaf: CPUID_EXT_FEATURES,
        register: CpuidRegister::Ecx,
        mask: CPUID_EXT_ECX_SVM,
    };
    /// None: SVM's enable bit is EFER.SVME, not a CR4 bit.
    const GUEST_FORBIDDEN_CR4: u64 = 0;
    const ENABLE_BIT: &'static str = "EFER.SVME";

    fn detect() -> Capabilities {
        if !svm::has_svm() {
            return Capabilities::NotSupportedByCpu;
        }
        // SAFETY: CPUID advertises SVM, so this is an AMD CPU with VM_CR.
        match unsafe { svm::read_support() }.usable() {
            Ok(()) => Capabilities::Supported,
            Err(SvmUnusable::DisabledLocked) => Capabilities::DisabledByFirmware,
            Err(SvmUnusable::NotSupported | SvmUnusable::NoAsids) => Capabilities::NotSupportedByCpu,
        }
    }

    unsafe fn control_capabilities() -> SvmSupport {
        // SAFETY: the caller confirmed SVM support, so VM_CR exists.
        unsafe { svm::read_support() }
    }

    fn report_control_capabilities(support: &SvmSupport) {
        log!(
            "SVM usable: features={:#x} ASIDs={} nested paging={} next-RIP save={}{}.",
            support.features,
            support.asids,
            support.nested_paging(),
            support.next_rip_save(),
            if flush_by_asid(*support) { ", flush by ASID" } else { "" }
        );
    }

    /// Every one: SVM intercepts none of these instructions unless asked,
    /// so the guest runs them as the CPU has them, and CPUID shows them as
    /// the CPU does.
    fn optional_features(_support: &SvmSupport) -> OptionalFeatures {
        OptionalFeatures {
            rdtscp: true,
            invpcid: true,
            xsaves: true,
            waitpkg: true,
        }
    }

    fn report_guest_features(_features: OptionalFeatures, xcr0: u64, host_cr4: u64) {
        log!(
            "Guest features: XCR0 supported {xcr0:#x} (XSETBV {}); host CR4 {host_cr4:#x}.",
            if xcr0 != 0 { "run in the host" } else { "gets #GP" },
        );
    }

    /// None fixed: SVM constrains neither CR0 nor CR4 while it is on.
    unsafe fn cr_constraints() -> CrConstraints {
        CrConstraints {
            cr0_must_be_one: 0,
            cr0_may_be_one: u64::MAX,
            cr4_must_be_one: 0,
            cr4_may_be_one: u64::MAX,
        }
    }

    unsafe fn enable() -> Result<Self, VirtualizationError> {
        let original_cr0 = cr::read_cr0();
        let original_cr4 = cr::read_cr4();
        let hsave = allocate_zeroed_page(VirtualizationError::AllocationFailed)?.as_ptr() as u64;
        // SAFETY: the caller confirmed SVM is usable; `hsave` is a fresh
        // zeroed page, never freed.
        unsafe { svm::enable(hsave) };
        log!("SVM on: EFER.SVME set, VM_HSAVE_PA={hsave:#018x}.");
        Ok(Self {
            original_cr0,
            original_cr4,
        })
    }

    /// CR0 and CR4 as they were before (the run switched CR4 to the
    /// host's), as Intel's VMXOFF path puts them back, then EFER without
    /// SVME.
    unsafe fn disable(self) -> Result<(), VirtualizationError> {
        // SAFETY: the caller guarantees SVM is on with no guest running;
        // `original_cr0` and `original_cr4` were the live values before
        // `enable`, and EFER exists on every 64-bit CPU.
        unsafe {
            cr::write_cr0(self.original_cr0);
            cr::write_cr4(self.original_cr4);
            let efer = msr::rdmsr(IA32_EFER);
            msr::wrmsr(IA32_EFER, efer & !EFER_SVME);
        }
        Ok(())
    }

    fn original_cr0(&self) -> u64 {
        self.original_cr0
    }

    fn original_cr4(&self) -> u64 {
        self.original_cr4
    }

    fn enabled() -> bool {
        // SAFETY: EFER exists on every 64-bit CPU; reading it has no side
        // effect.
        (unsafe { msr::rdmsr(IA32_EFER) } & EFER_SVME) != 0
    }

    fn report_error(error: VirtualizationError) {
        log!("Turning SVM on failed: {error:?}");
    }
}
