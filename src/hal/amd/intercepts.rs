//! `hal::Intercepts` for SVM: the MSR and I/O permission maps (`exit_maps`).

use super::control::{MsrAccess, msrpm_bit};
use super::exit_maps::{ExitMaps, ExitMapsError};
use crate::hal::types::GuestKind;
use crate::hv::intercepted::intercepted;
use crate::x86::mtrr::{self, MtrrCapabilities};

/// The MSRPM and the IOPM.
pub struct Intercepts(ExitMaps);

impl Intercepts {
    /// The maps, for the VMCB.
    pub(super) fn maps(&self) -> &ExitMaps {
        &self.0
    }
}

impl crate::hal::Intercepts for Intercepts {
    type Error = ExitMapsError;

    /// Every MTRR address, and the policy lists for `kind`
    /// (`hv::intercepted`).
    fn allocate(kind: GuestKind) -> Result<Self, ExitMapsError> {
        let (msrs, ports) = intercepted(kind);
        ExitMaps::allocate(msrs, ports).map(Self)
    }

    /// Whether the MSR lies in one of the MSRPM's three ranges.
    fn covers_msr(msr: u32) -> bool {
        msrpm_bit(msr, MsrAccess::Read).is_some()
    }

    /// Where the maps are and what they intercept, in the lines Intel's
    /// bitmaps print. On AMD the passed-through IA32_KERNEL_GS_BASE lives in
    /// the VMCB (VMSAVE and VMLOAD switch it), where the exit handler reads
    /// and writes the guest's value.
    fn report(&self, kind: GuestKind, mtrr: MtrrCapabilities) {
        let maps = &self.0;
        log!(
            "Permission maps allocated: MSRPM={:#018x} IOPM={:#018x}.",
            maps.msrpm_address(),
            maps.iopm_address(),
        );
        let (msrs, ports) = intercepted(kind);
        log!("Intercepted MSRs (read and write) for the {kind:?} guest: {msrs:x?}; intercepted ports: {ports:x?}.");
        let addresses = mtrr::every_msr_address().count();
        match mtrr.msrs().count() {
            0 => log!(
                "Intercepted MTRR addresses (read and write): all {addresses}, which get #GP: the CPU has no MTRRs."
            ),
            count => log!(
                "Intercepted MTRR addresses (read and write): all {addresses}. The guest reads and writes a copy of the {count} MSRs the CPU has (IA32_MTRR_DEF_TYPE, {} fixed-range MSRs and {} variable ranges); the other {} get #GP.",
                if mtrr.fixed() { mtrr::FIXED_MSRS } else { 0 },
                mtrr.variable_count(),
                addresses - count,
            ),
        }
    }

    fn report_error(error: ExitMapsError) {
        log!("Permission map allocation failed: {error:?}");
    }
}
