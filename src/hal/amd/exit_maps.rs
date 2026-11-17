//! Part 7 (with part 2's index math): the MSR and I/O permission maps,
//! which guest RDMSR/WRMSR and IN/OUT exit to Aleph0. The AMD counterpart
//! of `hal::intel::exit_bitmaps`.
//!
//! With `intercept3::MSR_PROT` set, RDMSR/WRMSR exit only for MSRs whose bit
//! is set in the MSRPM, and with `intercept3::IOIO_PROT` IN/OUT exit only
//! for ports whose bit is set in the IOPM. Everything else reaches the
//! hardware directly.
//!
//! ```text
//! MSRPM, 8 KiB (AMD64 APM Vol. 2, 15.11): two bits per MSR, read then write
//! offset 0x0000  MSR 0x00000000..=0x00001FFF
//! offset 0x0800  MSR 0xC0000000..=0xC0001FFF
//! offset 0x1000  MSR 0xC0010000..=0xC0011FFF
//! RDMSR/WRMSR of any other MSR always exits.
//!
//! IOPM, 12 KiB (APM 15.10.1): one bit per port, 0x0000..=0xFFFF.
//! An access to several ports exits if any of their bits is set.
//! ```
//!
//! As on Intel, every address an MTRR MSR can have exits, read and write, so
//! the guest reads and writes a copy of them (`guest_mtrrs.rs`): the CPU's
//! still set the memory types of Aleph0's own accesses. Which other MSRs and
//! ports exit is the caller's: the test guest adds IA32_KERNEL_GS_BASE and a
//! scratch port. On AMD IA32_KERNEL_GS_BASE is part of the state VMSAVE and
//! VMLOAD switch (the VMCB's `kernel_gs_base`), so the exit handler reads
//! and writes the guest's value there, not the real MSR
//! (`exit::vmcb_held_msr`).

use core::ptr::NonNull;

use super::control::{ControlArea, IOPM_SIZE, MSRPM_SIZE, MsrAccess, iopm_bit, msrpm_bit};
use crate::x86::mtrr;
use crate::x86::page::{PAGE_SIZE, allocate_zeroed_pages};

/// Sets, in a zeroed MSRPM and IOPM, the read and write bits of every
/// address an MTRR MSR can have and of `msrs`, and the bits of `ports`.
/// An MSR outside the three MSRPM ranges has no bit: it always exits.
pub fn fill(msrs: &[u32], ports: &[u16], msrpm: &mut [u8], iopm: &mut [u8]) {
    for intercepted in msrs.iter().copied().chain(mtrr::every_msr_address()) {
        for access in [MsrAccess::Read, MsrAccess::Write] {
            if let Some((offset, bit)) = msrpm_bit(intercepted, access) {
                msrpm[offset] |= 1 << bit;
            }
        }
    }
    for &port in ports {
        let (offset, bit) = iopm_bit(port);
        iopm[offset] |= 1 << bit;
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExitMapsError {
    /// UEFI could not provide the map pages.
    AllocationFailed,
}

/// The MSRPM and IOPM pages, each one contiguous allocation. Never freed:
/// the CPU reads them on every guest RDMSR, WRMSR, IN and OUT for as long as
/// the VMCB points at them.
pub struct ExitMaps {
    msrpm: NonNull<u8>,
    iopm: NonNull<u8>,
}

impl ExitMaps {
    /// Allocates the two maps zeroed and sets the bits [`fill`] sets: every
    /// MTRR address, `msrs` and `ports`.
    pub fn allocate(msrs: &[u32], ports: &[u16]) -> Result<Self, ExitMapsError> {
        let msrpm = allocate_zeroed_pages(MSRPM_SIZE / PAGE_SIZE, ExitMapsError::AllocationFailed)?;
        let iopm = allocate_zeroed_pages(IOPM_SIZE / PAGE_SIZE, ExitMapsError::AllocationFailed)?;
        // SAFETY: two separate fresh zeroed allocations of exactly these
        // sizes, which nothing else refers to while these slices live.
        let (msrpm_bytes, iopm_bytes) = unsafe {
            (
                core::slice::from_raw_parts_mut(msrpm.as_ptr(), MSRPM_SIZE),
                core::slice::from_raw_parts_mut(iopm.as_ptr(), IOPM_SIZE),
            )
        };
        fill(msrs, ports, msrpm_bytes, iopm_bytes);
        Ok(Self { msrpm, iopm })
    }

    /// Physical address of the MSRPM. UEFI identity-maps memory, so it is
    /// the allocation's address.
    pub fn msrpm_address(&self) -> u64 {
        self.msrpm.as_ptr() as u64
    }

    /// Physical address of the IOPM.
    pub fn iopm_address(&self) -> u64 {
        self.iopm.as_ptr() as u64
    }

    /// Points `control` at the two maps. Their intercepts
    /// (`intercept3::MSR_PROT`, `intercept3::IOIO_PROT`) are bring-up's to
    /// set.
    pub fn write_to(&self, control: &mut ControlArea) {
        control.msrpm_base_pa = self.msrpm_address();
        control.iopm_base_pa = self.iopm_address();
    }
}

#[cfg(test)]
#[path = "../../../tests/unit/hal/amd/exit_maps.rs"]
mod tests;
