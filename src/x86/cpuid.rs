//! CPUID-based feature detection.

use core::arch::x86_64::{__cpuid, CpuidResult};

/// The CPU's vendor, which picks the backend at run time: VMX on Intel,
/// SVM on AMD.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Vendor {
    /// "GenuineIntel".
    Intel,
    /// "AuthenticAMD".
    Amd,
    /// Any other vendor string, which Aleph0 has no backend for.
    Other([u8; 12]),
}

impl Vendor {
    /// The vendor CPUID leaf 0 names: its 12-byte string in EBX, EDX, ECX,
    /// in that order.
    pub fn from_leaf0(leaf0: CpuidResult) -> Self {
        let mut name = [0u8; 12];
        name[0..4].copy_from_slice(&leaf0.ebx.to_le_bytes());
        name[4..8].copy_from_slice(&leaf0.edx.to_le_bytes());
        name[8..12].copy_from_slice(&leaf0.ecx.to_le_bytes());
        match &name {
            b"GenuineIntel" => Self::Intel,
            b"AuthenticAMD" => Self::Amd,
            _ => Self::Other(name),
        }
    }

    /// This CPU's vendor. One CPUID, at startup, picks the backend for the
    /// whole run.
    pub fn read() -> Self {
        Self::from_leaf0(__cpuid(0))
    }
}

/// CPUID.(EAX=7,ECX=0):ECX bit 4 (OSPKE): a copy of CR4.PKE.
pub const CPUID_7_ECX_OSPKE: u32 = 1 << 4;
/// CPUID leaf of the XSAVE components: subleaf 0 reports in EDX:EAX the XCR0
/// bits the CPU supports.
pub const XSAVE_LEAF: u32 = 0xD;

/// CPUID leaf whose EAX is the highest extended leaf.
const EXTENDED_MAX: u32 = 0x8000_0000;
/// CPUID leaf of the extended feature flags.
const EXTENDED_FEATURES: u32 = 0x8000_0001;
/// CPUID leaf of the address sizes: EAX bits 7:0 are MAXPHYADDR.
pub const ADDRESS_SIZES: u32 = 0x8000_0008;
/// MAXPHYADDR when CPUID lacks [`ADDRESS_SIZES`]: 36 on a CPU with PAE,
/// which every 64-bit one has (Intel SDM Vol. 3A, "Physical Address
/// Size").
const DEFAULT_PHYSICAL_ADDRESS_BITS: u8 = 36;

/// Checks CPUID.1:EDX.MTRR\[bit 12\]: whether the CPU has memory type range
/// registers. IA32_MTRRCAP and the other MTRR MSRs exist only if it does.
pub fn has_mtrr() -> bool {
    __cpuid(1).edx >> 12 & 1 == 1
}

/// Checks CPUID.80000001H:EDX\[bit 26\] (Page1GB): whether a PDPT entry may
/// map a 1 GiB page. Leaf 0x80000001 exists on every 64-bit CPU, since its
/// EDX bit 29 is what reports long mode.
pub fn has_1g_pages() -> bool {
    __cpuid(EXTENDED_FEATURES).edx >> 26 & 1 == 1
}

/// MAXPHYADDR, how many bits a physical address has: CPUID.80000008H:EAX
/// bits 7:0. Physical-address bits from there up are reserved, in page
/// tables as in the MTRRs.
pub fn physical_address_bits() -> u8 {
    if __cpuid(EXTENDED_MAX).eax < ADDRESS_SIZES {
        return DEFAULT_PHYSICAL_ADDRESS_BITS;
    }
    __cpuid(ADDRESS_SIZES).eax as u8
}

#[cfg(test)]
#[path = "../../tests/unit/x86/cpuid.rs"]
mod tests;
