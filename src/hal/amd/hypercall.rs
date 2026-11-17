//! `hal::Hypercall` for SVM: VMMCALL. Aleph0's own call from the guest
//! (`hv::vmcall`), and Hyper-V's hypercalls, forwarded with VMMCALL. The AMD
//! counterpart of `hal::intel::hypercall`; the rest of Hyper-V's interface
//! (detection, synthetic MSRs, the MSR forwarders) is shared:
//! `hv::hyperv`.
//!
//! On AMD, Hyper-V's hypercall page holds VMMCALL, not VMCALL, so the
//! guest's hypercalls exit with code VMMCALL (0x81, `exit::code::VMMCALL`,
//! with `control::intercept4::VMMCALL` set; without that intercept VMMCALL
//! raises #UD in the guest). Aleph0 repeats each one with VMMCALL itself,
//! outside the guest, so L0 Hyper-V serves it as an access of Aleph0's own
//! virtual processor, as on Intel:
//!
//! ```text
//! guest (L2)                    Aleph0 (L1)                        Hyper-V (L0)
//! VMMCALL, RCX/RDX/R8 ─#VMEXIT─▶ forward_hypercall: same RCX/RDX/R8,
//!                    0x81        VMMCALL ─────────────────────────▶ serves it
//!        RAX, RCX, RDX, R8 ◀──── into the guest's registers ◀─────── RAX = status
//! ```
//!
//! A hypercall before the guest enabled the hypercall page gets #UD from
//! Hyper-V; the host fault handler resumes it at the forwarder's recovery
//! point (`recovery_point`), and the guest gets the #UD at its own VMMCALL.

use core::arch::asm;

use crate::hal::types::{HypercallRegisters, RecoveryPoint, Refused};

// `svm_hyperv_forward_hypercall` reads and writes the fields at these
// offsets.
const _: () = {
    assert!(core::mem::offset_of!(HypercallRegisters, rcx) == 0);
    assert!(core::mem::offset_of!(HypercallRegisters, rdx) == 8);
    assert!(core::mem::offset_of!(HypercallRegisters, r8) == 16);
    assert!(core::mem::offset_of!(HypercallRegisters, rax) == 24);
};

/// IDT vector the hypercall forwarder recovers from.
const INVALID_OPCODE: u8 = 6;

// The hypercall forwarder, Microsoft x64 ABI (`efiapi`), returning 0 when
// VMMCALL ran and 1 when Hyper-V raised #UD: Intel's
// `hyperv_forward_hypercall` with VMMCALL (0F 01 D9) for VMCALL.
core::arch::global_asm!(
    r#"
    .globl svm_hyperv_forward_hypercall
    .globl svm_hyperv_hypercall_instruction
    .globl svm_hyperv_hypercall_refused
svm_hyperv_forward_hypercall:
    push rcx
    mov rax, rcx
    mov rcx, [rax]
    mov rdx, [rax + 8]
    mov r8, [rax + 16]
svm_hyperv_hypercall_instruction:
    vmmcall
    pop r9
    mov [r9], rcx
    mov [r9 + 8], rdx
    mov [r9 + 16], r8
    mov [r9 + 24], rax
    xor eax, eax
    ret
svm_hyperv_hypercall_refused:
    pop r9
    mov eax, 1
    ret
"#
);

unsafe extern "efiapi" {
    fn svm_hyperv_forward_hypercall(registers: *mut HypercallRegisters) -> u64;
}

unsafe extern "C" {
    static svm_hyperv_hypercall_instruction: u8;
    static svm_hyperv_hypercall_refused: u8;
}

/// VMMCALL.
pub struct Hypercall;

impl crate::hal::Hypercall for Hypercall {
    unsafe fn call(rcx: u64) {
        // SAFETY: the caller guarantees Aleph0's guest, so the VMMCALL exits
        // to Aleph0, which changes no guest register for it and resumes past
        // it. It touches no memory and no stack.
        unsafe {
            asm!(
                "vmmcall",
                inout("rcx") rcx => _,
                options(nomem, nostack),
            );
        }
    }

    /// The guest's hypercall in `registers`, made again outside the guest:
    /// Hyper-V's results replace them.
    unsafe fn forward(registers: &mut HypercallRegisters) -> Result<(), Refused> {
        // SAFETY: the caller's guarantees; the forwarder reads and writes
        // only `registers` and its own stack.
        match unsafe { svm_hyperv_forward_hypercall(registers) } {
            0 => Ok(()),
            _ => Err(Refused),
        }
    }

    fn recovery_point() -> RecoveryPoint {
        RecoveryPoint {
            instruction: &raw const svm_hyperv_hypercall_instruction,
            vector: INVALID_OPCODE,
            resume: &raw const svm_hyperv_hypercall_refused,
        }
    }
}

#[cfg(test)]
#[path = "../../../tests/unit/hal/amd/hypercall.rs"]
mod tests;
