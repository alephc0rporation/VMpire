//! The run loop.
//!
//! The AMD counterpart of Intel's VMLAUNCH/VMRESUME path (`hal::intel::vmexit`).
//! Where Intel's exit jumps to HOST_RIP on a separate stack, AMD's `vmrun`
//! returns to the instruction after it, like a call. So this is a plain
//! sequence, not an entry stub:
//!
//! 1. `vmsave` the host state page: `vmrun` does not switch FS, GS, TR, LDTR,
//!    the SYSENTER MSRs or the SYSCALL MSRs, so the host's are saved by hand.
//! 2. load the 14 general-purpose registers the VMCB does not hold from the
//!    [`GprFrame`] (RAX and RSP come from the VMCB, loaded by `vmrun`).
//! 3. `clgi` to hold NMIs and interrupts across the switch.
//! 4. `vmload` the guest VMCB (its FS, GS, TR, ... into the CPU), `vmrun` it.
//! 5. on exit, `vmsave` the guest VMCB (its FS, GS, ... back), `vmload` the
//!    host state page (the host's back).
//! 6. save the 14 registers into the [`GprFrame`], `stgi`.
//!
//! The host state page that `vmsave`/`vmload` use here is a page of its own,
//! in VMCB format: not the `VM_HSAVE_PA` page (`svm::enable`), where `vmrun`
//! saves the rest of the host state in a format of the CPU's own that
//! software must not read or write (AMD64 APM Vol. 2, 15.30.4). KVM keeps
//! the two apart the same way (`save_area` and the host VMCB).
//!
//! Only the 14 general-purpose registers are saved. As in Intel's stub, the
//! `x86_64-unknown-uefi` target is soft-float with SSE disabled, so the exit
//! handler never touches vector state; before an OS guest that uses XMM runs,
//! this would have to save the guest's vector registers too.

use super::registers::GprFrame;
use super::vmcb::Vmcb;

/// `exit_code` of a failed `vmrun` (illegal guest state): -1 (APM Vol. 2,
/// 15.6). The guest did not run.
pub const VMEXIT_INVALID: u64 = u64::MAX;

core::arch::global_asm!(
    r#"
    .globl svm_vmrun
svm_vmrun:
    // Microsoft x64 ABI: rcx = guest VMCB pa, rdx = host state page pa,
    // r8 = *mut GprFrame. Preserve the callee-saved host registers (rbx,
    // rbp, rdi, rsi, r12-r15 under this ABI).
    push rbp
    push rbx
    push rdi
    push rsi
    push r12
    push r13
    push r14
    push r15

    // Stash the three arguments: after these pushes the frame is
    // [rsp]=gprs, [rsp+8]=host state pa, [rsp+16]=guest VMCB pa.
    push rcx
    push rdx
    push r8

    // Save the host's FS/GS/TR/LDTR and syscall/sysenter MSRs.
    mov rax, rdx
    vmsave rax

    // Load the guest's general-purpose registers (not RAX/RSP: vmrun loads
    // those from the VMCB). Use rax as the frame base, then load rax last.
    mov rax, r8
    mov rbx, [rax]
    mov rcx, [rax + 0x08]
    mov rdx, [rax + 0x10]
    mov rsi, [rax + 0x18]
    mov rdi, [rax + 0x20]
    mov rbp, [rax + 0x28]
    mov r8,  [rax + 0x30]
    mov r9,  [rax + 0x38]
    mov r10, [rax + 0x40]
    mov r11, [rax + 0x48]
    mov r12, [rax + 0x50]
    mov r13, [rax + 0x58]
    mov r14, [rax + 0x60]
    mov r15, [rax + 0x68]

    // Enter the guest. rax = guest VMCB pa for vmload/vmrun/vmsave.
    mov rax, [rsp + 16]
    clgi
    vmload rax
    vmrun rax
    vmsave rax

    // Restore the host's FS/GS/TR/... and allow interrupts again.
    mov rax, [rsp + 8]
    vmload rax
    stgi

    // Save the guest's general-purpose registers. Reload the frame pointer
    // from the stack: the guest may have changed rsi/rdi. push rax first so
    // the stashed pointers keep their offsets.
    push rax
    mov rax, [rsp + 8]
    mov [rax],        rbx
    mov [rax + 0x08], rcx
    mov [rax + 0x10], rdx
    mov [rax + 0x18], rsi
    mov [rax + 0x20], rdi
    mov [rax + 0x28], rbp
    mov [rax + 0x30], r8
    mov [rax + 0x38], r9
    mov [rax + 0x40], r10
    mov [rax + 0x48], r11
    mov [rax + 0x50], r12
    mov [rax + 0x58], r13
    mov [rax + 0x60], r14
    mov [rax + 0x68], r15
    pop rax

    // Discard the three stashed arguments and restore callee-saved.
    add rsp, 24
    pop r15
    pop r14
    pop r13
    pop r12
    pop rsi
    pop rdi
    pop rbx
    pop rbp
    ret
"#
);

unsafe extern "C" {
    /// Enters the guest and returns when it exits. `guest_vmcb_pa` is the
    /// guest VMCB's physical address, `host_state_pa` that of the host state
    /// page: a zeroed, 4 KiB-aligned page of its own that `vmsave` and
    /// `vmload` keep the host's FS, GS, TR, LDTR and syscall MSRs in (not the
    /// `svm::enable` page), and `gprs` the 14 registers the VMCB does not
    /// hold. Identity-mapped pages make a virtual address its own physical
    /// address.
    pub(super) fn svm_vmrun(guest_vmcb_pa: u64, host_state_pa: u64, gprs: *mut GprFrame);
}

// The asm addresses the frame by the same byte offsets this struct uses:
// a reordered field would silently load the wrong register.
const _: () = {
    use core::mem::offset_of;
    assert!(offset_of!(GprFrame, rbx) == 0x00);
    assert!(offset_of!(GprFrame, rcx) == 0x08);
    assert!(offset_of!(GprFrame, rdx) == 0x10);
    assert!(offset_of!(GprFrame, rsi) == 0x18);
    assert!(offset_of!(GprFrame, rdi) == 0x20);
    assert!(offset_of!(GprFrame, rbp) == 0x28);
    assert!(offset_of!(GprFrame, r8) == 0x30);
    assert!(offset_of!(GprFrame, r15) == 0x68);
};

/// What a VMRUN ([`svm_vmrun`]) left in the VMCB control area.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RunOutcome {
    /// `exit_code`: the reason the guest exited ([`VMEXIT_INVALID`] when
    /// `vmrun` refused the guest).
    pub exit_code: u64,
    /// `exit_info_1`.
    pub exit_info_1: u64,
    /// `exit_info_2`.
    pub exit_info_2: u64,
}

impl RunOutcome {
    /// Reads the exit fields from the control area.
    pub fn read(vmcb: &Vmcb) -> Self {
        let control = &vmcb.control;
        Self {
            exit_code: u64::from(control.exit_code) | (u64::from(control.exit_code_hi) << 32),
            exit_info_1: control.exit_info_1,
            exit_info_2: control.exit_info_2,
        }
    }

    /// Whether `vmrun` refused the guest (illegal state): the guest never ran.
    pub fn is_invalid(self) -> bool {
        self.exit_code == VMEXIT_INVALID
    }
}

#[cfg(test)]
#[path = "../../../tests/unit/hal/amd/run.rs"]
mod tests;
