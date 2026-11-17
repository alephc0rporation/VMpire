//! Where SVM leaves each guest register after an exit.
//!
//! - RAX, RSP, RIP, RFLAGS, CR0, CR3, CR4, DR7, EFER and PAT: in the VMCB's
//!   save area, which #VMEXIT writes and VMRUN loads. PAT is the VMCB's
//!   G_PAT, which the CPU uses for the guest only with nested paging on.
//! - FS and GS bases: in the save area too, but moved by VMSAVE and VMLOAD,
//!   not by VMRUN (see `vmcb.rs`).
//! - RBX, RCX, RDX, RSI, RDI, RBP and R8 to R15: nowhere in the VMCB. The CPU
//!   leaves them live at the exit, so the run loop pushes them into a
//!   [`GprFrame`] and pops them back before the next VMRUN.
//!
//! The run loop hands them to the exit handler as one `GuestRegisters`
//! frame, RAX taken from the VMCB (`vcpu::registers`).

/// The general-purpose registers the VMCB does not hold, as the run loop
/// saves them after an exit.
#[repr(C)]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct GprFrame {
    pub rbx: u64,
    pub rcx: u64,
    pub rdx: u64,
    pub rsi: u64,
    pub rdi: u64,
    pub rbp: u64,
    pub r8: u64,
    pub r9: u64,
    pub r10: u64,
    pub r11: u64,
    pub r12: u64,
    pub r13: u64,
    pub r14: u64,
    pub r15: u64,
}

// The run loop's assembly will push and pop this frame by offset: keep it
// 14 registers with RBX first and R15 last.
const _: () = {
    assert!(size_of::<GprFrame>() == 14 * 8);
    assert!(core::mem::offset_of!(GprFrame, rbx) == 0);
    assert!(core::mem::offset_of!(GprFrame, r15) == 13 * 8);
};
