//! The AMD backend of [`GuestRegisters`]: where SVM leaves each guest
//! register after an exit.
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
//! A write takes effect at the next VMRUN, the same as on Intel. Nothing
//! here can fail: every register is plain memory.

use core::convert::Infallible;

use super::vmcb::Vmcb;
use crate::hal::amd::register_access::{GuestRegister, GuestRegisters};

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

/// The guest's registers at one exit: its VMCB and the frame the run loop
/// saved.
pub struct SvmGuestRegisters<'a> {
    vmcb: &'a mut Vmcb,
    gprs: &'a mut GprFrame,
}

impl<'a> SvmGuestRegisters<'a> {
    pub fn new(vmcb: &'a mut Vmcb, gprs: &'a mut GprFrame) -> Self {
        Self { vmcb, gprs }
    }
}

impl GuestRegisters for SvmGuestRegisters<'_> {
    type Error = Infallible;

    fn read(&self, register: GuestRegister) -> Result<u64, Infallible> {
        let save = &self.vmcb.save;
        let gprs = &*self.gprs;
        Ok(match register {
            GuestRegister::Rax => save.rax,
            GuestRegister::Rbx => gprs.rbx,
            GuestRegister::Rcx => gprs.rcx,
            GuestRegister::Rdx => gprs.rdx,
            GuestRegister::Rsi => gprs.rsi,
            GuestRegister::Rdi => gprs.rdi,
            GuestRegister::Rbp => gprs.rbp,
            GuestRegister::Rsp => save.rsp,
            GuestRegister::R8 => gprs.r8,
            GuestRegister::R9 => gprs.r9,
            GuestRegister::R10 => gprs.r10,
            GuestRegister::R11 => gprs.r11,
            GuestRegister::R12 => gprs.r12,
            GuestRegister::R13 => gprs.r13,
            GuestRegister::R14 => gprs.r14,
            GuestRegister::R15 => gprs.r15,
            GuestRegister::Rip => save.rip,
            GuestRegister::Rflags => save.rflags,
            GuestRegister::Cr0 => save.cr0,
            GuestRegister::Cr3 => save.cr3,
            GuestRegister::Cr4 => save.cr4,
            GuestRegister::Dr7 => save.dr7,
            GuestRegister::Efer => save.efer,
            GuestRegister::Pat => save.g_pat,
            GuestRegister::FsBase => save.fs.base,
            GuestRegister::GsBase => save.gs.base,
        })
    }

    fn write(&mut self, register: GuestRegister, value: u64) -> Result<(), Infallible> {
        let save = &mut self.vmcb.save;
        let gprs = &mut *self.gprs;
        let slot = match register {
            GuestRegister::Rax => &mut save.rax,
            GuestRegister::Rbx => &mut gprs.rbx,
            GuestRegister::Rcx => &mut gprs.rcx,
            GuestRegister::Rdx => &mut gprs.rdx,
            GuestRegister::Rsi => &mut gprs.rsi,
            GuestRegister::Rdi => &mut gprs.rdi,
            GuestRegister::Rbp => &mut gprs.rbp,
            GuestRegister::Rsp => &mut save.rsp,
            GuestRegister::R8 => &mut gprs.r8,
            GuestRegister::R9 => &mut gprs.r9,
            GuestRegister::R10 => &mut gprs.r10,
            GuestRegister::R11 => &mut gprs.r11,
            GuestRegister::R12 => &mut gprs.r12,
            GuestRegister::R13 => &mut gprs.r13,
            GuestRegister::R14 => &mut gprs.r14,
            GuestRegister::R15 => &mut gprs.r15,
            GuestRegister::Rip => &mut save.rip,
            GuestRegister::Rflags => &mut save.rflags,
            GuestRegister::Cr0 => &mut save.cr0,
            GuestRegister::Cr3 => &mut save.cr3,
            GuestRegister::Cr4 => &mut save.cr4,
            GuestRegister::Dr7 => &mut save.dr7,
            GuestRegister::Efer => &mut save.efer,
            GuestRegister::Pat => &mut save.g_pat,
            GuestRegister::FsBase => &mut save.fs.base,
            GuestRegister::GsBase => &mut save.gs.base,
        };
        *slot = value;
        Ok(())
    }
}

#[cfg(test)]
#[path = "../../../tests/unit/hal/amd/registers.rs"]
mod tests;
