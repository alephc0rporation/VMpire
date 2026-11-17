//! The AMD SVM backend.
//!
//! On AMD the guest is described by a VMCB (`vmcb.rs`): a 4 KiB page of
//! ordinary memory that VMRUN reads on entry and writes back on every exit.
//! The VMCB holds the guest's RAX, RSP, RIP, RFLAGS, control registers and
//! more; the other general-purpose registers stay in the CPU, and the run
//! loop saves them itself (`registers::GprFrame`).
//!
//! Parts:
//! - `svm.rs`: turning SVM on.
//! - `control.rs`: the control area and which events exit.
//! - `guest.rs`: guest state setup into the VMCB save area.
//! - `run.rs`: VMRUN and the register save/restore around it.
//! - `npt/`: nested page tables, the identity map EPT is on Intel.
//! - `exit_maps.rs`: the MSR and I/O permission maps.
//! - `events.rs`: the guest's NMI masking, and the NMI and interrupt
//!   windows SVM has no exit for.
//! - `vmcb.rs`, `registers.rs`: the VMCB layout and the registers it does
//!   not hold.
//!
//! The `hal` traits, one file each, as in `hal/intel`:
//! `virtualization.rs`, `vcpu.rs` (with the run loop), `vm_exit.rs`,
//! `nested_paging.rs`, `intercepts.rs`, `hypercall.rs`. Everything the
//! exit handler decides is the shared code's (`hv`), as on Intel.

pub mod control;
pub mod events;
pub mod exit_maps;
pub mod guest;
pub mod npt;
pub mod registers;
pub mod run;
pub mod svm;
pub mod vmcb;

mod hypercall;
mod intercepts;
mod nested_paging;
mod vcpu;
mod virtualization;
pub(crate) mod vm_exit;

// One type per `hal` trait (HAL_STRUCTURE.md section 1).
pub use hypercall::Hypercall;
pub use intercepts::Intercepts;
pub use nested_paging::NestedPaging;
pub use vcpu::Vcpu;
pub use virtualization::HardwareVirtualization;
pub use vm_exit::VmExit;
