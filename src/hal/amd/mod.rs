//! The AMD SVM backend.
//!
//! On AMD the guest is described by a VMCB (`vmcb.rs`): a 4 KiB page of
//! ordinary memory that VMRUN reads on entry and writes back on every exit.
//! The VMCB holds the guest's RAX, RSP, RIP, RFLAGS, control registers and
//! more; the other general-purpose registers stay in the CPU, and the run
//! loop saves them itself (`registers::GprFrame`).
//!
//! Parts, in the order of `hal-notes/AMD_TODO.md`:
//! - `svm.rs` (1): turning SVM on.
//! - `control.rs` (2): the control area and which events exit.
//! - `guest.rs` (3): guest state setup into the VMCB save area.
//! - `run.rs` (4): the run loop (vmrun and the register save/restore).
//! - `exit.rs` (5): exit decode and the acts (CPUID, MSR, I/O, HLT, RIP).
//! - `features.rs`: what the guest is offered (`GuestFeatures`), which
//!   CPUID hides the rest of.
//! - `npt/` (6): nested page tables, the identity map EPT is on Intel.
//! - `exit_maps.rs`, `guest_mtrrs.rs` (7): the MSR and I/O permission maps,
//!   and the guest's copy of its MTRRs, which the NPT's types follow.
//! - `events.rs` (8): NMIs, reinjected as Intel does, and the events an exit
//!   interrupted, re-delivered.
//! - `hyperv.rs` (9): Hyper-V's hypercalls, forwarded with VMMCALL.
//! - `vmcb.rs`, `registers.rs`: the VMCB layout and guest register access.

// SVM bring-up and the run loop do not exist yet, so only the host tests use
// this backend for now.
#![allow(dead_code, reason = "no SVM bring-up or run loop uses the backend yet")]

pub mod bring_up;
pub mod control;
pub mod events;
pub mod exit;
pub mod exit_maps;
pub mod features;
pub mod guest;
pub mod guest_mtrrs;
pub mod handler;
pub mod hyperv;
pub mod npt;
pub mod register_access;
pub mod registers;
pub mod run;
pub mod svm;
pub mod test_guest;
pub mod vmcb;
