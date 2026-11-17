//! `hal::Vcpu` for SVM: the VMCB and the run loop, over the functions of
//! `control`, `guest`, `run` and `events`. The VMCB fill, the guest state,
//! the switch to the host environment and the loop are the AMD bring-up's
//! and exit handler's; each exit goes to the shared exit handler
//! (`hv::vmexit::handle_exit`) through [`VmExit`](super::VmExit).
//!
//! Where Intel's VM exit loads the host state from the VMCS, VMRUN saves the
//! state it runs on and every #VMEXIT restores it, so the run loop switches
//! to Aleph0's host environment first ([`switch_to_host`]), as the AMD
//! bring-up did. As Intel's exits run on the VM-exit stack, the run loop
//! runs on a stack of its own ([`RUN_STACK_SIZE`]): for the firmware guest
//! ([`Vcpu::run_in_place`]) the firmware goes on as the guest on the stack it
//! ran on, so the loop cannot share it.

use core::arch::asm;
use core::mem::offset_of;
use core::ptr::NonNull;
use core::sync::atomic::{AtomicBool, AtomicPtr, Ordering};

use super::control::{ControlArea, NESTED_CTL_NP_ENABLE, intercept3, intercept4, tlb_control};
use super::events::{self, EventInjection, NmiState};
use super::guest::segment_from_descriptor;
use super::intercepts::Intercepts;
use super::nested_paging::NestedPaging;
use super::npt::HOST_PAT;
use super::registers::GprFrame;
use super::run::{RunOutcome, svm_vmrun};
use super::svm::{EFER_SVME, SvmSupport};
use super::vm_exit::{self, IA32_CSTAR, IA32_LSTAR, IA32_SFMASK, IA32_STAR, VmExit};
use super::vmcb::{SaveArea, Segment, Vmcb};
use crate::hal::types::{GuestEntry, GuestKind, GuestRegisters, GuestStop, HostTables, OptionalFeatures};
use crate::hv::exit_decision::ExitAction;
use crate::x86::cr::{self, CrControls};
use crate::x86::msr::{self, IA32_EFER, IA32_PAT};
use crate::x86::page::{PAGE_SIZE, allocate_zeroed_page, allocate_zeroed_pages};
use crate::x86::segment::{self, DescriptorTable};

/// The ASID the guest runs with. 0 is the host's; any other is a guest's.
const GUEST_ASID: u32 = 1;
/// RFLAGS with only the reserved bit 1 set: IF = 0, as the test guest runs.
const RFLAGS_RESERVED: u64 = 0x2;
/// DR7 and DR6 at reset.
const DR7_RESET: u64 = 0x400;
const DR6_RESET: u64 = 0xFFFF_0FF0;

/// The run loop's stack: 64 KiB, as Intel's VM-exit stack, whose exits
/// used 21168 bytes in a debug build.
const RUN_STACK_PAGES: usize = 16;
const RUN_STACK_SIZE: usize = RUN_STACK_PAGES * PAGE_SIZE;

/// Everything one vCPU keeps, in pages of its own that are never freed: the
/// VMCB, the registers the VMCB does not hold, and what the run loop
/// remembers between exits.
#[repr(C)]
pub(super) struct VcpuState {
    /// The guest VMCB; first, so it stays 4 KiB-aligned.
    pub(super) vmcb: Vmcb,
    /// The 14 registers saved across VMRUN.
    pub(super) gprs: GprFrame,
    /// The guest's NMIs: masking, the IRET it is stopped at, single-stepping.
    pub(super) nmi: NmiState,
    /// The NMI window the exit handler asked for. Atomic: the host NMI
    /// handler sets it too.
    pub(super) nmi_window: AtomicBool,
    /// The interrupt window the exit handler asked for.
    pub(super) interrupt_window: AtomicBool,
    /// The page VMSAVE and VMLOAD keep the host's FS, GS, TR, LDTR and
    /// syscall MSRs in (not the VM_HSAVE_PA page).
    host_state: u64,
    /// The host environment and CR4 [`Vcpu::run`] switches to.
    host: Option<(HostTables, u64)>,
    /// The firmware's PAT, which the run replaced with [`HOST_PAT`] and
    /// [`Vcpu::release`] puts back; `None` until the run switched.
    firmware_pat: Option<u64>,
    /// How the run loop ended, for the code that switched to its stack.
    stop: Option<GuestStop<VcpuError>>,
}

/// The vCPU current on this CPU: [`Vcpu::create`] sets it,
/// [`Vcpu::release`] clears it. The exit handler reaches it through
/// [`VmExit`], and the nested paging through [`current_control`].
static CURRENT: AtomicPtr<VcpuState> = AtomicPtr::new(core::ptr::null_mut());

/// The current vCPU's state.
///
/// # Safety
///
/// A vCPU is current (after [`Vcpu::create`]).
pub(super) unsafe fn current() -> *mut VcpuState {
    let state = CURRENT.load(Ordering::Acquire);
    debug_assert!(!state.is_null(), "no current vCPU");
    state
}

/// The current VMCB's control area, for a change the next VMRUN must see
/// (a TLB flush).
///
/// # Safety
///
/// Exit context: a vCPU is current and the guest is stopped, and nothing
/// else holds a reference to its control area.
pub(super) unsafe fn current_control<'a>() -> &'a mut ControlArea {
    // SAFETY: the caller's guarantees.
    unsafe { &mut (*current()).vmcb.control }
}

/// The VMCB, its host state page and the run loop's memory and stack.
pub struct Vcpu {
    state: NonNull<VcpuState>,
    /// The run loop's stack, [`RUN_STACK_SIZE`] bytes, zeroed at first.
    stack: NonNull<u8>,
}

/// Why a vCPU step failed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VcpuError {
    /// UEFI could not provide the VMCB or host state pages.
    AllocationFailed,
    /// No next-RIP save (CPUID 0x8000000A:EDX.NRIPS): Aleph0 moves RIP past
    /// an instruction with the VMCB's NRIP, as Intel does with the exit's
    /// instruction length.
    NoNextRipSave,
    /// The run started before the host state was configured.
    HostStateMissing,
    /// VMRUN refused the guest state (exit code -1) at the first entry.
    InvalidState,
    /// The VMCB self-test read back something else than was written.
    SelfTest,
}

impl Vcpu {
    fn state(&self) -> *mut VcpuState {
        self.state.as_ptr()
    }

    /// Top of the run stack, 16-byte aligned for the x64 calling convention.
    fn stack_top(&self) -> u64 {
        let top = self.stack.as_ptr() as u64 + RUN_STACK_SIZE as u64;
        assert_eq!(top % 16, 0);
        top
    }

    /// The host environment the run switches to, or why there is none.
    fn host_configured(&self) -> Result<(), VcpuError> {
        // SAFETY: the guest is not running, so nothing writes the state.
        match unsafe { (*self.state()).host } {
            Some(_) => Ok(()),
            None => Err(VcpuError::HostStateMissing),
        }
    }
}

/// Clears GIF: NMIs and interrupts are held until VMRUN sets it again in the
/// guest, or STGI in the host.
fn clgi() {
    // SAFETY: SVM is on (EFER.SVME), which CLGI needs; it changes only GIF.
    unsafe { asm!("clgi", options(nomem, nostack, preserves_flags)) };
}

/// Sets GIF again, when the run loop reports a window without a VMRUN
/// (which would have set it).
fn stgi() {
    // SAFETY: SVM is on (EFER.SVME), which STGI needs; it changes only GIF.
    unsafe { asm!("stgi", options(nomem, nostack, preserves_flags)) };
}

/// The guest state of the current CPU, as the VMCB save area holds it: its
/// segments, descriptor tables, control registers, `efer` and `pat`, and the
/// syscall MSRs VMLOAD loads. RIP, RSP, RFLAGS and the IDT are the caller's.
///
/// # Safety
///
/// The current GDT must be live and readable through its limit.
unsafe fn current_guest_state(efer: u64, pat: u64) -> SaveArea {
    let gdtr = segment::read_gdtr();
    // SAFETY: the caller guarantees a readable GDT.
    let state = unsafe { segment::read_all() };
    let read = |selector: u16, base: u64| -> Segment {
        // SAFETY: the caller guarantees a readable GDT.
        match unsafe { segment::read_descriptor(&gdtr, selector) } {
            Some(descriptor) => segment_from_descriptor(selector, descriptor.0, base),
            None => Segment {
                selector,
                ..Segment::default()
            },
        }
    };
    let mut save = SaveArea::zeroed();
    save.cs = read(state.cs, 0);
    save.ss = read(state.ss, 0);
    save.ds = read(state.ds, 0);
    save.es = read(state.es, 0);
    save.fs = read(state.fs, state.fs_base);
    save.gs = read(state.gs, state.gs_base);
    save.tr = read(state.tr, state.tr_base);
    save.gdtr = Segment {
        selector: 0,
        attrib: 0,
        limit: u32::from(gdtr.limit),
        base: gdtr.base,
    };
    save.cpl = 0;
    save.efer = efer;
    save.cr0 = cr::read_cr0();
    save.cr3 = cr::read_cr3();
    save.cr4 = cr::read_cr4();
    save.dr7 = DR7_RESET;
    save.dr6 = DR6_RESET;
    save.g_pat = pat;
    // SAFETY: every 64-bit CPU has these MSRs; reading has no side effect.
    unsafe {
        save.star = msr::rdmsr(IA32_STAR);
        save.lstar = msr::rdmsr(IA32_LSTAR);
        save.cstar = msr::rdmsr(IA32_CSTAR);
        save.sfmask = msr::rdmsr(IA32_SFMASK);
        save.kernel_gs_base = msr::rdmsr(msr::IA32_KERNEL_GS_BASE);
        save.sysenter_cs = msr::rdmsr(msr::IA32_SYSENTER_CS);
        save.sysenter_esp = msr::rdmsr(msr::IA32_SYSENTER_ESP);
        save.sysenter_eip = msr::rdmsr(msr::IA32_SYSENTER_EIP);
    }
    save
}

/// Switches the host to Aleph0's own environment, the AMD counterpart of a
/// VM exit loading the VMCS host fields: interrupts off, `host_cr4`
/// (OSXSAVE added when Aleph0 runs XSETBV), [`HOST_PAT`] for the nested
/// walks, then its CR3, GDT, TR and IDT. VMRUN saves this state and every
/// #VMEXIT restores it, so every exit runs on it.
///
/// # Safety
///
/// `host`'s tables must be complete and never freed, and its GDT must keep
/// the current selectors (`HostEnvironment::build` with the firmware's).
unsafe fn switch_to_host(host: &HostTables, host_cr4: u64) {
    // SAFETY: the caller's guarantees. The TR selector names the host GDT's
    // available TSS descriptor, which LTR marks busy.
    unsafe {
        asm!("cli", options(nomem, nostack));
        cr::write_cr4(host_cr4);
        msr::wrmsr(IA32_PAT, HOST_PAT);
        cr::write_cr3(host.cr3);
        segment::load_gdtr(&DescriptorTable {
            base: host.gdtr_base,
            limit: host.gdtr_limit,
        });
        asm!("ltr {0:x}", in(reg) host.selectors.tr, options(nostack, preserves_flags));
        segment::load_idtr(&DescriptorTable {
            base: host.idtr_base,
            limit: host.idtr_limit,
        });
    }
}

/// The exit the run loop reports for a window it opened itself: SVM has no
/// NMI-window or interrupt-window exit of the kind VMX has.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Window {
    Nmi,
    Interrupt,
}

/// Before VMRUN, with GIF clear: what the window the exit handler asked
/// for needs, worked out as the AMD exit handler did for a pending NMI
/// (`events::nmi_plan`), with the window in the place of the pending NMI:
///
/// - None asked for, or already single-stepping: nothing.
/// - Nothing blocks (no interrupt shadow, no NMI masking for an NMI window,
///   no other event queued): the window is open now, and the run loop
///   reports it without entering the guest.
/// - An NMI window while the guest is in the NMI handler Aleph0 injected,
///   before its IRET exit: the IRET intercept, on since the injection,
///   brings the next exit.
/// - Otherwise: single-step what blocks, TF and RF set, #DB intercepted.
///
/// An interrupt window waits only for the shadow (and a queued event).
fn prepare_windows(state: &mut VcpuState) -> Option<Window> {
    let window = if state.nmi_window.load(Ordering::Acquire) {
        Window::Nmi
    } else if state.interrupt_window.load(Ordering::Acquire) {
        Window::Interrupt
    } else {
        return None;
    };
    let probe = match window {
        Window::Nmi => NmiState {
            pending: true,
            ..state.nmi
        },
        Window::Interrupt => NmiState {
            pending: true,
            masked: false,
            iret_rip: None,
            ..state.nmi
        },
    };
    let control = &mut state.vmcb.control;
    let event_queued = control.event_inj & EventInjection::VALID != 0;
    let (plan, next) = events::nmi_plan(probe, control.int_state, state.vmcb.save.rflags, event_queued);
    if plan.inject {
        return Some(window);
    }
    state.nmi.single_step = next.single_step;
    if let Some(rflags) = events::apply(plan, control) {
        state.vmcb.save.rflags = rflags;
    }
    None
}

/// After an exit the run loop handles itself, the IRET or #DB of its own
/// NMI window machinery (`events::iret_exit`, `events::debug_exit`): whether
/// it was one. Neither reaches the exit handler.
fn own_exit(state: &mut VcpuState, code: u64) -> bool {
    match code {
        events::code::IRET => {
            let (plan, next) = events::iret_exit(state.nmi, state.vmcb.save.rip);
            state.nmi = next;
            events::apply(plan, &mut state.vmcb.control);
            true
        }
        events::code::DEBUG => match events::debug_exit(state.nmi, state.vmcb.save.rflags) {
            (events::DebugExit::NmiStep(plan), next) => {
                state.nmi = next;
                if let Some(rflags) = events::apply(plan, &mut state.vmcb.control) {
                    state.vmcb.save.rflags = rflags;
                }
                true
            }
            (events::DebugExit::Guest, _) => false,
        },
        _ => false,
    }
}

/// The guest's registers for the exit handler: the 14 the run loop saved,
/// and RAX from the VMCB.
fn registers(state: &VcpuState) -> GuestRegisters {
    let gprs = &state.gprs;
    GuestRegisters {
        rax: state.vmcb.save.rax,
        rbx: gprs.rbx,
        rcx: gprs.rcx,
        rdx: gprs.rdx,
        rsi: gprs.rsi,
        rdi: gprs.rdi,
        rbp: gprs.rbp,
        r8: gprs.r8,
        r9: gprs.r9,
        r10: gprs.r10,
        r11: gprs.r11,
        r12: gprs.r12,
        r13: gprs.r13,
        r14: gprs.r14,
        r15: gprs.r15,
    }
}

/// Puts the exit handler's changes to `frame` back where VMRUN takes them.
fn store_registers(state: &mut VcpuState, frame: &GuestRegisters) {
    state.vmcb.save.rax = frame.rax;
    state.gprs = GprFrame {
        rbx: frame.rbx,
        rcx: frame.rcx,
        rdx: frame.rdx,
        rsi: frame.rsi,
        rdi: frame.rdi,
        rbp: frame.rbp,
        r8: frame.r8,
        r9: frame.r9,
        r10: frame.r10,
        r11: frame.r11,
        r12: frame.r12,
        r13: frame.r13,
        r14: frame.r14,
        r15: frame.r15,
    };
}

/// The AMD exit handler's loop, on the run stack: switch to the host
/// environment, then for each round, with GIF clear, open the windows the
/// exit handler asked for; VMRUN; handle the run loop's own IRET and #DB
/// exits; give every other exit to the shared exit handler. Returns how the
/// guest stopped; for the firmware only a refused first VMRUN returns, any
/// other stop halts in the exit handler.
///
/// # Safety
///
/// The vCPU is current and configured, the guest not running, so nothing
/// else uses `state`.
unsafe fn run_loop(state: *mut VcpuState) -> GuestStop<VcpuError> {
    // SAFETY: the caller guarantees the vCPU is current and configured,
    // and the guest not running, so nothing else uses the state.
    let Some((host, host_cr4)) = (unsafe { (*state).host }) else {
        return GuestStop::LaunchFailed {
            error: VcpuError::HostStateMissing,
        };
    };
    // SAFETY: PAT exists on every 64-bit CPU. The host tables are
    // complete and never freed (`configure_host_state`).
    unsafe {
        (*state).firmware_pat = Some(msr::rdmsr(IA32_PAT));
        switch_to_host(&host, host_cr4);
    }
    let mut entered = false;
    loop {
        // 1. The windows, with GIF clear.
        clgi();
        // SAFETY: the guest is stopped; only this loop and the exit
        // handler (not running now) use the state.
        let window = prepare_windows(unsafe { &mut *state });
        let code = if let Some(window) = window {
            stgi();
            match window {
                Window::Nmi => vm_exit::NMI_WINDOW,
                Window::Interrupt => vm_exit::INTERRUPT_WINDOW,
            }
        } else {
            // 2. VMRUN.
            // SAFETY: SVM is on, the VMCB describes the guest the vCPU
            // was configured with, and the host state page is live and
            // never freed. Identity-mapped memory makes the addresses
            // physical.
            unsafe {
                let vmcb = &raw mut (*state).vmcb as u64;
                svm_vmrun(vmcb, (*state).host_state, &raw mut (*state).gprs);
            }
            // SAFETY: the guest stopped at this exit.
            let state = unsafe { &mut *state };
            let outcome = RunOutcome::read(&state.vmcb);
            if outcome.is_invalid() {
                if !entered {
                    return GuestStop::LaunchFailed {
                        error: VcpuError::InvalidState,
                    };
                }
                // A resume the CPU refused: the stop report of the last
                // exit, as Intel's failed VMRESUME (`vmresume_failed`).
                crate::output::set_phase(crate::output::Phase::VmExit);
                crate::serial_println!(
                    "{} VMRUN refused the guest state (exit code -1); stopping the guest; {}.",
                    crate::TAG,
                    crate::hv::vmexit::after_stop(GuestKind::SELECTED)
                );
                crate::hv::vmexit::stop_after_failed_resume(&registers(state));
                return GuestStop::ResumeFailed {
                    error: VcpuError::InvalidState,
                };
            }
            entered = true;
            let code = outcome.exit_code;
            // The CPU took the queued event or reported it in
            // EXITINTINFO; the first run's flush is done.
            let control = &mut state.vmcb.control;
            control.event_inj = 0;
            control.tlb_control = tlb_control::NONE;
            state.nmi = state.nmi.at_exit(state.vmcb.save.rip);
            if own_exit(state, code) {
                continue;
            }
            code as u32
        };

        // 3. The exit handler.
        // SAFETY: as above.
        let mut frame = registers(unsafe { &*state });
        // SAFETY: exit context: SVM on, this vCPU current, the guest
        // stopped at this exit (or at the window the loop opened).
        let mut exit = unsafe { VmExit::at(state, window.is_some().then_some(code)) };
        let (action, number) = crate::hv::vmexit::handle_exit(&mut exit, &mut frame);
        // SAFETY: as above.
        store_registers(unsafe { &mut *state }, &frame);
        match action {
            ExitAction::Resume(_) => {}
            ExitAction::Finished => return GuestStop::Finished { exits: number },
            ExitAction::Unexpected => return GuestStop::Unexpected { number, reason: code },
        }
    }
}

/// [`run_loop`] for the code that switched to the run stack: keeps how the
/// guest stopped in the state, and returns to the stack it came from.
extern "efiapi" fn run_loop_entry(state: *mut VcpuState) {
    // SAFETY: `Vcpu::run` and `Vcpu::run_in_place` pass the current vCPU's
    // state, configured, with the guest not running.
    let stop = unsafe { run_loop(state) };
    // SAFETY: as above; the loop has ended.
    unsafe { (*state).stop = Some(stop) };
}

impl crate::hal::Vcpu for Vcpu {
    type Error = VcpuError;

    fn allocate() -> Result<Self, VcpuError> {
        let pages = size_of::<VcpuState>().div_ceil(PAGE_SIZE);
        let state = allocate_zeroed_pages(pages, VcpuError::AllocationFailed)?.cast::<VcpuState>();
        let host_state = allocate_zeroed_page(VcpuError::AllocationFailed)?.as_ptr() as u64;
        let stack = allocate_zeroed_pages(RUN_STACK_PAGES, VcpuError::AllocationFailed)?;
        // SAFETY: a fresh page-aligned allocation as long as a `VcpuState`,
        // which nothing else refers to. Never freed.
        unsafe {
            state.as_ptr().write(VcpuState {
                vmcb: Vmcb::zeroed(),
                gprs: GprFrame::default(),
                nmi: NmiState::default(),
                nmi_window: AtomicBool::new(false),
                interrupt_window: AtomicBool::new(false),
                host_state,
                host: None,
                firmware_pat: None,
                stop: None,
            });
        }
        let vcpu = Self { state, stack };
        log!(
            "VMCB allocated at {:#018x}; host state page at {host_state:#018x}; run stack top={:#018x} size={RUN_STACK_SIZE}.",
            state.as_ptr() as u64,
            vcpu.stack_top(),
        );
        Ok(vcpu)
    }

    unsafe fn create(&mut self) -> Result<(), VcpuError> {
        // A VMCB is plain memory, zeroed by `allocate`: nothing to load.
        CURRENT.store(self.state(), Ordering::Release);
        Ok(())
    }

    /// Puts the firmware's PAT back, if the run replaced it.
    unsafe fn release(&self) -> Result<(), VcpuError> {
        // SAFETY: the caller guarantees the guest is not running, so nothing
        // else uses the state.
        let state = unsafe { &mut *self.state() };
        if let Some(pat) = state.firmware_pat.take() {
            // SAFETY: PAT exists on every 64-bit CPU, and this is the value
            // it held before the run.
            unsafe { msr::wrmsr(IA32_PAT, pat) };
        }
        CURRENT.store(core::ptr::null_mut(), Ordering::Release);
        Ok(())
    }

    unsafe fn self_test(&self) -> Result<(), VcpuError> {
        const PATTERN: u64 = 0xA5A5_5A5A_0F0F_F0F0;
        // SAFETY: the VMCB is this vCPU's own memory, and no guest runs yet.
        let read = unsafe {
            let control = &raw mut (*self.state()).vmcb.control.tsc_offset;
            control.write_volatile(PATTERN);
            let read = control.read_volatile();
            control.write_volatile(0);
            read
        };
        if read != PATTERN {
            return Err(VcpuError::SelfTest);
        }
        log!("VMCB self-test passed: a control-area field read back what was written.");
        Ok(())
    }

    /// The intercepts of the AMD bring-up: NMI, shutdown, CPUID, HLT, the
    /// I/O and MSR permission maps, VMRUN (which must be intercepted, or VMRUN
    /// refuses the guest, APM 15.5.1), VMMCALL and XSETBV; every exception for
    /// the test guest, as on Intel, none for the firmware.
    unsafe fn configure_controls(
        &self,
        support: &SvmSupport,
        kind: GuestKind,
        _features: OptionalFeatures,
    ) -> Result<(), VcpuError> {
        if !support.next_rip_save() {
            return Err(VcpuError::NoNextRipSave);
        }
        // SAFETY: the caller guarantees the vCPU is current with no guest
        // running.
        let control = unsafe { &mut (*self.state()).vmcb.control };
        control.intercept_word3 = intercept3::NMI
            | intercept3::SHUTDOWN
            | intercept3::CPUID
            | intercept3::HLT
            | intercept3::IOIO_PROT
            | intercept3::MSR_PROT;
        control.intercept_word4 = intercept4::VMRUN | intercept4::VMMCALL | intercept4::XSETBV;
        control.intercept_exceptions = match kind {
            GuestKind::BuiltInTest => u32::MAX,
            GuestKind::Firmware => 0,
        };
        control.asid = GUEST_ASID;
        control.tlb_control = tlb_control::FLUSH_ALL;
        log!(
            "VMCB intercepts for the {kind:?} guest: word3={:#010x} word4={:#010x} exceptions={:#010x}, ASID {GUEST_ASID}.",
            control.intercept_word3,
            control.intercept_word4,
            control.intercept_exceptions,
        );
        Ok(())
    }

    unsafe fn set_nested_paging(&self, paging: &NestedPaging) -> Result<(), VcpuError> {
        // SAFETY: as for `configure_controls`.
        let control = unsafe { &mut (*self.state()).vmcb.control };
        control.nested_ctl = NESTED_CTL_NP_ENABLE;
        control.nested_cr3 = paging.npt().ncr3().raw();
        log!("VMCB nested paging on: N_CR3={:#018x}.", control.nested_cr3);
        Ok(())
    }

    unsafe fn set_intercepts(&self, intercepts: &Intercepts) -> Result<(), VcpuError> {
        // SAFETY: as for `configure_controls`.
        let control = unsafe { &mut (*self.state()).vmcb.control };
        intercepts.maps().write_to(control);
        log!(
            "VMCB MSRPM={:#018x} IOPM={:#018x}.",
            control.msrpm_base_pa,
            control.iopm_base_pa
        );
        Ok(())
    }

    /// Kept for [`run`](Self::run), which switches to it: VMRUN saves the
    /// state it runs on, and every exit restores it.
    unsafe fn configure_host_state(&self, tables: &HostTables, cr4: u64) -> Result<(), VcpuError> {
        // SAFETY: as for `configure_controls`.
        let state = unsafe { &mut *self.state() };
        state.host = Some((*tables, cr4));
        log!(
            "Host environment for the run: CR3={:#018x} GDT={:#018x} TSS={:#018x} IDT={:#018x} CR4={cr4:#x}; switched to when the guest starts.",
            tables.cr3,
            tables.gdtr_base,
            tables.tr_base,
            tables.idtr_base,
        );
        Ok(())
    }

    /// Nothing: VMRUN returns to the run loop after each exit.
    unsafe fn configure_exit_entry(&self) -> Result<(), VcpuError> {
        log!("VMRUN returns to the run loop at each exit: no exit entry to set.");
        Ok(())
    }

    /// Nothing to write: SVM fixes no CR0/CR4 bit, so the guest owns them all
    /// and no control-register access exits.
    unsafe fn configure_cr_intercepts(&self, controls: &CrControls) -> Result<(), VcpuError> {
        log!(
            "CR0/CR4: no intercepts; SVM fixes no bit, the guest owns them (CR0 mask={:#x}, CR4 mask={:#x}).",
            controls.cr0_mask,
            controls.cr4_mask,
        );
        Ok(())
    }

    unsafe fn set_guest_state(&self, entry: GuestEntry) -> Result<(), VcpuError> {
        // The firmware's RIP, RSP and RFLAGS are written by `run_in_place`,
        // which starts it where it is.
        let (rip, rsp, idtr) = match entry {
            GuestEntry::At { rip, rsp, idtr } => (rip, rsp, idtr),
            GuestEntry::InPlace { idtr } => (0, 0, idtr),
        };
        // SAFETY: as for `configure_controls`.
        let vmcb = unsafe { &mut (*self.state()).vmcb };
        // SAFETY: EFER and PAT exist on every 64-bit CPU. EFER has SVME
        // already (`enable`), which VMRUN needs in the guest's too.
        let (efer, pat) = unsafe { (msr::rdmsr(IA32_EFER), msr::rdmsr(IA32_PAT)) };
        // SAFETY: the caller guarantees the GDT is live and stable.
        vmcb.save = unsafe { current_guest_state(efer | EFER_SVME, pat) };
        vmcb.save.rip = rip;
        vmcb.save.rsp = rsp;
        vmcb.save.rflags = RFLAGS_RESERVED;
        vmcb.save.idtr = Segment {
            selector: 0,
            attrib: 0,
            limit: u32::from(idtr.limit),
            base: idtr.base,
        };
        log!(
            "VMCB at {:#018x}: guest RIP={:#018x} RSP={:#018x} CR3={:#018x} CR4={:#x} EFER={:#x} CS={:#06x} (attrib {:#05x}).",
            &raw const *vmcb as u64,
            vmcb.save.rip,
            vmcb.save.rsp,
            vmcb.save.cr3,
            vmcb.save.cr4,
            vmcb.save.efer,
            vmcb.save.cs.selector,
            vmcb.save.cs.attrib,
        );
        if let GuestEntry::InPlace { .. } = entry {
            log!(
                "Guest rip, rsp and rflags are placeholders: the in-place launch writes its own, and the code after it goes on as the guest."
            );
        }
        Ok(())
    }

    /// The run loop ([`run_loop`]) on the run stack, until the test guest
    /// stops.
    unsafe fn run(&self) -> GuestStop<VcpuError> {
        if let Err(error) = self.host_configured() {
            return GuestStop::LaunchFailed { error };
        }
        let state = self.state();
        // SAFETY: switches to the run stack, Aleph0's own and never freed,
        // calls the loop with the Microsoft x64 ABI (RSP 16-byte aligned at
        // the call, 32 bytes of shadow space) and switches back. R12 keeps
        // the caller's RSP across the call (callee-saved); every register the
        // ABI lets the callee change is clobbered.
        unsafe {
            asm!(
                "mov r12, rsp",
                "mov rsp, {top}",
                "sub rsp, 32",
                "call {entry}",
                "mov rsp, r12",
                top = in(reg) self.stack_top(),
                entry = sym run_loop_entry,
                in("rcx") state,
                out("r12") _,
                clobber_abi("efiapi"),
            );
        }
        // SAFETY: the loop has ended; nothing else uses the state.
        unsafe { (*state).stop.take() }.expect("the run loop records how the guest stopped")
    }

    /// The firmware goes on as the guest, as Intel's `launch_in_place`: the
    /// guest's RIP is the instruction after the switch, its RSP and RFLAGS
    /// the ones it runs with, and its other registers the ones the code
    /// around it keeps (the run loop loads them from the frame at VMRUN).
    /// The run loop then runs on the run stack and never comes back here,
    /// except when the first VMRUN is refused.
    unsafe fn run_in_place(&self) -> Result<(), GuestStop<VcpuError>> {
        if let Err(error) = self.host_configured() {
            return Err(GuestStop::LaunchFailed { error });
        }
        let state = self.state();
        // SAFETY: as for `configure_controls`; the addresses are taken, not
        // references, so the asm below writes the fields directly.
        let (save, gprs) = unsafe { (&raw mut (*state).vmcb.save, &raw mut (*state).gprs) };
        let failed: u64;
        // SAFETY: writes RIP, RSP and RFLAGS into the save area and the
        // registers into the frame, then switches stacks as `run` does. The
        // guest starts at `2:` with RSP as it is here and every register as
        // stored, so the code after this block goes on as before, as the
        // guest; registers the ABI lets a call change, and R12, are
        // clobbered. PUSHFQ and POP balance before RSP is stored. If the loop
        // returns (the first VMRUN refused), RSP comes back from R12.
        unsafe {
            asm!(
                "lea rax, [rip + 2f]",
                "mov [{save} + {rip}], rax",
                "pushfq",
                "pop rax",
                "mov [{save} + {rflags}], rax",
                "mov [{save} + {rsp}], rsp",
                "mov [{gprs} + {rbx}], rbx",
                "mov [{gprs} + {rsi}], rsi",
                "mov [{gprs} + {rdi}], rdi",
                "mov [{gprs} + {rbp}], rbp",
                "mov [{gprs} + {r12}], r12",
                "mov [{gprs} + {r13}], r13",
                "mov [{gprs} + {r14}], r14",
                "mov [{gprs} + {r15}], r15",
                "mov r12, rsp",
                "mov rsp, {top}",
                "sub rsp, 32",
                "call {entry}",
                "mov rsp, r12",
                "mov eax, 1",
                "jmp 3f",
                // The guest starts here.
                "2:",
                "xor eax, eax",
                "3:",
                save = in(reg) save,
                gprs = in(reg) gprs,
                top = in(reg) self.stack_top(),
                rip = const offset_of!(SaveArea, rip),
                rflags = const offset_of!(SaveArea, rflags),
                rsp = const offset_of!(SaveArea, rsp),
                rbx = const offset_of!(GprFrame, rbx),
                rsi = const offset_of!(GprFrame, rsi),
                rdi = const offset_of!(GprFrame, rdi),
                rbp = const offset_of!(GprFrame, rbp),
                r12 = const offset_of!(GprFrame, r12),
                r13 = const offset_of!(GprFrame, r13),
                r14 = const offset_of!(GprFrame, r14),
                r15 = const offset_of!(GprFrame, r15),
                entry = sym run_loop_entry,
                in("rcx") state,
                out("rax") failed,
                out("r12") _,
                clobber_abi("efiapi"),
            );
        }
        if failed == 0 {
            return Ok(());
        }
        // SAFETY: the loop has ended; nothing else uses the state.
        Err(unsafe { (*state).stop.take() }.expect("the run loop records how the guest stopped"))
    }

    /// The run stack: bytes from its lowest written word to its top.
    unsafe fn exit_stack_usage(&self) -> Option<(usize, usize)> {
        // SAFETY: the stack's own allocation, RUN_STACK_SIZE bytes, which no
        // run loop writes now (the caller guarantees the guest stopped).
        let words = unsafe {
            core::slice::from_raw_parts(self.stack.as_ptr().cast::<u64>(), RUN_STACK_SIZE / size_of::<u64>())
        };
        let used = words
            .iter()
            .position(|&word| word != 0)
            .map_or(0, |lowest| (words.len() - lowest) * size_of::<u64>());
        Some((used, RUN_STACK_SIZE))
    }

    fn report_error(error: VcpuError) {
        match error {
            VcpuError::InvalidState => log!("VMRUN refused the guest state (exit code -1)."),
            error => log!("VMCB step failed: {error:?}"),
        }
    }
}
