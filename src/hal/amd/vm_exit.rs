//! `hal::VmExit` for SVM: the exit as the VMCB holds it after VMRUN
//! returned (EXITCODE, EXITINFO1/2, EXITINTINFO, the save area), decoded
//! into the neutral types the way the AMD exit handler decoded it
//! (`exit::decode`, `events`), and the VMCB writes before the next VMRUN.
//!
//! The VMCB is plain memory, so every field is always readable: no reader
//! fails for a field it could not read, as a VMREAD can.

use core::fmt;
use core::sync::atomic::Ordering;

use super::events::{self, EventInjection, NmiPlan};
use super::vcpu::{self, VcpuState};
use super::vmcb::SaveArea;
use crate::hal::types::{
    CrAccess, CrAccessKind, CrNumber, CrWrite, EventKind, ExitReason, GuestRegisters, InjectedEvent, Interruptibility,
    IoAccess, IoSize, NMI_VECTOR, NestedPageFault, ReasonDescription,
};
use crate::x86::msr::{self, IA32_KERNEL_GS_BASE, IA32_SYSENTER_CS, IA32_SYSENTER_EIP, IA32_SYSENTER_ESP};

/// SVM exit codes (AMD64 APM Vol. 2, Appendix C).
pub mod code {
    /// Read of CR0 to CR15: 0x00 plus the register.
    pub const CR_READ: u32 = 0x00;
    /// Write of CR0 to CR15: 0x10 plus the register.
    pub const CR_WRITE: u32 = 0x10;
    /// An exception: 0x40 plus its vector.
    pub const EXCEPTION: u32 = 0x40;
    /// A physical NMI while the guest ran.
    pub const NMI: u32 = 0x61;
    /// A virtual interrupt the guest can take (VINTR).
    pub const VINTR: u32 = 0x64;
    /// The guest's IRET, before it runs.
    pub const IRET: u32 = 0x74;
    pub const CPUID: u32 = 0x72;
    pub const HLT: u32 = 0x78;
    pub const IOIO: u32 = 0x7B;
    pub const MSR: u32 = 0x7C;
    /// A triple fault in the guest.
    pub const SHUTDOWN: u32 = 0x7F;
    pub const VMRUN: u32 = 0x80;
    pub const VMMCALL: u32 = 0x81;
    pub const MONITOR: u32 = 0x8A;
    pub const MWAIT: u32 = 0x8B;
    pub const XSETBV: u32 = 0x8D;
    /// A nested page fault.
    pub const NPF: u32 = 0x400;
    /// VMRUN refused the guest state: -1.
    pub const INVALID: u32 = u32::MAX;
}

/// The exit the run loop reports when it opened an NMI window itself, with
/// IRET intercepts and single steps (`vcpu::prepare_windows`): SVM has no
/// NMI-window exit. Reported with the IRET exit's code, the means of it.
pub(super) const NMI_WINDOW: u32 = code::IRET;
/// The exit the run loop reports when it opened an interrupt window
/// itself: VINTR's code, SVM's interrupt-window exit.
pub(super) const INTERRUPT_WINDOW: u32 = code::VINTR;

/// IA32_STAR, IA32_LSTAR, IA32_CSTAR and IA32_SFMASK: the SYSCALL MSRs.
pub(super) const IA32_STAR: u32 = 0xC000_0081;
pub(super) const IA32_LSTAR: u32 = 0xC000_0082;
pub(super) const IA32_CSTAR: u32 = 0xC000_0083;
pub(super) const IA32_SFMASK: u32 = 0xC000_0084;

/// IOIO EXITINFO1: bit 0 IN, bit 2 a string instruction, bit 3 REP, bits 6:4
/// the operand size (8, 16, 32 bits), bits 31:16 the port.
const IOIO_TYPE_IN: u64 = 1 << 0;
const IOIO_STRING: u64 = 1 << 2;
const IOIO_REP: u64 = 1 << 3;
const IOIO_SZ8: u64 = 1 << 4;
const IOIO_SZ16: u64 = 1 << 5;
const IOIO_SZ32: u64 = 1 << 6;

/// MSR EXITINFO1 bit 0: WRMSR (1) or RDMSR (0).
const MSR_WRITE: u64 = 1 << 0;

/// NPF EXITINFO1: bit 0 the page was present, 1 a write, 3 a reserved bit
/// set in an entry, 4 an instruction fetch, 33 the fault was on a guest
/// page-table access, not on the final translation.
const NPF_PRESENT: u64 = 1 << 0;
const NPF_WRITE: u64 = 1 << 1;
const NPF_RESERVED: u64 = 1 << 3;
const NPF_FETCH: u64 = 1 << 4;
const NPF_GUEST_PAGE_TABLE: u64 = 1 << 33;

/// CR access EXITINFO1 with decode assists: bit 63 valid, bits 3:0 the
/// general-purpose register of the MOV.
const CR_DECODE_VALID: u64 = 1 << 63;

/// EVENTINJ / EXITINTINFO: bits 10:8 the type, bit 11 an error code, bit 31
/// valid.
const EVENT_TYPE_SHIFT: u32 = 8;
const EVENT_ERROR_VALID: u32 = 1 << 11;
const EVENT_VALID: u32 = 1 << 31;

/// Exception vectors that push an error code.
fn has_error_code(vector: u8) -> bool {
    matches!(vector, 8 | 10..=14 | 17 | 21 | 29 | 30)
}

/// Why a reader could not answer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VmExitError {
    /// A control-register exit without decode assists: the CPU does not say
    /// which register the MOV used.
    NoDecodeAssist,
    /// No next RIP in the VMCB (no NRIPS): the instruction length is
    /// unknown.
    NoNextRip,
}

/// The guest stopped at an exit, in the current vCPU's VMCB.
pub struct VmExit {
    state: *mut VcpuState,
    /// The window the run loop opened itself, with the code it reports for
    /// it, in the place of the VMCB's EXITCODE.
    window: Option<u32>,
}

impl VmExit {
    /// The exit of `state`, or the window (`window`) the run loop opened.
    ///
    /// # Safety
    ///
    /// Exit context with `state` the current vCPU's, which nothing else
    /// uses while this lives.
    pub(super) unsafe fn at(state: *mut VcpuState, window: Option<u32>) -> Self {
        Self { state, window }
    }

    fn state(&self) -> &VcpuState {
        // SAFETY: `at`'s guarantees.
        unsafe { &*self.state }
    }

    fn state_mut(&mut self) -> &mut VcpuState {
        // SAFETY: `at`'s guarantees.
        unsafe { &mut *self.state }
    }

    /// EXITCODE, as the reports and counters show it (-1 as 0xFFFF_FFFF).
    fn code(&self) -> u32 {
        self.window.unwrap_or(self.state().vmcb.control.exit_code)
    }

    fn info1(&self) -> u64 {
        if self.window.is_some() {
            0
        } else {
            self.state().vmcb.control.exit_info_1
        }
    }

    fn info2(&self) -> u64 {
        if self.window.is_some() {
            0
        } else {
            self.state().vmcb.control.exit_info_2
        }
    }

    /// EXITINTINFO with its error code, `None` for a window the loop opened.
    fn exit_int_info(&self) -> Option<(u32, u32)> {
        let control = &self.state().vmcb.control;
        self.window
            .is_none()
            .then_some((control.exit_int_info, control.exit_int_info_err))
    }
}

/// The neutral reason of exit `code` with EXITINFO1 `info1`.
pub(crate) fn decode_reason(code: u32, info1: u64) -> ExitReason {
    match code {
        code::INVALID => ExitReason::EntryFailure(code),
        0x00..=0x1F => ExitReason::ControlRegisterAccess,
        0x40..=0x5F | code::NMI => ExitReason::ExceptionOrNmi,
        code::VINTR => ExitReason::InterruptWindow,
        code::IRET => ExitReason::NmiWindow,
        code::CPUID => ExitReason::Cpuid,
        code::HLT => ExitReason::Hlt,
        code::VMMCALL => ExitReason::Hypercall,
        code::IOIO => ExitReason::IoInstruction,
        code::MSR if info1 & MSR_WRITE != 0 => ExitReason::Wrmsr,
        code::MSR => ExitReason::Rdmsr,
        code::MWAIT => ExitReason::Mwait,
        code::MONITOR => ExitReason::Monitor,
        code::NPF if info1 & NPF_RESERVED != 0 => ExitReason::NestedPagingMisconfig,
        code::NPF => ExitReason::NestedPageFault,
        code::XSETBV => ExitReason::Xsetbv,
        _ => ExitReason::Other(code),
    }
}

/// The exception or NMI of exit `code`, with its error code from EXITINFO1.
pub(crate) fn decode_exception_or_nmi(code: u32, info1: u64) -> Option<InjectedEvent> {
    match code {
        code::NMI => Some(InjectedEvent::NMI),
        0x40..=0x5F => {
            let vector = (code - code::EXCEPTION) as u8;
            Some(InjectedEvent {
                kind: EventKind::HardwareException,
                vector,
                error_code: has_error_code(vector).then_some(info1 as u32),
                instruction_len: None,
            })
        }
        _ => None,
    }
}

/// The EVENTINJ / EXITINTINFO type of each event kind (AMD64 APM Vol. 2,
/// 15.20): 0 an external interrupt, 2 an NMI, 3 an exception, 4 a software
/// interrupt (INTn), 6 a software exception (INT3, INTO), as the AMD exit
/// handler re-armed them.
fn event_type(kind: EventKind) -> u32 {
    match kind {
        EventKind::ExternalInterrupt => 0,
        EventKind::Nmi => 2,
        EventKind::HardwareException | EventKind::PrivilegedSoftwareException => 3,
        EventKind::SoftwareInterrupt => 4,
        EventKind::SoftwareException => 6,
        EventKind::Other(kind) => u32::from(kind),
    }
}

/// The event EXITINTINFO (`info`, with `error_code`) reports, `None` when
/// the exit interrupted no delivery. `length` is the instruction length, for
/// a software interrupt or exception. Vector 2 is re-delivered as a true
/// NMI, whatever type it has, as the AMD exit handler's `redelivery` and
/// Intel's do.
pub(crate) fn decode_interrupted(info: u32, error_code: u32, length: Option<u32>) -> Option<InjectedEvent> {
    if info & EVENT_VALID == 0 {
        return None;
    }
    let vector = info as u8;
    let event_type = ((info >> EVENT_TYPE_SHIFT) & 0b111) as u8;
    if vector == NMI_VECTOR && matches!(event_type, 2 | 3) {
        return Some(InjectedEvent::NMI);
    }
    let kind = match event_type {
        0 => EventKind::ExternalInterrupt,
        2 => EventKind::Nmi,
        3 => EventKind::HardwareException,
        4 => EventKind::SoftwareInterrupt,
        6 => EventKind::SoftwareException,
        other => EventKind::Other(other),
    };
    let event = InjectedEvent {
        kind,
        vector,
        error_code: (info & EVENT_ERROR_VALID != 0).then_some(error_code),
        instruction_len: None,
    };
    Some(InjectedEvent {
        instruction_len: if event.needs_instruction_len() { length } else { None },
        ..event
    })
}

/// IOIO EXITINFO1 as an access.
pub(crate) fn decode_io_access(info1: u64) -> IoAccess {
    let size = if info1 & IOIO_SZ8 != 0 {
        Some(IoSize::Byte)
    } else if info1 & IOIO_SZ16 != 0 {
        Some(IoSize::Word)
    } else if info1 & IOIO_SZ32 != 0 {
        Some(IoSize::Dword)
    } else {
        None
    };
    IoAccess {
        port: (info1 >> 16) as u16,
        size,
        is_in: info1 & IOIO_TYPE_IN != 0,
        is_string: info1 & IOIO_STRING != 0,
        is_rep: info1 & IOIO_REP != 0,
    }
}

/// NPF EXITINFO1 and EXITINFO2 (the GPA) as a fault. SVM reports the access
/// and whether the page was present, not the entry's permissions: a write to
/// a present page is one to a page mapped read-only, as the AMD exit handler
/// took it, and the linear address is never reported.
pub(crate) fn decode_fault(info1: u64, gpa: u64) -> NestedPageFault {
    let present = info1 & NPF_PRESENT != 0;
    let write = info1 & NPF_WRITE != 0;
    let fetch = info1 & NPF_FETCH != 0;
    NestedPageFault {
        gpa,
        read: !write && !fetch,
        write,
        fetch,
        page_readable: present,
        page_writable: !(present && write),
        page_executable: !(present && fetch),
        linear_valid: true,
        translated_access: info1 & NPF_GUEST_PAGE_TABLE == 0,
    }
}

/// The VMCB field holding the guest's value of `msr` while the host runs,
/// or `None` for an MSR the CPU does not switch, whose real register is the
/// guest's. VMLOAD and VMSAVE switch these with the guest's VMCB around each
/// VMRUN (AMD64 APM Vol. 2, 15.5.2), so a passthrough RDMSR or WRMSR of one
/// reads and writes this field: a real RDMSR in the host would reach the
/// host's value. KVM's `svm_get_msr` and `svm_set_msr` do the same.
pub(crate) fn vmcb_held_msr(save: &mut SaveArea, msr: u32) -> Option<&mut u64> {
    match msr {
        IA32_KERNEL_GS_BASE => Some(&mut save.kernel_gs_base),
        IA32_STAR => Some(&mut save.star),
        IA32_LSTAR => Some(&mut save.lstar),
        IA32_CSTAR => Some(&mut save.cstar),
        IA32_SFMASK => Some(&mut save.sfmask),
        IA32_SYSENTER_CS => Some(&mut save.sysenter_cs),
        IA32_SYSENTER_ESP => Some(&mut save.sysenter_esp),
        IA32_SYSENTER_EIP => Some(&mut save.sysenter_eip),
        _ => None,
    }
}

/// `event` as EVENTINJ's low half.
fn encode(event: InjectedEvent) -> u32 {
    EVENT_VALID
        | (event_type(event.kind) << EVENT_TYPE_SHIFT)
        | if event.error_code.is_some() {
            EVENT_ERROR_VALID
        } else {
            0
        }
        | u32::from(event.vector)
}

/// What exit `code` is, for the reports.
pub(crate) fn code_name(code: u32) -> &'static str {
    match code {
        code::CR_READ..=0x0F => "CR read",
        0x10..=0x1F => "CR write",
        0x40..=0x5F => "exception",
        code::NMI => "NMI",
        code::VINTR => "VINTR (interrupt window)",
        code::IRET => "IRET (NMI window)",
        code::CPUID => "CPUID",
        code::HLT => "HLT",
        code::IOIO => "IOIO",
        code::MSR => "MSR",
        code::SHUTDOWN => "shutdown",
        code::VMRUN => "VMRUN",
        code::VMMCALL => "VMMCALL",
        code::MONITOR => "MONITOR",
        code::MWAIT => "MWAIT",
        code::XSETBV => "XSETBV",
        code::NPF => "nested page fault",
        code::INVALID => "VMEXIT_INVALID",
        _ => "other",
    }
}

/// What the history keeps of an exit: EXITCODE, EXITINFO1/2, RIP and
/// EXITINTINFO.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Record {
    code: Option<u32>,
    info1: u64,
    info2: u64,
    rip: u64,
    exit_int_info: u32,
}

impl fmt::Display for Record {
    /// `code=0x7c (MSR) rip=0x7eae1028 info1=0x1 info2=0x0 intinfo=0x0`.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.code {
            Some(code) => write!(f, "code={code:#x} ({})", code_name(code))?,
            None => f.write_str("code=?")?,
        }
        write!(
            f,
            " rip={:#x} info1={:#x} info2={:#x} intinfo={:#x}",
            self.rip, self.info1, self.info2, self.exit_int_info
        )
    }
}

impl crate::hal::VmExit for VmExit {
    type Error = VmExitError;
    type Record = Record;

    const HYPERCALL: &'static str = "VMMCALL";
    const EMPTY_RECORD: Record = Record {
        code: None,
        info1: 0,
        info2: 0,
        rip: 0,
        exit_int_info: 0,
    };
    /// Exit codes 0 to 0x400 (NPF), the highest Aleph0 meets; -1
    /// (VMEXIT_INVALID) counts with the other reasons.
    const REASON_COUNTERS: usize = 0x401;

    unsafe fn current() -> Self {
        // SAFETY: the caller guarantees exit context, so a vCPU is current.
        let state = unsafe { vcpu::current() };
        Self { state, window: None }
    }

    fn reason(&self) -> Result<ExitReason, VmExitError> {
        Ok(decode_reason(self.code(), self.info1()))
    }

    fn reason_code(&self) -> Result<u32, VmExitError> {
        Ok(self.code())
    }

    fn exception_or_nmi(&self) -> Result<Option<InjectedEvent>, VmExitError> {
        Ok(decode_exception_or_nmi(self.code(), self.info1()))
    }

    fn interrupted_event(&self) -> Result<Option<InjectedEvent>, VmExitError> {
        let length = self.instruction_length().ok().map(|length| length as u32);
        Ok(self
            .exit_int_info()
            .and_then(|(info, error_code)| decode_interrupted(info, error_code, length)))
    }

    /// The VMCB's one interrupt shadow bit (STI and MOV SS alike), reported
    /// as the STI shadow, and the NMI masking the run loop keeps
    /// (`events::NmiState`).
    fn interruptibility(&self) -> Result<Interruptibility, VmExitError> {
        let state = self.state();
        Ok(Interruptibility {
            sti_shadow: state.vmcb.control.int_state & events::INTERRUPT_SHADOW != 0,
            mov_ss_shadow: false,
            nmi_blocked: state.nmi.masked,
            other: 0,
        })
    }

    fn cr_access(&self) -> Result<CrAccess, VmExitError> {
        let code = self.code();
        let info1 = self.info1();
        if info1 & CR_DECODE_VALID == 0 {
            return Err(VmExitError::NoDecodeAssist);
        }
        Ok(CrAccess {
            cr: (code & 0xF) as u8,
            kind: if code >= code::CR_WRITE {
                CrAccessKind::MovTo
            } else {
                CrAccessKind::MovFrom
            },
            gpr: (info1 & 0xF) as u8,
        })
    }

    fn io_access(&self) -> Result<IoAccess, VmExitError> {
        Ok(decode_io_access(self.info1()))
    }

    fn nested_page_fault(&self) -> Result<NestedPageFault, VmExitError> {
        Ok(decode_fault(self.info1(), self.info2()))
    }

    /// Never: SVM keeps no NMI blocking of its own for an IRET to end.
    fn faulted_nmi_unblocking_iret(&self) -> Result<bool, VmExitError> {
        Ok(false)
    }

    fn guest_cpl(&self) -> Result<u8, VmExitError> {
        Ok(self.state().vmcb.save.cpl)
    }

    fn guest_rip(&self) -> Result<u64, VmExitError> {
        Ok(self.state().vmcb.save.rip)
    }

    /// NRIP minus RIP: Aleph0 moves RIP past an instruction with the VMCB's
    /// NRIP (`Vcpu::configure_controls` requires NRIPS).
    fn instruction_length(&self) -> Result<u64, VmExitError> {
        let vmcb = &self.state().vmcb;
        match vmcb.control.next_rip {
            0 => Err(VmExitError::NoNextRip),
            next => Ok(next.wrapping_sub(vmcb.save.rip)),
        }
    }

    fn guest_rsp(&self) -> Result<u64, VmExitError> {
        Ok(self.state().vmcb.save.rsp)
    }

    fn guest_cr4(&self) -> Result<u64, VmExitError> {
        Ok(self.state().vmcb.save.cr4)
    }

    /// The VMCB's copy for an MSR VMLOAD and VMSAVE switch
    /// ([`vmcb_held_msr`]), the real MSR otherwise.
    unsafe fn guest_msr(&self, msr: u32) -> u64 {
        // SAFETY: `at`'s guarantees; the field is read, not kept.
        let save = unsafe { &mut (*self.state).vmcb.save };
        match vmcb_held_msr(save, msr) {
            Some(field) => *field,
            // SAFETY: the caller guarantees the MSR exists and has no read
            // side effect.
            None => unsafe { msr::rdmsr(msr) },
        }
    }

    fn record(&self) -> Record {
        let state = self.state();
        Record {
            code: Some(self.code()),
            info1: self.info1(),
            info2: self.info2(),
            rip: state.vmcb.save.rip,
            exit_int_info: self.exit_int_info().map_or(0, |(info, _)| info),
        }
    }

    /// Never: the VMCB is memory.
    fn unreadable(&self) -> bool {
        false
    }

    fn set_rip(&mut self, rip: u64) -> Result<(), VmExitError> {
        self.state_mut().vmcb.save.rip = rip;
        Ok(())
    }

    /// As for [`guest_msr`](crate::hal::VmExit::guest_msr).
    unsafe fn set_guest_msr(&mut self, msr: u32, value: u64) {
        match vmcb_held_msr(&mut self.state_mut().vmcb.save, msr) {
            Some(field) => *field = value,
            // SAFETY: the caller guarantees the MSR exists and takes `value`.
            None => unsafe { msr::wrmsr(msr, value) },
        }
    }

    /// The shadow bit, and the NMI masking the run loop keeps: cleared for
    /// an NMI re-delivery, with the IRET it waited for forgotten.
    fn set_interruptibility(&mut self, interruptibility: Interruptibility) -> Result<(), VmExitError> {
        let state = self.state_mut();
        if interruptibility.sti_shadow || interruptibility.mov_ss_shadow {
            state.vmcb.control.int_state |= events::INTERRUPT_SHADOW;
        } else {
            state.vmcb.control.int_state &= !events::INTERRUPT_SHADOW;
        }
        state.nmi.masked = interruptibility.nmi_blocked;
        if !interruptibility.nmi_blocked {
            state.nmi.iret_rip = None;
        }
        Ok(())
    }

    /// EVENTINJ, with the error code in its high half. An NMI masks the
    /// guest's NMIs and turns the IRET intercept on, as the AMD exit
    /// handler's injection did (`events::nmi_plan`).
    fn inject_event(&mut self, event: InjectedEvent) -> Result<(), VmExitError> {
        let state = self.state_mut();
        if event.is_nmi() {
            let plan = NmiPlan {
                inject: true,
                intercept_iret: Some(true),
                ..NmiPlan::default()
            };
            events::apply(plan, &mut state.vmcb.control);
            state.nmi.masked = true;
            state.nmi.iret_rip = None;
            return Ok(());
        }
        let error_code = u64::from(event.error_code.unwrap_or(0));
        state.vmcb.control.event_inj = EventInjection(u64::from(encode(event)) | (error_code << 32)).0;
        Ok(())
    }

    /// Kept for the run loop, which opens the window itself
    /// (`vcpu::prepare_windows`).
    fn set_nmi_window(&mut self, on: bool) -> Result<(), VmExitError> {
        self.state().nmi_window.store(on, Ordering::Release);
        Ok(())
    }

    /// As for [`set_nmi_window`](crate::hal::VmExit::set_nmi_window).
    fn set_interrupt_window(&mut self, on: bool) -> Result<(), VmExitError> {
        self.state().interrupt_window.store(on, Ordering::Release);
        Ok(())
    }

    /// The save area's CR0 or CR4: SVM has no read shadow, and fixes no bit,
    /// so what the guest reads is what it has.
    fn write_cr(&mut self, write: CrWrite) -> Result<(), (&'static str, VmExitError)> {
        let save = &mut self.state_mut().vmcb.save;
        match write.cr {
            CrNumber::Cr0 => save.cr0 = write.real,
            CrNumber::Cr4 => save.cr4 = write.real,
        }
        Ok(())
    }

    /// The AMD exit handler's line: the code, RIP, EXITINFO1/2; RAX to RDX;
    /// and EXITINTINFO when the exit interrupted a delivery.
    fn report_read(&self, number: u64, registers: [u64; 4]) {
        let tag = crate::TAG;
        let code = self.code();
        let rip = self.state().vmcb.save.rip;
        crate::serial_println!(
            "{tag} VM exit #{number}: {} (code {code:#x}){} at rip={rip:#018x} info1={:#x} info2={:#x}.",
            code_name(code),
            if code == code::INVALID {
                " [VMRUN refused the guest state; guest did not run]"
            } else {
                ""
            },
            self.info1(),
            self.info2(),
        );
        let [rax, rbx, rcx, rdx] = registers;
        crate::serial_println!(
            "{tag} VM exit #{number}: guest rax={rax:#018x} rbx={rbx:#018x} rcx={rcx:#018x} rdx={rdx:#018x}"
        );
        if let Some((info, error_code)) = self.exit_int_info()
            && info & EVENT_VALID != 0
        {
            crate::serial_println!(
                "{tag} VM exit #{number}: during event delivery: EXITINTINFO={info:#010x} error code={error_code:#x}"
            );
        }
    }

    /// The guest's state from the VMCB, and its general-purpose registers.
    fn dump(&self, number: u64, registers: &GuestRegisters) {
        let tag = crate::TAG;
        let vmcb = &self.state().vmcb;
        let save = &vmcb.save;
        let control = &vmcb.control;
        crate::serial_println!(
            "{tag} VM exit #{number}: guest rip={:#018x} rsp={:#018x} rflags={:#x} cpl={}",
            save.rip,
            save.rsp,
            save.rflags,
            save.cpl
        );
        crate::serial_println!(
            "{tag} VM exit #{number}: guest cr0={:#x} cr3={:#x} cr4={:#x} efer={:#x} cs={:#06x} ss={:#06x}",
            save.cr0,
            save.cr3,
            save.cr4,
            save.efer,
            save.cs.selector,
            save.ss.selector
        );
        crate::serial_println!(
            "{tag} VM exit #{number}: int_state={:#x} event_inj={:#x} exit_int_info={:#x}",
            control.int_state,
            control.event_inj,
            control.exit_int_info
        );
        let r = registers;
        crate::serial_println!(
            "{tag} VM exit #{number}: rax={:#x} rbx={:#x} rcx={:#x} rdx={:#x} rsi={:#x} rdi={:#x} rbp={:#x}",
            r.rax,
            r.rbx,
            r.rcx,
            r.rdx,
            r.rsi,
            r.rdi,
            r.rbp
        );
        crate::serial_println!(
            "{tag} VM exit #{number}: r8={:#x} r9={:#x} r10={:#x} r11={:#x} r12={:#x} r13={:#x} r14={:#x} r15={:#x}",
            r.r8,
            r.r9,
            r.r10,
            r.r11,
            r.r12,
            r.r13,
            r.r14,
            r.r15
        );
    }

    fn describe_reason(code: u32) -> ReasonDescription {
        ReasonDescription {
            number: if code == code::INVALID { u16::MAX } else { code as u16 },
            name: code_name(code),
            entry_failure: code == code::INVALID,
        }
    }

    fn encode_event(event: InjectedEvent) -> u32 {
        encode(event)
    }

    /// Bit 0 the interrupt shadow (`int_state`), bit 31 the NMI masking the
    /// run loop keeps.
    fn encode_interruptibility(state: Interruptibility) -> u32 {
        u32::from(state.sti_shadow || state.mov_ss_shadow) | (u32::from(state.nmi_blocked) << 31)
    }
}

#[cfg(test)]
#[path = "../../../tests/unit/hal/amd/vm_exit.rs"]
mod tests;
