//! Unit tests for `src/hal/amd/vm_exit.rs`: the decoders that turn the
//! VMCB's exit fields into the neutral types. Ported from the AMD exit
//! handler's decode and re-delivery tests (`exit.rs`, `events.rs`).
//!
//! Compiled only by `cargo test-host`.

use super::*;
use crate::hal::VmExit as _;
use crate::hal::amd::vmcb::Vmcb;

#[test]
fn every_handled_code_has_its_reason() {
    assert_eq!(decode_reason(code::CPUID, 0), ExitReason::Cpuid);
    assert_eq!(decode_reason(code::HLT, 0), ExitReason::Hlt);
    assert_eq!(decode_reason(code::VMMCALL, 0), ExitReason::Hypercall);
    assert_eq!(decode_reason(code::IOIO, 0), ExitReason::IoInstruction);
    assert_eq!(decode_reason(code::XSETBV, 0), ExitReason::Xsetbv);
    assert_eq!(decode_reason(code::MWAIT, 0), ExitReason::Mwait);
    assert_eq!(decode_reason(code::MONITOR, 0), ExitReason::Monitor);
    assert_eq!(decode_reason(code::NMI, 0), ExitReason::ExceptionOrNmi);
    assert_eq!(decode_reason(code::EXCEPTION + 14, 0), ExitReason::ExceptionOrNmi);
    assert_eq!(decode_reason(code::CR_WRITE + 4, 0), ExitReason::ControlRegisterAccess);
    assert_eq!(decode_reason(NMI_WINDOW, 0), ExitReason::NmiWindow);
    assert_eq!(decode_reason(INTERRUPT_WINDOW, 0), ExitReason::InterruptWindow);
}

#[test]
fn an_msr_exit_tells_read_from_write_by_exit_info_1() {
    assert_eq!(decode_reason(code::MSR, 0), ExitReason::Rdmsr);
    assert_eq!(decode_reason(code::MSR, 1), ExitReason::Wrmsr);
}

#[test]
fn a_nested_page_fault_with_rsvd_is_a_misconfiguration() {
    assert_eq!(decode_reason(code::NPF, 0x6), ExitReason::NestedPageFault);
    assert_eq!(decode_reason(code::NPF, 0x9), ExitReason::NestedPagingMisconfig);
}

#[test]
fn a_refused_vmrun_is_an_entry_failure_and_anything_else_other() {
    assert_eq!(decode_reason(code::INVALID, 0), ExitReason::EntryFailure(u32::MAX));
    assert_eq!(decode_reason(0x401, 0), ExitReason::Other(0x401));
    assert_eq!(decode_reason(code::SHUTDOWN, 0), ExitReason::Other(code::SHUTDOWN));
}

#[test]
fn an_ioio_exit_reads_the_port_direction_and_size() {
    // IN, 1 byte, port 0x3F8: TYPE_IN | SZ8 | (port << 16).
    let access = decode_io_access((1 << 0) | (1 << 4) | (0x3F8 << 16));
    assert_eq!(
        access,
        IoAccess {
            port: 0x3F8,
            size: Some(IoSize::Byte),
            is_in: true,
            is_string: false,
            is_rep: false,
        }
    );
    // OUTS with REP, 4 bytes, port 0x80.
    let access = decode_io_access((1 << 6) | (1 << 2) | (1 << 3) | (0x80 << 16));
    assert_eq!(access.port, 0x80);
    assert_eq!(access.size, Some(IoSize::Dword));
    assert!(!access.is_in && access.is_string && access.is_rep);
}

#[test]
fn an_exception_carries_its_error_code_only_when_it_pushes_one() {
    let page_fault = decode_exception_or_nmi(code::EXCEPTION + 14, 0x6).unwrap();
    assert_eq!(page_fault.kind, EventKind::HardwareException);
    assert_eq!(page_fault.vector, 14);
    assert_eq!(page_fault.error_code, Some(0x6));
    let invalid_opcode = decode_exception_or_nmi(code::EXCEPTION + 6, 0x6).unwrap();
    assert_eq!(invalid_opcode.error_code, None);
    assert_eq!(decode_exception_or_nmi(code::NMI, 0), Some(InjectedEvent::NMI));
    assert_eq!(decode_exception_or_nmi(code::CPUID, 0), None);
}

#[test]
fn no_valid_exitintinfo_means_no_redelivery() {
    assert_eq!(decode_interrupted(0x0B0E, 4, Some(2)), None);
}

#[test]
fn an_interrupted_nmi_is_re_delivered_as_an_nmi_of_either_type() {
    for info in [0x8000_0202, 0x8000_0302] {
        assert_eq!(decode_interrupted(info, 0, None), Some(InjectedEvent::NMI), "{info:#x}");
    }
}

#[test]
fn an_interrupted_page_fault_keeps_its_error_code() {
    // #PF: type 3, EV, vector 14, error code 0x6.
    let event = decode_interrupted(0x8000_0B0E, 6, None).unwrap();
    assert_eq!(event.kind, EventKind::HardwareException);
    assert_eq!(event.vector, 14);
    assert_eq!(event.error_code, Some(6));
    // Without EV the error code is dropped.
    let event = decode_interrupted(0x8000_0320, 6, None).unwrap();
    assert_eq!(event.error_code, None);
}

#[test]
fn a_software_interrupt_takes_the_instruction_length() {
    // INT 0x80: type 4, re-delivered with the length NRIP gives.
    let event = decode_interrupted(0x8000_0480, 0, Some(2)).unwrap();
    assert_eq!(event.kind, EventKind::SoftwareInterrupt);
    assert_eq!(event.instruction_len, Some(2));
    // INT3: type 6.
    let event = decode_interrupted(0x8000_0603, 0, Some(1)).unwrap();
    assert_eq!(event.kind, EventKind::SoftwareException);
    // A hardware exception needs none.
    assert_eq!(
        decode_interrupted(0x8000_030D, 0, Some(3)).unwrap().instruction_len,
        None
    );
}

#[test]
fn eventinj_round_trips_the_events_aleph0_injects() {
    assert_eq!(encode(InjectedEvent::NMI), 0x8000_0202);
    // #GP(0): type 3, EV, vector 13.
    assert_eq!(encode(InjectedEvent::GENERAL_PROTECTION), 0x8000_0B0D);
    // #UD: no error code.
    assert_eq!(encode(InjectedEvent::INVALID_OPCODE), 0x8000_0306);
    for event in [InjectedEvent::GENERAL_PROTECTION, InjectedEvent::INVALID_OPCODE] {
        let error_code = event.error_code.unwrap_or(0);
        assert_eq!(decode_interrupted(encode(event), error_code, None), Some(event));
    }
}

#[test]
fn a_write_to_a_present_page_is_the_protected_write() {
    // Present, write, final translation.
    let fault = decode_fault(0x1_0000_0003, 0x7EAD_5000);
    assert_eq!(fault.gpa, 0x7EAD_5000);
    assert!(fault.is_protected_write());
    // A write to a page not present is not.
    assert!(!decode_fault(0x1_0000_0002, 0x7EAD_5000).is_protected_write());
    // Nor one the guest's own page walk made.
    assert!(!decode_fault(0x2_0000_0003, 0x7EAD_5000).is_protected_write());
    // A fetch is not a read.
    let fetch = decode_fault(0x11, 0x1000);
    assert!(fetch.fetch && !fetch.read && !fetch.write);
}

#[test]
fn the_msrs_vmload_switches_are_read_and_written_in_the_vmcb() {
    let mut vmcb = Box::new(Vmcb::zeroed());
    *vmcb_held_msr(&mut vmcb.save, 0xC000_0102).unwrap() = 0x7A1E_0000_0000;
    assert_eq!(vmcb.save.kernel_gs_base, 0x7A1E_0000_0000);
    for (msr, expected) in [(IA32_STAR, 1), (IA32_LSTAR, 2), (IA32_CSTAR, 3), (IA32_SFMASK, 4)] {
        *vmcb_held_msr(&mut vmcb.save, msr).unwrap() = expected;
    }
    assert_eq!(
        (vmcb.save.star, vmcb.save.lstar, vmcb.save.cstar, vmcb.save.sfmask),
        (1, 2, 3, 4)
    );
    // IA32_TSC_AUX is not switched: its real register is the guest's.
    assert!(vmcb_held_msr(&mut vmcb.save, 0xC000_0103).is_none());
}

#[test]
fn every_code_up_to_npf_has_a_counter_and_a_refused_vmrun_the_other_one() {
    let npf = VmExit::describe_reason(code::NPF);
    assert!(usize::from(npf.number) < VmExit::REASON_COUNTERS);
    assert!(!npf.entry_failure);
    let invalid = VmExit::describe_reason(code::INVALID);
    assert!(usize::from(invalid.number) >= VmExit::REASON_COUNTERS);
    assert!(invalid.entry_failure);
    assert_eq!(VmExit::describe_reason(code::CPUID).name, "CPUID");
}
