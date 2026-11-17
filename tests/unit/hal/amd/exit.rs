//! Unit tests for `src/hal/amd/exit.rs`.
//!
//! Compiled only by `cargo test-host`. The VMCB is plain memory, so these
//! decode exits and carry out the acts against a real [`SvmGuestRegisters`],
//! exactly as the run loop will.

use super::*;
use crate::hal::amd::registers::{GprFrame, SvmGuestRegisters};
use crate::hal::amd::vmcb::Vmcb;

/// A VMCB and frame with the guest registers set, and the register view over
/// them.
struct Guest {
    vmcb: Box<Vmcb>,
    gprs: GprFrame,
}

impl Guest {
    fn new() -> Self {
        Self {
            vmcb: Box::new(Vmcb::zeroed()),
            gprs: GprFrame::default(),
        }
    }

    fn regs(&mut self) -> SvmGuestRegisters<'_> {
        SvmGuestRegisters::new(&mut self.vmcb, &mut self.gprs)
    }
}

#[test]
fn a_cpuid_exit_reads_the_leaf_and_subleaf() {
    let mut guest = Guest::new();
    guest.vmcb.save.rax = 1; // leaf, in the VMCB
    guest.gprs.rcx = 0; // subleaf, in the frame
    let exit = decode(code::CPUID, 0, 0, &guest.regs());
    assert_eq!(exit, Exit::Cpuid { leaf: 1, subleaf: 0 });
}

#[test]
fn an_msr_exit_tells_read_from_write_by_exit_info_1() {
    let mut guest = Guest::new();
    guest.gprs.rcx = 0xC000_0080; // EFER, the MSR number in ECX
    assert_eq!(decode(code::MSR, 0, 0, &guest.regs()), Exit::Rdmsr { msr: 0xC000_0080 });

    // WRMSR: bit 0 of exit_info_1 set, value from EDX:EAX.
    guest.vmcb.save.rax = 0xAAAA_BBBB;
    guest.gprs.rdx = 0x1111_2222;
    assert_eq!(
        decode(code::MSR, 1, 0, &guest.regs()),
        Exit::Wrmsr {
            msr: 0xC000_0080,
            value: 0x1111_2222_AAAA_BBBB,
        }
    );
}

#[test]
fn an_ioio_exit_reads_the_port_direction_and_size() {
    // IN, 1 byte, port 0x3F8: TYPE_IN | SZ8 | (port << 16).
    let mut guest = Guest::new();
    let info_in = (1 << 0) | (1 << 4) | (0x3F8 << 16);
    assert_eq!(
        decode(code::IOIO, info_in, 0, &guest.regs()),
        Exit::In {
            port: 0x3F8,
            size: IoSize::Byte
        }
    );

    // OUT, 4 bytes, port 0x80, value from EAX.
    guest.vmcb.save.rax = 0xDEAD_BEEF;
    let info_out = (1 << 6) | (0x80 << 16);
    assert_eq!(
        decode(code::IOIO, info_out, 0, &guest.regs()),
        Exit::Out {
            port: 0x80,
            size: IoSize::Dword,
            value: 0xDEAD_BEEF,
        }
    );
}

#[test]
fn hlt_vmmcall_and_npf_decode() {
    let mut guest = Guest::new();
    assert_eq!(decode(code::HLT, 0, 0, &guest.regs()), Exit::Hlt);
    assert_eq!(decode(code::VMMCALL, 0, 0, &guest.regs()), Exit::Vmmcall);
    assert_eq!(
        decode(code::NPF, 0x6, 0x7EAD_5000, &guest.regs()),
        Exit::NestedPageFault {
            gpa: 0x7EAD_5000,
            error_code: 0x6,
        }
    );
}

#[test]
fn an_xsetbv_exit_reads_the_xcr_and_value() {
    let mut guest = Guest::new();
    guest.gprs.rcx = 0;
    guest.vmcb.save.rax = 0xFFFF_FFFF_0000_0007; // only EAX counts
    guest.gprs.rdx = 0;
    assert_eq!(
        decode(code::XSETBV, 0, 0, &guest.regs()),
        Exit::Xsetbv { xcr: 0, value: 0x7 }
    );
}

/// XSETBV with x87, SSE, AVX and AVX-512 supported.
const XSETBV_FEATURES: GuestFeatures = GuestFeatures { xcr0: 0xE7 };

#[test]
fn xsetbv_of_a_value_xcr0_takes_is_run() {
    for value in [0x1, 0x3, 0x7, 0xE7] {
        assert_eq!(
            xsetbv_action(0, value, XSETBV_FEATURES),
            XsetbvAction::Run { value },
            "{value:#x}"
        );
    }
}

#[test]
fn xsetbv_gets_gp_for_every_rule() {
    // MPX and AMX supported too, to reach their rules.
    let features = GuestFeatures { xcr0: 0x6_00FF };
    for (xcr, value, why) in [
        (1, 0x3, "not XCR0"),
        (0, 0x1_0003, "outside the supported bits"),
        (0, 0x2, "x87 clear"),
        (0, 0x5, "AVX without SSE"),
        (0, 0x67, "AVX-512 partly set"),
        (0, 0xE3, "AVX-512 without AVX"),
        (0, 0xB, "MPX partly set"),
        (0, 0x2_0003, "AMX partly set"),
    ] {
        assert_eq!(
            xsetbv_action(xcr, value, features),
            XsetbvAction::GeneralProtection,
            "{why}"
        );
    }
    // The pairs set together are fine.
    for value in [0x1B, 0x6_0003] {
        assert!(xcr0_valid(value, features.xcr0), "{value:#x}");
    }
}

#[test]
fn without_xsetbv_in_the_host_every_value_gets_gp() {
    for value in [0x1, 0x3] {
        assert_eq!(
            xsetbv_action(0, value, GuestFeatures::NONE),
            XsetbvAction::GeneralProtection
        );
    }
}

#[test]
fn the_injection_constants_match_intels_encoding() {
    // Same low 32 bits as Intel's InterruptionInfo constants.
    assert_eq!(EventInjection::NMI, EventInjection(0x8000_0202));
    assert_eq!(EventInjection::GENERAL_PROTECTION, EventInjection(0x8000_0B0D));
    assert_eq!(EventInjection::INVALID_OPCODE, EventInjection(0x8000_0306));
}

#[test]
fn only_gp_delivers_an_error_code() {
    assert!(EventInjection::GENERAL_PROTECTION.delivers_error_code());
    assert!(!EventInjection::INVALID_OPCODE.delivers_error_code());
    assert!(!EventInjection::NMI.delivers_error_code());
}

#[test]
fn vector_is_none_when_not_valid() {
    assert_eq!(EventInjection::GENERAL_PROTECTION.vector(), Some(13));
    assert_eq!(EventInjection(0x0B0D).vector(), None);
}

#[test]
fn queue_event_writes_eventinj_with_error_code_0() {
    let mut vmcb = Box::new(Vmcb::zeroed());
    queue_event(&mut vmcb.control, EventInjection::GENERAL_PROTECTION);
    let written = EventInjection(vmcb.control.event_inj);
    assert_eq!(written, EventInjection::GENERAL_PROTECTION);
    assert_eq!(written.error_code(), 0);

    queue_event(&mut vmcb.control, EventInjection::INVALID_OPCODE);
    assert_eq!(EventInjection(vmcb.control.event_inj), EventInjection::INVALID_OPCODE);
}

#[test]
fn an_unhandled_code_is_other() {
    let mut guest = Guest::new();
    assert_eq!(decode(0x400 + 1, 0, 0, &guest.regs()), Exit::Other { exit_code: 0x401 });
}

#[test]
fn cpuid_policy_hides_the_svm_bit() {
    let raw = CpuidResult {
        eax: 0,
        ebx: 0,
        ecx: 0xFFFF_FFFF,
        edx: 0,
    };
    let seen = cpuid_policy(0x8000_0001, 0, raw, 0, OFFERS_XSETBV);
    assert_eq!(seen.ecx & (1 << 2), 0, "SVM bit must be clear");
    // Every other bit is left as the processor reported it.
    assert_eq!(seen.ecx | (1 << 2), 0xFFFF_FFFF);
}

/// Features with XSETBV offered: nothing hidden.
const OFFERS_XSETBV: GuestFeatures = GuestFeatures { xcr0: 0xE7 };

#[test]
fn cpuid_policy_hides_xsave_when_bring_up_does_not_offer_it() {
    let raw = CpuidResult {
        eax: 0,
        ebx: 0,
        ecx: 1 << 26,
        edx: 0,
    };
    assert_eq!(cpuid_policy(1, 0, raw, 0, GuestFeatures::NONE).ecx & (1 << 26), 0);
    assert_ne!(cpuid_policy(1, 0, raw, 0, OFFERS_XSETBV).ecx & (1 << 26), 0);
}

#[test]
fn cpuid_policy_mirrors_the_guest_ospke() {
    let raw = CpuidResult {
        eax: 0,
        ebx: 0,
        ecx: 0,
        edx: 0,
    };
    // Guest CR4.PKE set -> guest sees OSPKE, whatever the host had.
    assert_ne!(cpuid_policy(7, 0, raw, 1 << 22, OFFERS_XSETBV).ecx & (1 << 4), 0);
    // Guest CR4.PKE clear -> hidden even if the host had it.
    let host_had_it = CpuidResult { ecx: 1 << 4, ..raw };
    assert_eq!(cpuid_policy(7, 0, host_had_it, 0, OFFERS_XSETBV).ecx & (1 << 4), 0);
    // Subleaf 1 of leaf 7 is not touched.
    assert_eq!(cpuid_policy(7, 1, host_had_it, 0, OFFERS_XSETBV), host_had_it);
}

#[test]
fn cpuid_policy_mirrors_the_guest_osxsave() {
    let raw = CpuidResult {
        eax: 0,
        ebx: 0,
        ecx: 0,
        edx: 0,
    };
    // Host CPUID has OSXSAVE clear; guest CR4.OSXSAVE set -> guest sees it.
    let with_osxsave = cpuid_policy(1, 0, raw, 1 << 18, OFFERS_XSETBV);
    assert_ne!(with_osxsave.ecx & (1 << 27), 0);
    // Guest CR4.OSXSAVE clear -> guest does not see it, even if host had it.
    let host_had_it = CpuidResult { ecx: 1 << 27, ..raw };
    let without = cpuid_policy(1, 0, host_had_it, 0, OFFERS_XSETBV);
    assert_eq!(without.ecx & (1 << 27), 0);
}

#[test]
fn apply_cpuid_writes_all_four_registers() {
    let mut guest = Guest::new();
    let result = CpuidResult {
        eax: 0x0009_0672,
        ebx: 0x0002_0800,
        ecx: 0xF7FA_3203,
        edx: 0x1F8B_FBFF,
    };
    apply_cpuid(&mut guest.regs(), result).unwrap();
    assert_eq!(guest.vmcb.save.rax, 0x0009_0672); // RAX in the VMCB
    assert_eq!(guest.gprs.rbx, 0x0002_0800); // RBX in the frame
    assert_eq!(guest.gprs.rcx, 0xF7FA_3203);
    assert_eq!(guest.gprs.rdx, 0x1F8B_FBFF);
}

#[test]
fn apply_rdmsr_splits_the_value_across_edx_eax() {
    let mut guest = Guest::new();
    apply_rdmsr(&mut guest.regs(), 0x1111_2222_3333_4444).unwrap();
    assert_eq!(guest.vmcb.save.rax, 0x3333_4444); // low 32 in EAX
    assert_eq!(guest.gprs.rdx, 0x1111_2222); // high 32 in EDX
}

#[test]
fn apply_in_keeps_the_upper_bits_for_a_narrow_read() {
    let mut guest = Guest::new();
    guest.vmcb.save.rax = 0xFFFF_FFFF_FFFF_FFFF;
    apply_in(&mut guest.regs(), IoSize::Byte, 0x42).unwrap();
    // Only AL changes.
    assert_eq!(guest.vmcb.save.rax, 0xFFFF_FFFF_FFFF_FF42);

    guest.vmcb.save.rax = 0xFFFF_FFFF_FFFF_FFFF;
    apply_in(&mut guest.regs(), IoSize::Dword, 0x1234_5678).unwrap();
    // A 32-bit read zeroes the upper 32 bits, as hardware does.
    assert_eq!(guest.vmcb.save.rax, 0x1234_5678);
}

#[test]
fn advance_rip_moves_past_the_instruction() {
    let mut guest = Guest::new();
    guest.vmcb.save.rip = 0x7EAD_4009;
    advance_rip(&mut guest.regs(), 0x7EAD_400B).unwrap();
    assert_eq!(guest.vmcb.save.rip, 0x7EAD_400B);
}

/// Hyper-V's MTRR MSRs: eight variable ranges, the fixed ones, MAXPHYADDR
/// 46, as Intel's emulation tests.
fn hyper_v_mtrrs() -> MtrrCapabilities {
    MtrrCapabilities::new((1 << 8) | 8, 46).unwrap()
}

/// IA32_KERNEL_GS_BASE, the one MSR Intel's test guest intercepts besides
/// the MTRRs.
const KERNEL_GS_BASE: u32 = 0xC000_0102;

#[test]
fn an_mtrr_goes_to_the_guests_copy() {
    for address in [0x2FF, 0x250, 0x26F, 0x200, 0x20F] {
        let msr = hyper_v_mtrrs().msr(address).unwrap();
        assert_eq!(
            rdmsr_action(address, &[], hyper_v_mtrrs(), false),
            MsrAction::MtrrRead { msr },
            "{address:#x}"
        );
    }
    // Variable range 7's mask: 2 MiB, valid.
    let msr = hyper_v_mtrrs().msr(0x20F).unwrap();
    assert_eq!(
        wrmsr_action(0x20F, 0x3FFF_FFE0_0800, &[], hyper_v_mtrrs(), false),
        MsrAction::MtrrWrite {
            msr,
            value: 0x3FFF_FFE0_0800
        }
    );
}

#[test]
fn an_mtrr_value_the_cpu_refuses_gets_gp() {
    for (address, value) in [(0x2FF, 0xD00), (0x20E, 0x20_0000 | 2), (0x20F, (1 << 46) | 0x800)] {
        assert_eq!(
            wrmsr_action(address, value, &[], hyper_v_mtrrs(), false),
            MsrAction::GeneralProtection,
            "{address:#x} <- {value:#x}"
        );
    }
}

#[test]
fn an_mtrr_address_the_cpu_lacks_gets_gp() {
    for (mtrr, msr) in [
        (hyper_v_mtrrs(), 0x210),
        (MtrrCapabilities::new(8, 46).unwrap(), 0x250),
        (MtrrCapabilities::NONE, 0x2FF),
    ] {
        assert_eq!(route_msr(msr, &[], mtrr, false), MsrRoute::AbsentMtrr, "{msr:#x}");
        assert_eq!(rdmsr_action(msr, &[], mtrr, false), MsrAction::GeneralProtection);
    }
}

#[test]
fn an_intercepted_msr_passes_through_with_a_canonical_value() {
    let intercepted = [KERNEL_GS_BASE];
    assert_eq!(
        rdmsr_action(KERNEL_GS_BASE, &intercepted, hyper_v_mtrrs(), false),
        MsrAction::Passthrough
    );
    assert_eq!(
        wrmsr_action(
            KERNEL_GS_BASE,
            0xFFFF_8000_0000_0000,
            &intercepted,
            hyper_v_mtrrs(),
            false
        ),
        MsrAction::Passthrough
    );
    assert_eq!(
        wrmsr_action(
            KERNEL_GS_BASE,
            0x8000_0000_0000_0000,
            &intercepted,
            hyper_v_mtrrs(),
            false
        ),
        MsrAction::GeneralProtection
    );
    // Not in the list: its bit is clear, so it should not have exited.
    assert_eq!(
        route_msr(KERNEL_GS_BASE, &[], hyper_v_mtrrs(), false),
        MsrRoute::NotIntercepted
    );
}

#[test]
fn msrs_outside_the_three_ranges_are_synthetic_under_hyper_v_or_gp() {
    assert_eq!(route_msr(0x4000_0020, &[], hyper_v_mtrrs(), true), MsrRoute::Synthetic);
    assert_eq!(
        route_msr(0x4000_0020, &[], hyper_v_mtrrs(), false),
        MsrRoute::OutOfRange
    );
    // 0xC0010000..=0xC0011FFF is the MSRPM's third range, which Intel's
    // bitmap does not have: in range here.
    assert_eq!(
        route_msr(0xC001_0114, &[], hyper_v_mtrrs(), false),
        MsrRoute::NotIntercepted
    );
    assert_eq!(
        rdmsr_action(0x4000_0020, &[], hyper_v_mtrrs(), false),
        MsrAction::GeneralProtection
    );
    assert_eq!(
        rdmsr_action(0x4000_0020, &[], hyper_v_mtrrs(), true),
        MsrAction::Synthetic
    );
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
