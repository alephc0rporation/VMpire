//! Part 12's subject: AMD bring-up, the sequence that puts parts 1 to 11
//! together. The AMD counterpart of `bring_up` (Intel's).
//!
//! 1. Check SVM ([`svm::read_support`]): usable, with nested paging and
//!    next-RIP save.
//! 2. Allocate the test guest's pages and check how the page tables map
//!    them.
//! 3. Read the MTRRs, build the NPT identity map with their types and the
//!    guest's data and stack pages read-only ([`Npt::build`]), and allocate
//!    the MSR and I/O permission maps ([`ExitMaps`]). Hand the guest's copy
//!    of its MTRRs to the exit handler.
//! 4. Detect Hyper-V's interface, read the features the guest is offered
//!    ([`GuestFeatures`]), and build what Aleph0 runs on at an exit
//!    ([`HostEnvironment`]) with AMD's handler hooks.
//! 5. Fill the VMCB: intercepts, ASID, nested paging, the maps, and the
//!    guest state from the current CPU with the test guest's RIP, RSP and
//!    IDT.
//! 6. Turn SVM on ([`svm::enable`]), switch to the host environment and to
//!    [`npt::HOST_PAT`], and run the guest ([`handler::run`]) until it stops.
//! 7. Switch back to the firmware's tables, PAT and CR4, and turn SVM off.
//!
//! The firmware run, where the firmware itself becomes the guest, is the
//! next stage; without the `test-guest` feature this returns
//! [`SvmBringUpError::FirmwareGuestNotBuilt`].

use core::arch::asm;

use super::control::{ControlArea, NESTED_CTL_NP_ENABLE, intercept3, intercept4, tlb_control};
use super::exit_maps::{ExitMaps, ExitMapsError};
use super::features::{self, GuestFeatures};
use super::guest::segment_from_descriptor;
use super::handler::{self, GuestStop, RunStats};
use super::npt::{self, Npt, NptError};
use super::run::Vcpu;
use super::svm::{self, SvmSupport, SvmUnusable};
use super::test_guest;
use super::vmcb::{SaveArea, Segment};
use crate::host::{FirmwareSelectors, HostEnvironment, HostError, HostHooks};
use crate::hv::guest::{GUEST_SCRATCH_PORT, GuestMemoryError};
use crate::x86::cpuid;
use crate::x86::cr;
use crate::x86::identity_map::spare_pts;
use crate::x86::msr::{self, EFER_NXE, IA32_EFER, IA32_PAT};
use crate::x86::mtrr::{self, MtrrError, MtrrState};
use crate::x86::page::{PAGE_SIZE, allocate_zeroed_page, allocate_zeroed_pages};
use crate::x86::segment::{self, DescriptorTable};

/// The ASID the guest runs with. 0 is the host's; any other is a guest's.
const GUEST_ASID: u32 = 1;
/// RFLAGS with only the reserved bit 1 set: IF = 0, as the test guest runs.
const RFLAGS_RESERVED: u64 = 0x2;
/// DR7 and DR6 at reset.
const DR7_RESET: u64 = 0x400;
const DR6_RESET: u64 = 0xFFFF_0FF0;
/// EFER bit 12 (SVME): VMRUN refuses a guest without it.
const EFER_SVME: u64 = svm::EFER_SVME;

// The syscall MSRs VMLOAD loads from the guest VMCB, beside FS, GS, TR and
// LDTR: the guest starts with the firmware's.
use super::exit::{IA32_CSTAR, IA32_LSTAR, IA32_SFMASK, IA32_STAR};

/// AMD's hooks for the shared host handlers (`host::handlers`):
/// Hyper-V refusals of a forwarded VMMCALL or MSR access are recovered; an
/// NMI needs nothing more than being marked pending, since the run loop
/// looks at it with GIF clear before each VMRUN.
pub static HOOKS: HostHooks = HostHooks {
    recovery: super::hyperv::recovery,
    on_nmi: no_nmi_action,
    context: "the SVM host",
};

fn no_nmi_action() {}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SvmBringUpError {
    /// SVM cannot be used.
    Svm(SvmUnusable),
    /// No nested paging (CPUID 0x8000000A:EDX.NP).
    NoNestedPaging,
    /// No next-RIP save (CPUID 0x8000000A:EDX.NRIPS): Aleph0 moves RIP past
    /// an instruction with the VMCB's NRIP, as Intel does with the exit's
    /// instruction length.
    NoNextRipSave,
    GuestMemory(GuestMemoryError),
    Mtrr(MtrrError),
    Npt(NptError),
    ExitMaps(ExitMapsError),
    Host(HostError),
    /// UEFI could not provide the VMCB or host pages.
    AllocationFailed,
    /// How the guest stopped, when it did not finish.
    Stopped(GuestStop),
    /// The guest finished, but with other exits than its script makes.
    UnexpectedExits(RunStats),
    /// The firmware run is not built yet.
    FirmwareGuestNotBuilt,
}

macro_rules! from_error {
    ($($from:ty => $variant:ident),* $(,)?) => {
        $(impl From<$from> for SvmBringUpError {
            fn from(error: $from) -> Self {
                Self::$variant(error)
            }
        })*
    };
}
from_error!(
    SvmUnusable => Svm,
    GuestMemoryError => GuestMemory,
    MtrrError => Mtrr,
    NptError => Npt,
    ExitMapsError => ExitMaps,
    HostError => Host,
);

/// How bring-up ended when everything went to plan.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SvmGuestRun {
    /// The test guest finished after `exits` exits, all of them handled, and
    /// SVM is off again.
    Finished { exits: u64 },
}

/// What the firmware ran with before the switch, put back by [`teardown`].
struct FirmwareState {
    cr3: u64,
    cr4: u64,
    pat: u64,
    gdtr: DescriptorTable,
    idtr: DescriptorTable,
    interrupts_enabled: bool,
}

/// Runs the AMD test guest: steps 1 to 7 of the module docs.
pub fn run() -> Result<SvmGuestRun, SvmBringUpError> {
    if !cfg!(feature = "test-guest") {
        return Err(SvmBringUpError::FirmwareGuestNotBuilt);
    }

    // 1. SVM.
    // SAFETY: the vendor check that brought us here found an AMD CPU, whose
    // VM_CR read_support needs only with the SVM CPUID bit set.
    let support = unsafe { svm::read_support() };
    support.usable()?;
    if !support.nested_paging() {
        return Err(SvmBringUpError::NoNestedPaging);
    }
    if !support.next_rip_save() {
        return Err(SvmBringUpError::NoNextRipSave);
    }
    log!(
        "SVM usable: features={:#x} ASIDs={} nested paging, next-RIP save{}.",
        support.features,
        support.asids,
        if flush_by_asid(support) { ", flush by ASID" } else { "" }
    );

    // 2. The test guest's pages.
    let guest = test_guest::allocate()?;
    // SAFETY: boot services are still active.
    unsafe { guest.check_mappings() }?;
    log!(
        "Guest memory allocated: code={:#018x} ({} bytes) data={:#018x} stack top={:#018x} NMI handler={:#018x}.",
        guest.code_base(),
        guest.code_len(),
        guest.data_page(),
        guest.stack_top(),
        guest.nmi_handler(),
    );

    // 3. MTRRs, NPT, maps.
    let mtrrs = MtrrState::new(mtrr::read()?)?;
    let capabilities = mtrrs.registers().capabilities();
    // SAFETY: EFER exists on every 64-bit CPU.
    let efer = unsafe { msr::rdmsr(IA32_EFER) };
    let npt_support = npt::check_support(
        support.nested_paging(),
        flush_by_asid(support),
        cpuid::physical_address_bits(),
        cpuid::has_1g_pages(),
        efer & EFER_NXE != 0,
    )?;
    let protected_pages = [guest.data_page(), guest.stack_page()];
    // SAFETY: boot services are still active, and UEFI identity-maps memory.
    let (npt, types) = unsafe { Npt::build(&protected_pages, npt_support, mtrrs.mtrrs(), spare_pts(capabilities)) }?;
    log!(
        "NPT built: N_CR3={:#018x}, map to {:#x} (PDs to {:#x}), {} split regions, {} spare PTs, {} table pages, flush {}.",
        npt.ncr3().raw(),
        npt.end(),
        npt.layout().pd_end(),
        npt.split_regions(),
        npt.spare_pts(),
        npt.table_pages(),
        npt_support.flush.name(),
    );
    let maps = ExitMaps::allocate(handler::INTERCEPTED_MSRS, &[GUEST_SCRATCH_PORT])?;
    log!(
        "MSRPM at {:#018x} and IOPM at {:#018x}: every MTRR address, IA32_KERNEL_GS_BASE (through the VMCB), and the guest's scratch port {GUEST_SCRATCH_PORT:#x}.",
        maps.msrpm_address(),
        maps.iopm_address(),
    );
    // The test guest runs from Aleph0's own pages, so nothing is hidden, as
    // on Intel.
    super::guest_mtrrs::publish(mtrrs, types);

    // 4. Hyper-V, features, host environment.
    let hyperv = crate::hv::hyperv::Detection::read();
    crate::hv::hyperv::publish(hyperv.offers_interface());
    let guest_features = GuestFeatures::read();
    features::publish(guest_features);
    // SAFETY: boot services are still active, and UEFI identity-maps memory.
    let host = unsafe { HostEnvironment::build(npt.memory_map_end(), FirmwareSelectors::read()) }?;
    crate::host::install(&HOOKS);
    log!(
        "Host environment: CR3={:#018x} GDT={:#018x} TSS={:#018x} IDT={:#018x}.",
        host.page_tables().cr3(),
        host.gdt_base(),
        host.tss_base(),
        host.idt_base(),
    );

    // 5. VMCB and host pages.
    let hsave = allocate_zeroed_page(SvmBringUpError::AllocationFailed)?.as_ptr() as u64;
    let host_state = allocate_zeroed_page(SvmBringUpError::AllocationFailed)?.as_ptr() as u64;
    let vcpu_pages = size_of::<Vcpu>().div_ceil(PAGE_SIZE);
    let vcpu = allocate_zeroed_pages(vcpu_pages, SvmBringUpError::AllocationFailed)?.cast::<Vcpu>();
    // SAFETY: a fresh zeroed, page-aligned allocation as long as a `Vcpu`,
    // which nothing else refers to; all-zero is a valid `Vcpu`. Never freed.
    let vcpu = unsafe { &mut *vcpu.as_ptr() };
    // SAFETY: PAT exists on every 64-bit CPU.
    let firmware_pat = unsafe { msr::rdmsr(IA32_PAT) };
    fill_control(&mut vcpu.vmcb.control, &npt, &maps);
    // SAFETY: the current GDT is live and readable through its limit.
    vcpu.vmcb.save = unsafe { current_guest_state(efer | EFER_SVME, firmware_pat) };
    vcpu.vmcb.save.rip = guest.code_base();
    vcpu.vmcb.save.rsp = guest.stack_top();
    vcpu.vmcb.save.rflags = RFLAGS_RESERVED;
    let idt = guest.idt();
    vcpu.vmcb.save.idtr = Segment {
        selector: 0,
        attrib: 0,
        limit: u32::from(idt.limit),
        base: idt.base,
    };
    log!(
        "VMCB at {:#018x}: guest RIP={:#018x} RSP={:#018x} CR3={:#018x} CR4={:#x} EFER={:#x} CS={:#06x} (attrib {:#05x}).",
        &raw const vcpu.vmcb as u64,
        vcpu.vmcb.save.rip,
        vcpu.vmcb.save.rsp,
        vcpu.vmcb.save.cr3,
        vcpu.vmcb.save.cr4,
        vcpu.vmcb.save.efer,
        vcpu.vmcb.save.cs.selector,
        vcpu.vmcb.save.cs.attrib,
    );

    // 6. SVM on, the host environment, and the run.
    let firmware = FirmwareState {
        cr3: cr::read_cr3(),
        cr4: cr::read_cr4(),
        pat: firmware_pat,
        gdtr: segment::read_gdtr(),
        idtr: segment::read_idtr(),
        interrupts_enabled: interrupts_enabled(),
    };
    // SAFETY: `usable` returned Ok; `hsave` is a fresh zeroed page, never
    // freed.
    unsafe { svm::enable(hsave) };
    log!("SVM on: EFER.SVME set, VM_HSAVE_PA={hsave:#018x}. Running the test guest; its exits are on serial only.");
    // SAFETY: the host environment's tables are complete and never freed;
    // its GDT keeps the current selectors, so the switch needs no segment
    // reload; interrupts go off first.
    unsafe { switch_to_host(&host, guest_features.host_cr4(firmware.cr4)) };

    let stop = handler::run(vcpu, host_state);

    // 7. Back to the firmware's state.
    // SAFETY: the guest has stopped; `firmware` was read from the live
    // state before the switch.
    unsafe { teardown(&firmware, efer) };
    log!(
        "Teardown: firmware CR3, GDTR, IDTR, PAT and CR4 back, SVM off, IF={}.",
        u8::from(firmware.interrupts_enabled)
    );

    match stop {
        GuestStop::Finished { exits, stats } => {
            if stats.matches(test_guest::EXPECTED_EXITS) {
                Ok(SvmGuestRun::Finished { exits })
            } else {
                Err(SvmBringUpError::UnexpectedExits(stats))
            }
        }
        other => Err(SvmBringUpError::Stopped(other)),
    }
}

/// Whether the CPU can flush one ASID's translations
/// (CPUID 0x8000000A:EDX bit 6).
fn flush_by_asid(support: SvmSupport) -> bool {
    support.features & svm::SVM_FEATURE_FLUSH_BY_ASID != 0
}

/// The control area for the test guest: the intercepts, the ASID, nested
/// paging and the two permission maps.
fn fill_control(control: &mut ControlArea, npt: &Npt, maps: &ExitMaps) {
    control.intercept_word3 = intercept3::NMI
        | intercept3::SHUTDOWN
        | intercept3::CPUID
        | intercept3::HLT
        | intercept3::IOIO_PROT
        | intercept3::MSR_PROT;
    // VMRUN must be intercepted, or VMRUN refuses the guest (APM 15.5.1).
    control.intercept_word4 = intercept4::VMRUN | intercept4::VMMCALL | intercept4::XSETBV;
    // As Intel's test guest: every exception exits, so a fault in the
    // guest's script stops it with its vector instead of running its IDT.
    control.intercept_exceptions = u32::MAX;
    control.asid = GUEST_ASID;
    control.tlb_control = tlb_control::FLUSH_ALL;
    control.nested_ctl = NESTED_CTL_NP_ENABLE;
    control.nested_cr3 = npt.ncr3().raw();
    maps.write_to(control);
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

/// Whether RFLAGS.IF is set.
fn interrupts_enabled() -> bool {
    let rflags: u64;
    // SAFETY: reads RFLAGS through the stack; no other effect.
    unsafe { asm!("pushfq", "pop {}", out(reg) rflags, options(nomem, preserves_flags)) };
    rflags & (1 << 9) != 0
}

/// Switches the host to Aleph0's own environment, the AMD counterpart of a
/// VM exit loading the VMCS host fields: interrupts off, `host_cr4`
/// (OSXSAVE added when Aleph0 runs XSETBV), [`npt::HOST_PAT`] for the
/// nested walks, then its CR3, GDT, TR and IDT. VMRUN saves this state and
/// every #VMEXIT restores it, so every exit runs on it.
///
/// # Safety
///
/// `host`'s tables must be complete and never freed, and its GDT must keep
/// the current selectors ([`HostEnvironment::build`] with the firmware's).
unsafe fn switch_to_host(host: &HostEnvironment, host_cr4: u64) {
    // SAFETY: the caller's guarantees. The TR selector names the host GDT's
    // available TSS descriptor, which LTR marks busy.
    unsafe {
        asm!("cli", options(nomem, nostack));
        cr::write_cr4(host_cr4);
        msr::wrmsr(IA32_PAT, npt::HOST_PAT);
        cr::write_cr3(host.page_tables().cr3());
        segment::load_gdtr(&DescriptorTable {
            base: host.gdt_base(),
            limit: host.gdt_limit(),
        });
        asm!("ltr {0:x}", in(reg) host.selectors().tr, options(nostack, preserves_flags));
        segment::load_idtr(&DescriptorTable {
            base: host.idt_base(),
            limit: (crate::host::VECTORS * crate::x86::segment::GATE_SIZE - 1) as u16,
        });
    }
}

/// Puts the firmware's state back after the guest stopped: its IDT, GDT
/// and CR3, its PAT and CR4, EFER without SVME (`efer` as it was before
/// SVM), then interrupts if it had them. TR stays the host TSS, as on Intel
/// it stays the context TSS: firmware never switches stacks through it.
///
/// # Safety
///
/// The guest must have stopped, and `firmware` must be the live state read
/// before [`switch_to_host`].
unsafe fn teardown(firmware: &FirmwareState, efer: u64) {
    // SAFETY: the caller's guarantees; the firmware's tables are still live
    // during boot services.
    unsafe {
        segment::load_idtr(&firmware.idtr);
        segment::load_gdtr(&firmware.gdtr);
        cr::write_cr3(firmware.cr3);
        msr::wrmsr(IA32_PAT, firmware.pat);
        cr::write_cr4(firmware.cr4);
        msr::wrmsr(IA32_EFER, efer & !EFER_SVME);
        if firmware.interrupts_enabled {
            asm!("sti", options(nomem, nostack));
        }
    }
}
