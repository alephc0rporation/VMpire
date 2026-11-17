//! The AMD test guest's code: Intel's script (`hv::guest`) with VMMCALL
//! for VMCALL and the SVM CPUID bit for the VMX one. Its memory, the values
//! its script uses and the checks of its mappings are shared
//! (`hv::guest`).
//!
//! In order, the guest:
//! 1. runs CPUID 0x80000001, which must show SVM clear (`exit::cpuid_policy`);
//! 2. reads, writes, reads back and restores IA32_KERNEL_GS_BASE through
//!    Aleph0, which keeps the guest's value in the VMCB;
//! 3. reads, writes, reads back and restores the COM1 scratch register
//!    through Aleph0 (IOIO exits, real IN/OUT in the host);
//! 4. turns CR4.OSXSAVE on, checks CPUID.1:ECX.OSXSAVE follows it, writes
//!    XCR0 with XSETBV through Aleph0 and reads it back, restores it, and
//!    turns OSXSAVE off again;
//! 5. runs RDTSCP;
//! 6. moves its last variable MTRR to [`GUEST_MTRR_BASE`] and back: only
//!    Aleph0's copy of its MTRRs and the NPT memory types change;
//! 7. reads the Local APIC's version register, device memory the NPT maps
//!    UC, in xAPIC mode;
//! 8. runs HLT with IF = 0: only Aleph0 resuming past it goes on;
//! 9. reads, writes and reads back its data page, which the NPT maps
//!    read-only: the write is a nested page fault, after which Aleph0 makes
//!    the page writable and the write runs again;
//! 10. makes main VMMCALL #1 with [`GUEST_MARKER`] in EAX, the data page's
//!     seed in ECX and its new value in EDX;
//! 11. waits for an NMI, which its handler reports with a VMMCALL of
//!     [`GUEST_NMI_MARKER`]; the reinjected NMI's delivery pushes onto the
//!     read-only stack page, a nested page fault Aleph0 grants and
//!     re-delivers;
//! 12. makes main VMMCALL #2 with `GUEST_MARKER + 1`, which ends the run.
//!
//! A wrong result ends in `ud2` with [`GUEST_EMULATION_FAIL_MARKER`] in EAX.

use crate::hv::features::CPUID_1_ECX_OSXSAVE;
use crate::hv::guest::{
    GUEST_EMULATION_FAIL_MARKER, GUEST_EPT_VALUE, GUEST_IO_VALUE, GUEST_MARKER, GUEST_MSR_VALUE, GUEST_MTRR_BASE,
    GUEST_NMI_MARKER, GUEST_SCRATCH_PORT, GUEST_WAIT_ITERATIONS, GUEST_XCR0_VALUE, GuestMemory, GuestMemoryError,
};
use crate::x86::apic::{APIC_BASE_MODE, APIC_BASE_XAPIC, IA32_APIC_BASE, INTEGRATED_APIC_VERSIONS, LAPIC_VERSION};
use crate::x86::cpuid::ADDRESS_SIZES;
use crate::x86::mtrr::{self, IA32_MTRR_PHYSBASE0, IA32_MTRRCAP, PHYSMASK_VALID};
use crate::x86::page::PAGE_SIZE;
use crate::x86::page_table::SIZE_2M;

/// CPUID.80000001H:ECX bit 2: SVM, which the guest must not see.
const CPUID_EXT_ECX_SVM_BIT: u32 = 2;

core::arch::global_asm!(
    r#"
    .globl svm_guest_code_start
    .globl svm_guest_nmi_handler
    .globl svm_guest_code_end
svm_guest_code_start:
    // CPUID always exits; Aleph0 hides SVM (CPUID.80000001H:ECX bit 2), so
    // a guest never tries VMRUN.
    mov eax, 0x80000001
    xor ecx, ecx
    cpuid
    bt ecx, {svm_bit}
    jc svm_guest_emulation_failed
    // IA32_KERNEL_GS_BASE exits through the MSRPM; VMSAVE/VMLOAD keep the
    // guest's value in the VMCB, which Aleph0 reads and writes. Save it,
    // write GUEST_MSR_VALUE, read it back, restore it.
    mov ecx, {kernel_gs_base}
    rdmsr
    mov esi, eax
    mov edi, edx
    mov eax, {msr_low}
    mov edx, {msr_high}
    wrmsr
    xor eax, eax
    xor edx, edx
    rdmsr
    cmp eax, {msr_low}
    jne svm_guest_emulation_failed
    cmp edx, {msr_high}
    jne svm_guest_emulation_failed
    mov eax, esi
    mov edx, edi
    wrmsr
    // The COM1 scratch register exits through the IOPM; Aleph0 runs the
    // real IN/OUT. Save it, write GUEST_IO_VALUE, read it back, restore it.
    mov dx, {scratch_port}
    in al, dx
    mov sil, al
    mov al, {io_value}
    out dx, al
    xor eax, eax
    in al, dx
    cmp al, {io_value}
    jne svm_guest_emulation_failed
    mov al, sil
    out dx, al
    // XSETBV exits (its intercept is on); Aleph0 checks the value and runs
    // it in the host. Turn CR4.OSXSAVE on, which CPUID.1:ECX.OSXSAVE must
    // then show, write GUEST_XCR0_VALUE, read it back with XGETBV (which
    // does not exit), restore XCR0, and turn OSXSAVE off.
    mov rax, cr4
    bts rax, {cr4_osxsave_bit}
    mov cr4, rax
    mov eax, 1
    xor ecx, ecx
    cpuid
    bt ecx, {cpuid_osxsave_bit}
    jnc svm_guest_emulation_failed
    xor ecx, ecx
    xgetbv
    mov esi, eax
    mov edi, edx
    mov eax, {xcr0_value}
    xor edx, edx
    xsetbv
    xgetbv
    cmp eax, {xcr0_value}
    jne svm_guest_emulation_failed
    test edx, edx
    jnz svm_guest_emulation_failed
    mov eax, esi
    mov edx, edi
    xsetbv
    mov rax, cr4
    btr rax, {cr4_osxsave_bit}
    mov cr4, rax
    // RDTSCP runs without an exit: SVM has no control for it.
    rdtscp
    // The MTRRs exit through the MSRPM; Aleph0 keeps a copy for the guest
    // and never writes the CPU's. R10 = the address bits below MAXPHYADDR.
    mov eax, {address_sizes}
    cpuid
    mov ecx, eax
    mov r10d, 1
    shl r10, cl
    dec r10
    // IA32_MTRRCAP is not intercepted: AL is the number of variable ranges.
    // Take the last one, ECX = its PHYSBASE, and save its pair.
    mov ecx, {mtrrcap}
    rdmsr
    movzx ecx, al
    lea ecx, [rcx + rcx + {physbase0} - 2]
    rdmsr
    mov r8d, eax
    mov r9d, edx
    inc ecx
    rdmsr
    mov r11d, eax
    mov r12d, edx
    // Off, then GUEST_MTRR_BASE with type UC.
    xor eax, eax
    xor edx, edx
    wrmsr
    dec ecx
    mov eax, {mtrr_base_uc}
    wrmsr
    // On for 2 MiB: the NPT maps the region as one UC page.
    inc ecx
    mov r13, r10
    and r13, -{two_mib}
    or r13, {physmask_valid}
    mov eax, r13d
    mov rdx, r13
    shr rdx, 32
    wrmsr
    // The copy reads back what was written.
    xor eax, eax
    xor edx, edx
    rdmsr
    shl rdx, 32
    or rax, rdx
    cmp rax, r13
    jne svm_guest_emulation_failed
    // Only 4 KiB: the region's pages differ, so it takes a spare PT.
    mov r13, r10
    and r13, -{page_size}
    or r13, {physmask_valid}
    mov eax, r13d
    mov rdx, r13
    shr rdx, 32
    wrmsr
    // Put the pair back: off, base, mask.
    xor eax, eax
    xor edx, edx
    wrmsr
    dec ecx
    mov eax, r8d
    mov edx, r9d
    wrmsr
    inc ecx
    mov eax, r11d
    mov edx, r12d
    wrmsr
    // The Local APIC's version register, in xAPIC mode: device memory the
    // NPT maps UC, which must not exit.
    mov ecx, {apic_base_msr}
    rdmsr
    mov ecx, eax
    and ecx, {apic_mode_bits}
    cmp ecx, {xapic_mode}
    jne svm_guest_local_apic_done
    shl rdx, 32
    or rax, rdx
    and rax, r10
    and rax, -{page_size}
    mov eax, dword ptr [rax + {lapic_version}]
    cmp al, {apic_version_min}
    jb svm_guest_emulation_failed
    cmp al, {apic_version_max}
    ja svm_guest_emulation_failed
svm_guest_local_apic_done:
    // HLT exits, with IF = 0: only Aleph0 resuming past it goes on.
    hlt
    xor ebx, ebx
    // The data page follows the code page.
    lea rdi, [rip + svm_guest_code_start]
    add rdi, 0x1000
    mov ecx, dword ptr [rdi]
    // The NPT maps the page read-only: a nested page fault, after which
    // Aleph0 makes it writable and this write runs again.
    mov dword ptr [rdi], {ept_value}
    mov edx, dword ptr [rdi]
    mov eax, {marker}
    vmmcall
    // Wait for an NMI; the handler sets EBX. Give up after a while.
    mov ecx, {iterations}
svm_guest_wait:
    test ebx, ebx
    jnz svm_guest_wait_done
    pause
    dec ecx
    jnz svm_guest_wait
svm_guest_wait_done:
    inc eax
    vmmcall
    ud2
svm_guest_nmi_handler:
    push rax
    mov eax, {nmi_marker}
    vmmcall
    pop rax
    mov ebx, 1
    iretq
svm_guest_emulation_failed:
    mov eax, {fail_marker}
    ud2
svm_guest_code_end:
"#,
    svm_bit = const CPUID_EXT_ECX_SVM_BIT,
    kernel_gs_base = const crate::x86::msr::IA32_KERNEL_GS_BASE,
    msr_low = const GUEST_MSR_VALUE as u32,
    msr_high = const (GUEST_MSR_VALUE >> 32) as u32,
    marker = const GUEST_MARKER,
    iterations = const GUEST_WAIT_ITERATIONS,
    nmi_marker = const GUEST_NMI_MARKER,
    ept_value = const GUEST_EPT_VALUE,
    fail_marker = const GUEST_EMULATION_FAIL_MARKER,
    scratch_port = const GUEST_SCRATCH_PORT,
    io_value = const GUEST_IO_VALUE,
    cr4_osxsave_bit = const crate::x86::cr::CR4_OSXSAVE.trailing_zeros(),
    cpuid_osxsave_bit = const CPUID_1_ECX_OSXSAVE.trailing_zeros(),
    xcr0_value = const GUEST_XCR0_VALUE,
    address_sizes = const ADDRESS_SIZES,
    mtrrcap = const IA32_MTRRCAP,
    physbase0 = const IA32_MTRR_PHYSBASE0,
    mtrr_base_uc = const GUEST_MTRR_BASE | mtrr::MemoryType::Uncacheable as u64,
    two_mib = const SIZE_2M,
    page_size = const PAGE_SIZE,
    physmask_valid = const PHYSMASK_VALID,
    apic_base_msr = const IA32_APIC_BASE,
    apic_mode_bits = const APIC_BASE_MODE,
    xapic_mode = const APIC_BASE_XAPIC,
    lapic_version = const LAPIC_VERSION,
    apic_version_min = const *INTEGRATED_APIC_VERSIONS.start(),
    apic_version_max = const *INTEGRATED_APIC_VERSIONS.end(),
);

unsafe extern "C" {
    static svm_guest_code_start: u8;
    static svm_guest_nmi_handler: u8;
    static svm_guest_code_end: u8;
}

/// The guest's instruction bytes, as assembled into the image.
fn guest_code() -> &'static [u8] {
    let start = &raw const svm_guest_code_start;
    let end = &raw const svm_guest_code_end;
    // The labels come from the one asm block above, start before end.
    let len = end.addr() - start.addr();
    // SAFETY: the range is the assembled code: in the image for as long as
    // it runs, and never written.
    unsafe { core::slice::from_raw_parts(start, len) }
}

/// Offset of `svm_guest_nmi_handler` from the start of the guest code.
fn nmi_handler_offset() -> usize {
    let start = &raw const svm_guest_code_start;
    let handler = &raw const svm_guest_nmi_handler;
    // Both labels come from the one asm block above, start before handler.
    handler.addr() - start.addr()
}

/// The test guest's pages, with AMD's guest code copied in.
pub fn allocate() -> Result<GuestMemory, GuestMemoryError> {
    GuestMemory::allocate_with(guest_code(), nmi_handler_offset())
}

/// How many exits of each kind a finished run must have had: three CPUIDs
/// (0x80000001, leaf 1, the address sizes), two XSETBVs, two RDMSRs and two
/// WRMSRs of IA32_KERNEL_GS_BASE, three RDMSRs and seven WRMSRs of the last
/// variable MTRR, two INs and two OUTs of the scratch register, one HLT, and
/// nothing forwarded to Hyper-V. As Intel's `expected_stats`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct ExpectedExits {
    pub cpuid: u32,
    pub xsetbv: u32,
    pub msr_reads: u32,
    pub msr_writes: u32,
    pub mtrr_reads: u32,
    pub mtrr_writes: u32,
    pub io_in: u32,
    pub io_out: u32,
    pub hlt: u32,
}

/// What a finished run must have counted.
pub const EXPECTED_EXITS: ExpectedExits = ExpectedExits {
    cpuid: 3,
    xsetbv: 2,
    msr_reads: 2,
    msr_writes: 2,
    mtrr_reads: 3,
    mtrr_writes: 7,
    io_in: 2,
    io_out: 2,
    hlt: 1,
};

#[cfg(test)]
#[path = "../../../tests/unit/hal/amd/test_guest.rs"]
mod tests;
