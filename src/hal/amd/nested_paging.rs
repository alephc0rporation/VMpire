//! `hal::NestedPaging` for SVM: the NPT identity map (`Npt`) and the tables
//! the exit handler changes (`TypeTables`), over the functions of `npt`.
//! The lines it prints are the ones the AMD bring-up printed, in the shape
//! of Intel's.

use core::convert::Infallible;
use core::fmt;

use super::npt::{
    self, FlushType, NPT_EXECUTE, NPT_READ, NPT_WRITE, Npt, NptError, NptSupport, ProtectedPage, TypeTables,
};
use super::vcpu;
use super::virtualization::flush_by_asid;
use crate::hal::ControlCapabilities;
use crate::hal::types::{Flush, Hidden, Permissions, Retyped, WalkEnd};
use crate::x86::identity_map::spare_pts;
use crate::x86::mtrr::{MemoryType, MtrrCapabilities, Mtrrs};
use crate::x86::page::Allocation;

/// The NPT identity map `Npt::build` made.
pub struct NestedPaging(Npt);

impl NestedPaging {
    /// The map, for the VMCB.
    pub(super) fn npt(&self) -> &Npt {
        &self.0
    }
}

fn flush(flush: FlushType) -> Flush {
    match flush {
        FlushType::Asid => Flush::SingleContext,
        FlushType::All => Flush::AllContexts,
    }
}

fn flush_type(flush: Flush) -> FlushType {
    match flush {
        Flush::SingleContext => FlushType::Asid,
        Flush::AllContexts => FlushType::All,
    }
}

/// The neutral walk end of an NPT walk. An entry with a reserved bit set and
/// one without U/S both make the CPU fault: both are entries it refuses. A
/// UC- page (a PAT entry Aleph0 never uses) is reported as UC.
fn walk_end(end: npt::walk::WalkEnd) -> WalkEnd {
    use npt::walk::WalkEnd as Npt;
    match end {
        Npt::Mapped {
            hpa,
            size,
            access,
            memory_type,
        } => WalkEnd::Mapped {
            hpa,
            size,
            permissions: Permissions {
                read: access & NPT_READ != 0,
                write: access & NPT_WRITE != 0,
                execute: access & NPT_EXECUTE != 0,
            },
            memory_type: memory_type.unwrap_or(MemoryType::Uncacheable),
        },
        Npt::NotPresent => WalkEnd::NotPresent,
        Npt::Reserved | Npt::Supervisor => WalkEnd::Misconfigured,
        Npt::OutsideTables { table } => WalkEnd::OutsideTables { table },
        Npt::BeyondWalk => WalkEnd::BeyondWalk,
    }
}

impl crate::hal::NestedPaging for NestedPaging {
    type Support = NptSupport;
    type Tables = TypeTables;
    type ProtectedPage = ProtectedPage;
    type Error = NptError;
    /// None: a flush on SVM is a VMCB write the next VMRUN acts on.
    type FlushError = Infallible;

    const NAME: &'static str = "NPT";
    const FLUSH: &'static str = "TLB_CONTROL";

    fn check_support(support: &ControlCapabilities, physical_address_bits: u8) -> Result<NptSupport, NptError> {
        // SAFETY: EFER exists on every 64-bit CPU; reading it has no side
        // effect.
        let efer = unsafe { crate::x86::msr::rdmsr(crate::x86::msr::IA32_EFER) };
        npt::check_support(
            support.nested_paging(),
            flush_by_asid(*support),
            physical_address_bits,
            crate::x86::cpuid::has_1g_pages(),
            efer & crate::x86::msr::EFER_NXE != 0,
        )
    }

    fn report_support(support: NptSupport) {
        log!(
            "NPT supported: 4-level walk, 2 MiB pages, flush {}; 1 GiB pages={}; NX={}; MAXPHYADDR {} bits.",
            support.flush.name(),
            support.rules.huge_pages,
            support.rules.nx,
            support.rules.physical_address_bits,
        );
    }

    fn physical_address_bits(support: NptSupport) -> u8 {
        support.rules.physical_address_bits
    }

    fn spare_tables(mtrr: MtrrCapabilities) -> usize {
        spare_pts(mtrr)
    }

    unsafe fn build(
        protected_pages: &[u64],
        support: NptSupport,
        mtrrs: &Mtrrs,
        spare_tables: usize,
    ) -> Result<(Self, TypeTables), NptError> {
        // SAFETY: the caller's obligation, as for `Npt::build`.
        let (npt, types) = unsafe { Npt::build(protected_pages, support, mtrrs, spare_tables) }?;
        Ok((Self(npt), types))
    }

    fn report(&self, types: &TypeTables, protected_pages: &[u64]) {
        let npt = &self.0;
        log!(
            "NPT built: N_CR3={:#018x}, map to {:#x} (PDs to {:#x}), {} split regions, {} spare PTs, {} table pages, flush {}.",
            npt.ncr3().raw(),
            npt.end(),
            npt.layout().pd_end(),
            npt.split_regions(),
            npt.spare_pts(),
            npt.table_pages(),
            types.flush().name(),
        );
        for page in protected_pages {
            log!("NPT read-only page: {page:#018x}.");
        }
        let ram = types.ram().ranges();
        log!(
            "NPT memory types: the MTRRs' for the {} RAM range(s) of the UEFI memory map, UC for everything else (MMIO, reserved memory, holes).",
            ram.len()
        );
        for &(start, end) in ram {
            log!("NPT RAM {start:#014x}..{end:#014x}");
        }
        for run in types.region_runs(0, npt.end()) {
            log!("NPT {:#014x}..{:#014x}: {}", run.start, run.end, run.mapping);
        }
    }

    fn memory_map_end(&self) -> u64 {
        self.0.memory_map_end()
    }

    unsafe fn hide(
        types: &mut TypeTables,
        allocations: impl Iterator<Item = Allocation>,
        mtrrs: &Mtrrs,
        visible: &[Allocation],
    ) -> Result<Hidden, NptError> {
        // SAFETY: the caller's obligation, as for `TypeTables::hide`.
        let hidden = unsafe { types.hide(allocations, mtrrs, visible) }?;
        Ok(Hidden {
            data_pages: hidden.data_pages,
            code_pages: hidden.code_pages,
            regions_split: hidden.regions_split,
        })
    }

    fn report_hidden(hidden: Hidden) {
        log!(
            "Aleph0 pages hidden from the guest: {} data page(s) redirected to the NPT shadow, \
             {} code page(s) read-only, {} region(s) split into a PT.",
            hidden.data_pages,
            hidden.code_pages,
            hidden.regions_split,
        );
    }

    /// `TypeTables::apply`, with the flush in the current VMCB's TLB_CONTROL.
    unsafe fn apply_memory_types(types: &mut TypeTables, mtrrs: &Mtrrs) -> Result<Retyped, Infallible> {
        // SAFETY: the caller guarantees exit context, so a vCPU is current
        // and the guest stopped.
        let control = unsafe { vcpu::current_control() };
        // SAFETY: the caller's obligation, as for `TypeTables::apply`.
        let retyped = unsafe { types.apply(mtrrs, control) };
        Ok(Retyped {
            regions: retyped.regions,
            span: retyped.span,
            split: retyped.split,
            merged: retyped.merged,
            uncached: retyped.uncached,
        })
    }

    fn flush_type(types: &TypeTables) -> Flush {
        flush(types.flush())
    }

    fn flush_name(flush: Flush) -> &'static str {
        flush_type(flush).name()
    }

    fn free_tables(types: &TypeTables) -> usize {
        types.free_pts()
    }

    fn walk(types: &TypeTables, gpa: u64) -> WalkEnd {
        walk_end(types.walk(types.ncr3(), gpa).end)
    }

    /// The NPT walk's own words for how it ended.
    fn fmt_walk_end(end: WalkEnd, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match end {
            WalkEnd::Mapped {
                hpa,
                size,
                permissions,
                memory_type,
            } => fmt::Display::fmt(
                &npt::walk::WalkEnd::Mapped {
                    hpa,
                    size,
                    access: if permissions.read { NPT_READ } else { 0 }
                        | if permissions.write { NPT_WRITE } else { 0 }
                        | if permissions.execute { NPT_EXECUTE } else { 0 },
                    memory_type: Some(memory_type),
                },
                f,
            ),
            WalkEnd::NotPresent => fmt::Display::fmt(&npt::walk::WalkEnd::NotPresent, f),
            WalkEnd::Misconfigured => {
                f.write_str("the last entry is one the CPU refuses: a reserved bit set, or no U/S")
            }
            WalkEnd::OutsideTables { table } => fmt::Display::fmt(&npt::walk::WalkEnd::OutsideTables { table }, f),
            WalkEnd::BeyondWalk => fmt::Display::fmt(&npt::walk::WalkEnd::BeyondWalk, f),
        }
    }

    /// The walk of `gpa` through the tables, each entry on its path, what
    /// the CPU refuses in it, and how the walk ended, as Intel's EPT walk.
    unsafe fn report_walk(types: Option<&TypeTables>, number: u64, gpa: u64) {
        let tag = crate::TAG;
        let Some(types) = types else {
            crate::serial_println!("{tag} VM exit #{number}: the NPT tables are not available; no NPT walk.");
            return;
        };
        let ncr3 = types.ncr3();
        let walk = types.walk(ncr3, gpa);
        crate::serial_println!(
            "{tag} VM exit #{number}: NPT walk of GPA {gpa:#018x} from N_CR3 {:#x}:",
            ncr3.raw()
        );
        for entry in walk.entries() {
            crate::serial_println!(
                "{tag} VM exit #{number}:   {} entry {:#x} at {:#x} = {:#018x}: {}",
                entry.level.name(),
                entry.index,
                entry.address,
                entry.raw,
                entry.faults,
            );
        }
        crate::serial_println!("{tag} VM exit #{number}:   {}.", walk.end);
    }

    fn report_regions(types: &TypeTables, number: u64, start: u64, end: u64) {
        let tag = crate::TAG;
        for run in types.region_runs(start, end) {
            crate::serial_println!(
                "{tag} VM exit #{number}: NPT {:#014x}..{:#014x}: {}",
                run.start,
                run.end,
                run.mapping
            );
        }
    }

    #[cfg(feature = "ept-misconfig-test")]
    unsafe fn misconfigure(types: &TypeTables, gpa: u64) {
        // SAFETY: the caller's obligation, as for `TypeTables::misconfigure`.
        unsafe { types.misconfigure(gpa) }
    }

    fn protected_page_for(gpa: u64) -> Option<ProtectedPage> {
        npt::protected_page_for(gpa)
    }

    fn protected_page_gpa(page: ProtectedPage) -> u64 {
        page.gpa
    }

    fn nmi_stack_page() -> u64 {
        npt::nmi_stack_page()
    }

    /// `ProtectedPage::grant_write`, with the flush in the current VMCB's
    /// TLB_CONTROL.
    unsafe fn grant_write(page: ProtectedPage) -> Result<Flush, Infallible> {
        // SAFETY: the caller guarantees exit context, so a vCPU is current
        // and the guest stopped.
        let control = unsafe { vcpu::current_control() };
        // SAFETY: the caller's obligation, as for `ProtectedPage::grant_write`.
        Ok(flush(unsafe { page.grant_write(control) }))
    }

    fn report_error(error: NptError) {
        match error {
            NptError::RangeTooLarge { end, limit } => log!(
                "NPT setup failed: the memory map reaches {end:#x}, past {limit:#x}, the end of the guest-physical address space."
            ),
            error => log!("NPT setup failed: {error:?}"),
        }
    }
}
