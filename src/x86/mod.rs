//! Plain x86-64: what any hypervisor on an x86 CPU needs, Intel or AMD.
//! Pages and page tables, MTRRs, MSRs, control registers, CPUID, segments
//! and descriptor tables, the Local APIC.

pub mod apic;
pub mod context_tables;
pub mod cpuid;
pub mod cr;
pub mod identity_map;
pub mod msr;
pub mod mtrr;
pub mod page;
pub mod page_table;
pub mod paging;
pub mod segment;
