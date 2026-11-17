#!/usr/bin/env bash
# Runs the host unit tests of the AMD SVM backend (src/arch/amd), of the
# shared code it uses (src/common, src/hal), and then the whole suite, so a
# change to the shared code that breaks Intel shows up too. For Linux or WSL.
#
# The AMD backend's VMCB is plain memory, so most of it runs here for real:
# the VMCB layout, the intercepts and permission maps, the exit decoding,
# CPUID, XSETBV, the NPT and its walk, the guest's MTRRs, the NMI handling
# and the Hyper-V forwarder's recovery point. Only what needs an AMD CPU
# with SVM (VMRUN, the MSR writes that turn SVM on) waits for run-qemu.sh.
set -euo pipefail

root=$(cd "$(dirname "$0")/../.." && pwd)
cd "$root"
target=x86_64-unknown-linux-gnu

# run NAME FILTER: one group of tests, by module path.
run() {
    echo "==> $1"
    local output
    if output=$(cargo test --quiet --target "$target" -- "$2" 2>&1); then
        grep -E '^test result' <<<"$output"
    else
        grep -E '^test result|FAILED|panicked|^error' <<<"$output"
        return 1
    fi
}

failed=0
run "AMD backend (src/arch/amd)" "arch::amd::" || failed=1
run "shared code (src/common)" "common::" || failed=1
run "hardware abstraction layer (src/hal)" "hal::" || failed=1
run "everything, Intel included" "" || failed=1

if [ "$failed" -eq 0 ]; then
    echo "host tests: PASS"
else
    echo "host tests: FAIL"
    exit 1
fi
