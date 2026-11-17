#!/usr/bin/env bash
# Runs the host unit tests of the AMD SVM backend (src/hal/amd), of the
# hardware abstraction layer (src/hal), and then the whole suite with each
# backend, so a change to the shared code that breaks Intel shows up too.
# For Linux or WSL.
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

# run NAME FILTER [FEATURES]: one group of tests, by module path, built
# with the AMD backend (`--features amd`) unless FEATURES says otherwise.
run() {
    echo "==> $1"
    local output
    if output=$(cargo test --quiet --target "$target" --features "${3:-amd}" -- "$2" 2>&1); then
        grep -E '^test result' <<<"$output"
    else
        grep -E '^test result|FAILED|panicked|^error' <<<"$output"
        return 1
    fi
}

failed=0
run "AMD backend (src/hal/amd)" "hal::amd::" || failed=1
run "hardware abstraction layer (src/hal)" "hal::" || failed=1
run "everything, with the AMD backend" "" || failed=1
run "everything, with the Intel backend" "" intel || failed=1

if [ "$failed" -eq 0 ]; then
    echo "host tests: PASS"
else
    echo "host tests: FAIL"
    exit 1
fi
