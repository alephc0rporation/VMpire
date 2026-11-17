#!/usr/bin/env bash
# Builds Aleph0 and boots it in QEMU on an emulated AMD CPU with SVM, under
# OVMF, then checks Aleph0's serial output for the lines an AMD run must
# print. For Linux or WSL, with qemu-system-x86 and ovmf installed.
#
# Acceleration:
# - KVM on an AMD host (`--accel kvm`, the default when /dev/kvm exists and
#   the host CPU is AMD): `-cpu host` passes the real CPU's SVM through,
#   next-RIP save included, which Aleph0 requires. The host needs nested
#   SVM on (kvm_amd nested=1, the default on recent kernels).
# - TCG, software emulation (`--accel tcg`, the default elsewhere): runs on
#   any host, an Intel one too, but emulates SVM without next-RIP save, so
#   SVM bring-up stops there (NoNextRipSave). Up to that point, the vendor
#   check and the start of bring-up, it still shows Aleph0 on an AMD CPU.
#
# Serial: COM1 is the guest's (run/amd/guest-serial.log), COM2 Aleph0's own
# (run/amd/serial.log), as on Hyper-V. This OVMF also mirrors its console to
# COM2, so each line Aleph0 prints to the console and to COM2 lands there
# twice; the report shows Aleph0's lines once.
#
# Options:
#   --test-guest     build with the test-guest feature: Aleph0 runs its
#                    built-in test guest under SVM and checks every part
#   --accel kvm|tcg  how QEMU runs the CPU (default: see above)
#   --release        build the release profile
#   --timeout N      give up after N seconds (default 120)
#   --keep           leave QEMU running after the checks, until you stop it
set -euo pipefail

root=$(cd "$(dirname "$0")/../.." && pwd)
cd "$root"

profile=debug
timeout=120
keep=0
test_guest=0
accel=
while [ $# -gt 0 ]; do
    case "$1" in
        --release) profile=release ;;
        --test-guest) test_guest=1 ;;
        --accel) accel=$2; shift ;;
        --timeout) timeout=$2; shift ;;
        --keep) keep=1 ;;
        *) echo "unknown option: $1" >&2; exit 2 ;;
    esac
    shift
done

if [ -z "$accel" ]; then
    if [ -w /dev/kvm ] && grep -q AuthenticAMD /proc/cpuinfo; then accel=kvm; else accel=tcg; fi
fi
case "$accel" in
    kvm) cpu=host ;;
    tcg) cpu=max,svm=on,npt=on,vendor=AuthenticAMD ;;
    *) echo "--accel is kvm or tcg" >&2; exit 2 ;;
esac

ovmf_code=${OVMF_CODE:-/usr/share/OVMF/OVMF_CODE_4M.fd}
ovmf_vars_template=${OVMF_VARS:-/usr/share/OVMF/OVMF_VARS_4M.fd}
for file in "$ovmf_code" "$ovmf_vars_template"; do
    [ -f "$file" ] || { echo "missing $file: install ovmf, or set OVMF_CODE and OVMF_VARS" >&2; exit 1; }
done
command -v qemu-system-x86_64 >/dev/null || { echo "missing qemu-system-x86_64: install qemu-system-x86" >&2; exit 1; }

out=$root/run/amd
esp=$out/esp
serial=$out/serial.log
guest_serial=$out/guest-serial.log
mkdir -p "$esp/EFI/BOOT"
rm -f "$serial" "$guest_serial"

features=()
[ "$test_guest" -eq 1 ] && features=(--features test-guest)
echo "==> cargo build ($profile${features:+, ${features[*]}})"
if [ "$profile" = release ]; then cargo build --quiet --release "${features[@]}"; else cargo build --quiet "${features[@]}"; fi
cp "target/x86_64-unknown-uefi/$profile/aleph0hypervisor.efi" "$esp/EFI/BOOT/BOOTX64.EFI"
cp "$ovmf_vars_template" "$out/OVMF_VARS.fd"

echo "==> QEMU: AMD CPU with SVM ($accel, -cpu $cpu), OVMF, 2 GiB"
qemu-system-x86_64 \
    -machine q35,accel="$accel" \
    -cpu "$cpu" \
    -smp 1 -m 2048 \
    -drive if=pflash,format=raw,readonly=on,file="$ovmf_code" \
    -drive if=pflash,format=raw,file="$out/OVMF_VARS.fd" \
    -drive format=raw,file=fat:rw:"$esp" \
    -serial file:"$guest_serial" \
    -serial file:"$serial" \
    -display none -no-reboot &
qemu=$!
trap 'kill "$qemu" 2>/dev/null || true' EXIT

# The lines an AMD run prints, in order: every part's, for the test guest.
expected=(
    "Serial output enabled on COM2"
    "Image copied to resident memory"
    "AMD CPU: SVM backend."
)
if [ "$test_guest" -eq 1 ]; then
    expected+=(
        "SVM usable:"                       # part 1
        "NPT built:"                        # part 6
        "MSRPM at"                          # parts 2 and 7
        "Host environment:"                 # parts 10 and 11's host side
        "VMCB at"                           # parts 2 and 3
        "SVM on: EFER.SVME set"             # part 1
        "(code 0x72)"                       # part 5: CPUID
        "(code 0x7b)"                       # part 5: IN/OUT
        "(code 0x8d)"                       # part 5: XSETBV
        "(code 0x7c)"                       # part 7: the MTRRs
        "(code 0x78)"                       # part 5: HLT
        "Data page"                         # part 6: the protected page
        "Main VMMCALL #1 handled"           # part 9: VMMCALL
        "Main VMMCALL #2 handled"
        "Teardown: firmware CR3"
        "Guest finished after"
        "Returning to firmware with SUCCESS"
    )
fi
last=${expected[-1]}

echo "==> waiting up to ${timeout} s for: $last"
for _ in $(seq "$timeout"); do
    if grep -qF "$last" "$serial" 2>/dev/null || ! kill -0 "$qemu" 2>/dev/null; then
        break
    fi
    sleep 1
done
# A moment more, for any line after the last expected one.
sleep 2

echo "==> Aleph0's lines on serial (run/amd/serial.log):"
grep -aF "[Aleph0" "$serial" 2>/dev/null | tr -d '\r' | uniq | sed 's/^/    /' || echo "    (none)"

failed=0
for line in "${expected[@]}"; do
    if grep -qF "$line" "$serial" 2>/dev/null; then
        echo "  ok    $line"
    else
        echo "  MISSING  $line"
        failed=1
    fi
done

if [ "$keep" -eq 1 ]; then
    echo "==> QEMU left running (pid $qemu); stop it with: kill $qemu"
    trap - EXIT
fi

if [ "$failed" -eq 0 ]; then
    echo "QEMU run: PASS"
else
    echo "QEMU run: FAIL"
    exit 1
fi
