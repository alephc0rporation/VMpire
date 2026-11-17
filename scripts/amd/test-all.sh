#!/usr/bin/env bash
# Tests every AMD part: the host unit tests (test-host.sh), then two boots
# in QEMU on an AMD CPU with SVM (run-qemu.sh): the plain one, and the test
# guest, which runs every part under SVM. Options go to run-qemu.sh. Exits
# non-zero if any fails.
set -uo pipefail

here=$(cd "$(dirname "$0")" && pwd)

host=PASS
"$here/test-host.sh" || host=FAIL
qemu=PASS
"$here/run-qemu.sh" "$@" || qemu=FAIL
guest=PASS
"$here/run-qemu.sh" --test-guest "$@" || guest=FAIL

echo
echo "AMD tests:"
echo "  host unit tests   $host"
echo "  QEMU boot         $qemu"
echo "  QEMU test guest   $guest"
[ "$host" = PASS ] && [ "$qemu" = PASS ] && [ "$guest" = PASS ]
