#!/usr/bin/env bash
# Runs the test suite under qemu-user for every CPU model the tests pin.
#
#   scripts/qemu-test.sh aarch64     # Neon/SVE models + SVE vector lengths
#   scripts/qemu-test.sh powerpc     # G4, BE POWER8, LE POWER8/9/10
#   scripts/qemu-test.sh             # both
#
# Needs (Debian/Ubuntu): qemu-user, gcc-aarch64-linux-gnu, gcc-powerpc-linux-gnu,
# gcc-powerpc64-linux-gnu, gcc-powerpc64le-linux-gnu, and the rustup targets.
#
# The model name reaches the tests in QEMU_CPU (qemu-user reads it and passes
# the environment through). `aarch64::tests::qemu_cpu_model_reports_the_expected_features`
# and `ops::sve::vl` tests pin what each aarch64 row must report;
# MIRACULIX_EXPECT_POWER does the same for the PowerPC rows. Without those
# guards a model that silently lacked a feature would pass by skipping.
set -euo pipefail
cd "$(dirname "$0")/.."

run() { # target env-prefix gnu-prefix qemu cpu [extra env...]
    local target=$1 env=$2 gnu=$3 qemu=$4 cpu=$5
    shift 5
    echo "=== $target  QEMU_CPU=$cpu $*"
    env "CARGO_TARGET_${env}_LINKER=${gnu}-gcc" \
        "CARGO_TARGET_${env}_RUNNER=${qemu} -L /usr/${gnu}" \
        "QEMU_CPU=${cpu}" "$@" \
        cargo test --quiet --target "$target"
}

aarch64() {
    local t=aarch64-unknown-linux-gnu e=AARCH64_UNKNOWN_LINUX_GNU g=aarch64-linux-gnu q=qemu-aarch64
    for cpu in cortex-a53 cortex-a72 cortex-a76 neoverse-n2 neoverse-v1 a64fx \
        max max,sve128=on max,sve256=on max,sve512=on; do
        run $t $e $g $q "$cpu"
    done
}

powerpc() {
    run powerpc-unknown-linux-gnu POWERPC_UNKNOWN_LINUX_GNU powerpc-linux-gnu qemu-ppc 7400 \
        MIRACULIX_EXPECT_POWER=vmx
    run powerpc64-unknown-linux-gnu POWERPC64_UNKNOWN_LINUX_GNU powerpc64-linux-gnu qemu-ppc64 power8 \
        MIRACULIX_EXPECT_POWER=power8
    for cpu in power8 power9 power10; do
        run powerpc64le-unknown-linux-gnu POWERPC64LE_UNKNOWN_LINUX_GNU powerpc64le-linux-gnu \
            qemu-ppc64le "$cpu" "MIRACULIX_EXPECT_POWER=$cpu"
    done
}

case "${1:-all}" in
    aarch64) aarch64 ;;
    powerpc) powerpc ;;
    all) aarch64; powerpc ;;
    *) echo "usage: $0 [aarch64|powerpc|all]" >&2; exit 2 ;;
esac
echo "all QEMU rows passed"
