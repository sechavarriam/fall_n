#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

export LSAN_OPTIONS="suppressions=${ROOT_DIR}/lsan.dev.supp"
export ASAN_OPTIONS="detect_leaks=1:leak_check_at_exit=1"

"${ROOT_DIR}/build/fall_n" "$@"
