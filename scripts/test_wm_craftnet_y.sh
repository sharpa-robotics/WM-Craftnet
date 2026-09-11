#!/usr/bin/env bash
# Copyright (c) 2026 The WM-Craftnet Authors
# SPDX-License-Identifier: Apache-2.0
#
# Interactive test entry for the trained nine-object y-axis policy.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

export TEST_OBJ_SET="set_y"
exec bash "${SCRIPT_DIR}/test_wm_craftnet.sh" "$@"
