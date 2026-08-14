#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PAPER_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${PAPER_DIR}"
uv run --locked python prototypes/toy_dinov2/run_toy_dinov2.py "$@"
