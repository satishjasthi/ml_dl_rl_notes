#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PAPER_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${PAPER_DIR}"
uv sync --locked
