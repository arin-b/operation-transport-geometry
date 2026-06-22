#!/usr/bin/env bash
set -euo pipefail
python -m otg.cli validate --preset fast --out runs/validation "$@"
