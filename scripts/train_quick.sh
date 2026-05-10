#!/usr/bin/env bash
set -euo pipefail
python -m deep_har.train --model cnn --epochs 2 --quick-run
