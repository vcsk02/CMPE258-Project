#!/usr/bin/env bash
set -euo pipefail
python -m deep_har.train --model cnn_lstm --epochs 40 --batch-size 64 --use-mlflow
