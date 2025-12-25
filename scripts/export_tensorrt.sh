#!/usr/bin/env bash
set -euo pipefail
if ! command -v trtexec >/dev/null 2>&1; then
  echo "trtexec not found. Install TensorRT and ensure trtexec is in PATH."
  exit 1
fi
trtexec --onnx=artifacts/model.onnx --saveEngine=artifacts/model.plan --fp16
echo "Saved TensorRT engine to artifacts/model.plan"
