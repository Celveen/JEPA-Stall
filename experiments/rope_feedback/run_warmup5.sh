#!/usr/bin/env bash
set -euo pipefail

rope_repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
rope_local="$rope_repo/eval_outputs/rope_local"
export PYFLEXROOT="$rope_local/AdaptiGraph-main/PyFleX"
export PYTHONPATH="$rope_local/pyflex-build${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-GPU-c7f7b7e4-d054-8f4b-bde9-a4861aaba830}"
export EGL_GPU="${EGL_GPU:-1}"
export __EGL_VENDOR_LIBRARY_FILENAMES="${__EGL_VENDOR_LIBRARY_FILENAMES:-/usr/share/glvnd/egl_vendor.d/10_nvidia.json}"
export SDL_VIDEODRIVER=dummy
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export PYTHONUNBUFFERED=1
exec "$rope_local/venv/bin/python" "$rope_repo/experiments/rope_feedback/run_warmup5.py" "$@"
