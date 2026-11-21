#!/usr/bin/env bash
set -uo pipefail

root="$(cd -- "$(dirname -- "$0")" && pwd)"
python=/home/appuser/anaconda3/envs/ts/bin/python
shapes=(I L square Z small_tee plus T)
gpus=(0 1 3 4 5 6 7)
pids=()
mkdir -p "$root/full"

for i in "${!shapes[@]}"; do
    shape="${shapes[i]}"
    gpu="${gpus[i]}"
    CUDA_VISIBLE_DEVICES="$gpu" OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
        "$python" "$root/run_eval.py" --shape "$shape" --output "$root/full/$shape" \
        > "$root/full/$shape.log" 2>&1 &
    pids+=("$!")
    printf '%s GPU %s PID %s\n' "$shape" "$gpu" "$!"
done

status=0
for i in "${!pids[@]}"; do
    if wait "${pids[i]}"; then
        printf '%s complete\n' "${shapes[i]}"
    else
        printf '%s failed. See %s\n' "${shapes[i]}" "$root/full/${shapes[i]}.log" >&2
        status=1
    fi
done
exit "$status"
