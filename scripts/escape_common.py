"""Small helpers shared by the escape analyses."""
import json
import math

import numpy as np


def finished(d):
    """A plan.py run directory that completed its final evaluation."""
    logs = d / "logs.json"
    return (d / "final_eval_results.npz").exists() and logs.exists() and "final_eval/success_rate" in logs.read_text()


def jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.exists() else []


def mcnemar(x, y):
    """Paired counts (x only, y only) and the exact two-sided McNemar p-value of two boolean arrays."""
    b, c = int(np.sum(x & ~y)), int(np.sum(~x & y))
    n = b + c
    if n == 0:
        return b, c, 1.0
    return b, c, min(1.0, 2 * sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n)
