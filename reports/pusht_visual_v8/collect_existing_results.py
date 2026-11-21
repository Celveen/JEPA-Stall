#!/usr/bin/env python3
"""Read archived native results only; missing entries remain explicitly absent."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from omegaconf import OmegaConf


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows, missing = [], []
    conditions = ("default", "redBlock", "redAnchor", "redAgent", "blur", "snp", "dark")
    for seed in (100, 200, 300):
        stem = "pushobj_T16k_bias_e3" + (f"_seed{seed}" if seed != 100 else "")
        for condition in conditions:
            for base in ("frozen", "adajepa"):
                directory = args.repo / "eval_outputs" / stem / condition / base
                result_path = directory / "final_eval_results.npz"
                if not result_path.exists():
                    missing.append(str(result_path))
                    continue
                cfg = OmegaConf.load(directory / ".hydra/config.yaml")
                assert cfg.seed == seed and cfg.n_evals == 50 and cfg.planner.max_iter == 30
                assert str(cfg.ckpt_base_path).endswith("checkpoints/pushobj_T16k_bias_e3_seed0")
                assert int(cfg.model_epoch) == 3
                data = np.load(result_path)
                success = data["successes"].astype(bool)
                assert len(success) == 50
                rows.append({"weights": "augmented", "base": base, "ecp": False,
                             "condition": "clean" if condition == "default" else condition,
                             "seed": seed, "n": int(len(success)), "successes": int(success.sum()),
                             "success_rate": float(success.mean()), "source": str(result_path),
                             "sha256": hashlib.sha256(result_path.read_bytes()).hexdigest(),
                             "protocol": "historical legacy_pool random target selection; native planner"})
    aggregates = []
    for condition in ("clean",) + conditions[1:]:
        row = {"condition": condition}
        for base in ("frozen", "adajepa"):
            selected = [r for r in rows if r["condition"] == condition and r["base"] == base]
            n, k = sum(r["n"] for r in selected), sum(r["successes"] for r in selected)
            row[base] = {"n": n, "successes": k, "success_rate": k/n if n else None}
        aggregates.append(row)
    result = {"kind": "archived native scores, not new ECP results",
              "comparison_scope": "Separate historical task sets; do not merge with external50 formal comparison",
              "rows": rows, "aggregates": aggregates, "missing": missing}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"loaded": len(rows), "missing": len(missing), "aggregates": aggregates}, indent=2))


if __name__ == "__main__":
    main()
