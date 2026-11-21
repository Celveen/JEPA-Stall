"""Archive completed outputs alongside the matched previous experiment."""

import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parent
PARENT = Path("/home/appuser/workspace/project/multimodal-test/JEPA-Stall/reports/lewm_contact_schedule_20261113")
DEST = PARENT / ROOT.name

assert not DEST.exists(), DEST
assert len(list((ROOT / "full").glob("*/summary.json"))) == 7
assert len(list((ROOT / "replayed").glob("*/objective20.npz"))) == 7
assert json.loads((ROOT / "results.json").read_text())["total"]["tasks"] == 350
assert (ROOT / "RESULTS.md").exists()
assert (ROOT / "diagnostics.json").exists()
shutil.copytree(ROOT, DEST, ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "plan_targets.pkl"))
readme = PARENT / "README.md"
readme.write_text(readme.read_text() + "\n新增 [接触目标权重 20 实验](lewm_ecp_045_contact_objective_w20/RESULTS.md)，仅将接触目标代价权重从 10 提高到 20，完整保存匹配对照、日志与物理回放。\n")
checksums = {str(p.relative_to(PARENT)): hashlib.sha256(p.read_bytes()).hexdigest()
             for p in sorted(PARENT.rglob("*")) if p.is_file() and p.name != "SHA256SUMS.json"}
(PARENT / "SHA256SUMS.json").write_text(json.dumps(checksums, indent=2) + "\n")
for path in ROOT.rglob("*"):
    if path.is_file() and "__pycache__" not in path.parts and path.name != "plan_targets.pkl":
        saved = DEST / path.relative_to(ROOT)
        assert hashlib.sha256(path.read_bytes()).digest() == hashlib.sha256(saved.read_bytes()).digest(), path
print(f"Archived and verified: {DEST}")
