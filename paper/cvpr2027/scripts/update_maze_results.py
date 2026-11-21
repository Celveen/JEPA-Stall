#!/usr/bin/env python3
"""Insert fully verified 100-task frozen Maze results, without selecting an arm."""
import argparse,json,re
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('--results',type=Path,required=True);a=p.parse_args()
s=json.loads((a.results/'summary.json').read_text());m=json.loads((a.results/'manifest.json').read_text())
assert s['n']==m['n']==100 and s['baseline_successes']==48 and m['primary']=='S' and m['budget_replans']==30
for arm in ['S','SP']:
 d=s['arms'][arm];assert d['pretrigger_pairs_verified']==50 and d['untriggered_reused']==50
 assert d['successes']==48+len(d['rescues'])-len(d['regressions'])
p=ROOT/'sections/experiments.tex';t=p.read_text()
v=s['arms']['S']['successes']
row=f'Maze & Released & Frozen &48.0&'+r'\textbf{'+f'{v:.1f}'+r'}\\'
t,n=re.subn(r'^Maze & Released & Frozen &48\.0&.*$',lambda _:row,t,flags=re.MULTILINE)
assert n==1, 'Expected exactly one frozen Maze row'
t=t.replace('Maze uses $S$, with an adaptation gate for AdaJEPA; the frozen intervention is being evaluated.', 'Maze uses $S$: movement-triggered for Frozen and loss-gated for AdaJEPA.')
p.write_text(t)
p=ROOT/'sections/appendix.tex';t=p.read_text();pending='Full-set correction counts will be inserted after all reruns pass verification.'
sp=s['arms']['SP'];d=s['arms']['S']
completed=f"All rerun pairs passed pre-trigger action/state verification. Frozen native MPC solves 48/100 tasks; the prespecified memory-only correction solves {v}/100, with {len(d['rescues'])} rescues and {len(d['regressions'])} regressions. The combined $S{{+}}P$ control solves {sp['successes']}/100, with {len(sp['rescues'])} rescues and {len(sp['regressions'])} regressions."
assert pending in t or completed in t, 'Unexpected frozen Maze narrative: review before replacing'
t=t.replace(pending,completed)
p.write_text(t)
(ROOT/'provenance/maze_frozen_v10_completed.json').write_text(json.dumps({'summary':s,'protocol':m,'primary_arm_selected_before_evaluation':'S'},indent=2)+'\n')
print(json.dumps(s,indent=2))
