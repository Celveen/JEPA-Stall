#!/usr/bin/env python3
"""Render method comparisons from audited baselines and completed primary runs.

Does not run experiments. Partial or smoke results are never substituted for
50-task results. Re-run after the primary scheduler finishes, then build_paper.py.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

from pushobj_results import REPLANS, SHAPES, TASKS_PER_SHAPE, baseline_rows, paired_rows, followup_rows

ROOT = Path(__file__).resolve().parents[1]
CONDITIONS = ('clean', 'blur', 'snp', 'dark', 'redAgent', 'redBlock', 'redAnchor')
TARGET_SHA = '1950b39e5d3ba13b30c14a435a400d84cd3fbe30138cdfed55e98b202d76277b'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, default=ROOT.parents[1] / 'eval_outputs/pusht_visual_v8')
    parser.add_argument('--require-complete', action='store_true')
    args = parser.parse_args()
    baseline = json.loads((ROOT / 'provenance/visual_baseline_data_v14.json').read_text())
    assert baseline['target_sha256'] == TARGET_SHA and baseline['n'] == 50
    current, evidence = {}, []
    if (args.results / 'manifest.json').exists():
        manifest = json.loads((args.results / 'manifest.json').read_text())
        assert manifest['target_sha256'] == TARGET_SHA and manifest['n_evals'] == 50
        assert manifest['max_replans'] == 30 and manifest['n_jobs'] == 56
        for job in manifest['jobs']:
            path = args.results / job['id'] / 'completion.json'
            if not path.exists():
                continue
            done = json.loads(path.read_text())
            assert done['n'] == 50 and done['fingerprint'] == job['fingerprint']
            raw = path.parent / done['attempt'] / 'final_eval_results.npz'
            assert raw.exists()
            with np.load(raw) as data:
                assert data['successes'].shape == (50,)
                successes = int(data['successes'].sum())
                assert abs(successes / 50 - done['success_rate']) < 1e-7
            score = successes * 2
            current[(job['weights'], job['base'], job['ecp'], job['condition'])] = score
            evidence.append({'id': job['id'], 'score_percent': score, 'successes': successes, 'n': 50,
                             'fingerprint': job['fingerprint'],
                             'raw_sha256': hashlib.sha256(raw.read_bytes()).hexdigest()})
    if args.require_complete:
        assert len(evidence) == 56, 'A complete matrix requires all 56 completed jobs'
        verified = json.loads((args.results / 'verification.json').read_text())
        assert verified['status'] == 'passed' and verified['paired_episodes'] == 1400
        checked = {r['id']: r for r in verified['rows']}
        for row in evidence:
            assert row['raw_sha256'] == checked[row['id']]['npz_sha256']
            assert row['successes'] == checked[row['id']]['successes']

    def values(weights, base, ecp):
        return [current.get((weights, base, ecp, c)) for c in CONDITIONS]

    main_scores = [row['success_percent'] for row in baseline['rows']] + [
        values('released', b, False) for b in ('frozen', 'adajepa')] + [
        values('augmented', b, True) for b in ('frozen', 'adajepa')]
    scored = [s + [sum(s[1:]) / 6 if all(v is not None for v in s[1:]) else None]
              for s in main_scores]
    best = [max(v[i] for v in scored if v[i] is not None) for i in range(8)]

    def cells(scores, highlight=False):
        shifted = scores[1:]
        avg = sum(shifted) / 6 if all(v is not None for v in shifted) else None
        numbers = scores + [avg]
        formatted = ['---' if v is None else (f'{v:.1f}' if i == 7 else f'{v:.0f}')
                     for i, v in enumerate(numbers)]
        return ' & '.join(r'\textbf{' + cell + '}' if highlight and numbers[i] == best[i] else cell
                          for i, cell in enumerate(formatted))

    repo, provenance = ROOT.parents[1], ROOT / 'provenance'
    pushobj = {**baseline_rows(repo, provenance), **paired_rows(repo, provenance)}
    completed = followup_rows(repo)
    for method in ('ours_frozen', 'ours_adajepa'):
        pushobj[method, 'GD'] = completed[method, 'GD']
    shape_counts = {}
    for key, row in pushobj.items():
        counts = row['successes_by_shape']
        shape_counts[key] = [counts[s] for s in SHAPES] + [sum(counts.values())]
    best_pushobj = [max(column) for column in zip(*shape_counts.values())]

    def shape_cells(method, planner):
        counts = shape_counts[method, planner]
        scores = [f'{100 * n / TASKS_PER_SHAPE:.0f}' for n in counts[:-1]]
        scores.append(f'{100 * counts[-1] / (len(SHAPES) * TASKS_PER_SHAPE):.1f}')
        return ' & '.join(r'\textbf{' + score + '}' if n == best else score
                          for score, n, best in zip(scores, counts, best_pushobj))

    rows = []
    cites = {'DINO-WM': 'zhou2025dinowm', 'LeWM': 'maes2026lewm', 'DDP-WM': 'yin2026ddpwm'}
    for row in baseline['rows']:
        method, planner = row['method'], row['planner']
        rows.append(f'{method}~\\cite{{{cites[method]}}} & {planner} & {shape_cells(method, planner)}'
                    f" & {cells(row['success_percent'], highlight=True)}" + r'\\')
    rows.append(r'\midrule')
    for label, w, b, e in [('Frozen (released)', 'released', 'frozen', False),
                           ('AdaJEPA (released)', 'released', 'adajepa', False)]:
        rows.append(f'{label} & GD & {shape_cells(f"released_{b}", "GD")}'
                    f' & {cells(values(w, b, e), highlight=True)}' + r'\\')
    rows.append(r'\midrule')
    for label, b in [(r'\textbf{Ours (frozen)}', 'frozen'), (r'\textbf{Ours (AdaJEPA)}', 'adajepa')]:
        rows.append(f'{label} & GD & {shape_cells(f"ours_{b}", "GD")}'
                    f' & {cells(values("augmented", b, True), highlight=True)}' + r'\\')
    table = r'''\begin{table*}[t]
\centering
\caption{\textbf{World-model control on PushObj and PushT.} Success (\%) at 150 controls. PushObj uses 50 tasks per shape and different task lists across model families. Sq. and sT denote square and small-T. Avg. covers seven shapes or six visual shifts. Agent, Object, and Goal denote red-color shifts. Bold marks column maxima. Ours uses augmented data, threshold 0.45 on PushObj and 0.15 on PushT. Dashes mark unmeasured results. See \cref{sec:pushobj-comparison,sec:pusht-extension,sec:completed-followups}.}
\label{tab:main}
\small
\setlength{\tabcolsep}{2pt}
\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}ll*{16}{r}@{}}
\toprule
& &\multicolumn{8}{c}{PushObj}&\multicolumn{8}{c}{PushT}\\
\cmidrule(lr){3-10}\cmidrule(lr){11-18}
Method & Opt. & I & L & Sq. & Z & sT & $+$ & T & Avg. & Clean & Blur & S\&P & Dark & Agent & Object & Goal & Avg.\\
\midrule
'''+ '\n'.join(rows) + r'''
\bottomrule
\end{tabular*}
\end{table*}
'''
    (ROOT / 'tables/world_model_comparison.tex').write_text(table)
    matrix_status = 'All 56 evaluations are complete.' if len(evidence) == 56 else 'Dashes denote measurements awaiting completion.'
    matrix = [r'\begin{table*}[t]', r'\centering',
              r'\caption{\textbf{Checkpoint and intervention comparisons on PushT.} Success (\%) on the same 50 tasks and at most 30 replans. Every Native/ECP pair shares the checkpoint, tasks, update policy, and pre-trigger random stream. '+matrix_status+' Avg. averages the six visual shifts.}',
              r'\label{tab:pusht-ecp}', r'\small', r'\setlength{\tabcolsep}{3.7pt}',
              r'\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}llcrrrrrrrr@{}}', r'\toprule',
              r'Checkpoint & Updates & ECP & Clean & Blur & S\&P & Dark & Red agent & Red object & Red goal & Avg.\\',r'\midrule']
    for w in ('released', 'augmented'):
        if w == 'augmented': matrix.append(r'\midrule')
        for b in ('frozen', 'adajepa'):
            for e in (False, True):
                matrix.append(f'{"Released" if w=="released" else "Data aug."} & {"Frozen" if b=="frozen" else "AdaJEPA"} & {"On" if e else "Off"} & {cells(values(w,b,e))}'+r'\\')
    matrix.extend([r'\bottomrule',r'\end{tabular*}',r'\end{table*}'])
    (ROOT / 'tables/pusht_checkpoint_matrix.tex').write_text('\n'.join(matrix)+'\n')
    out = {'primary_jobs_complete':len(evidence),'primary_jobs_total':56,'conditions':CONDITIONS,
           'target_sha256':TARGET_SHA,'primary_results':evidence,'baseline_rows':baseline['rows'],
           'smoke_used_as_primary':False,
           'main_comparison':'PushObj per-shape scores across two task lists; PushT uses shared tasks; all at 150 controls',
           'pushobj': {'shapes': SHAPES, 'tasks_per_shape': TASKS_PER_SHAPE, 'max_replans': REPLANS,
                       'rows': [dict(method=method, planner=planner, **row)
                                for (method, planner), row in pushobj.items()]}}
    (ROOT / 'provenance/comparison_v14.json').write_text(json.dumps(out,indent=2)+'\n')
    print(f'Updated comparison tables: {len(evidence)}/56 primary jobs completed; unmeasured cells retained.')


if __name__ == '__main__':
    main()
