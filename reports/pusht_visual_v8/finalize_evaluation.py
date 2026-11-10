#!/usr/bin/env python3
"""Watch the authorized evaluation; verify pairs and build a separate result draft."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
import zipfile
import numpy as np


def write(path, data):
    tmp=path.with_suffix('.tmp')
    tmp.write_text(json.dumps(data,indent=2)+'\n');tmp.replace(path)


def finalize(root, source, tectonic):
    manifest=json.loads((root/'manifest.json').read_text())
    assert manifest['n_evals']==50 and manifest['n_jobs']==56 and manifest['max_replans']==30
    data={}; rows=[]; paired=[]
    for job in manifest['jobs']:
        folder=root/job['id']; done=json.loads((folder/'completion.json').read_text())
        assert done['n']==50 and done['fingerprint']==job['fingerprint']
        attempt=folder/done['attempt']; raw=attempt/'final_eval_results.npz'
        with np.load(raw) as npz: arrays={k:npz[k] for k in npz.files}
        assert len(arrays['successes'])==50
        assert abs(arrays['successes'].mean()-done['success_rate'])<1e-7
        data[job['id']]=(arrays,attempt,done)
        rows.append({'id':job['id'],'successes':int(arrays['successes'].sum()),'n':50,
                     'success_percent':100*done['success_rate'],'npz_sha256':hashlib.sha256(raw.read_bytes()).hexdigest()})
    common=next(iter(data.values()))[0]
    for job in manifest['jobs']:
        arr,attempt,done=data[job['id']]
        np.testing.assert_array_equal(arr['state_0'],common['state_0'])
        np.testing.assert_array_equal(arr['state_g'],common['state_g'])
        if not job['ecp']: continue
        native,_,_=data[job['id'].removesuffix('_ecp')]
        ep=[json.loads(x) for x in (attempt/'stag_episodes.jsonl').read_text().splitlines() if x.strip()]
        assert len(ep)==50 and sorted(x['sample_idx'] for x in ep)==list(range(50))
        for episode in ep:
            i=episode['sample_idx']; first=episode['first_trigger']
            shared=first if first is not None else min(arr['actions'].shape[1],native['actions'].shape[1])
            np.testing.assert_allclose(arr['actions'][i,:shared],native['actions'][i,:shared],atol=1e-5,rtol=1e-5)
        ns=native['successes'].astype(bool); es=arr['successes'].astype(bool)
        paired.append({'id':job['id'],'native_successes':int(ns.sum()),'ecp_successes':int(es.sum()),
                       'rescues':int((~ns&es).sum()),'regressions':int((ns&~es).sum()),
                       'gain_pp':float((es.sum()-ns.sum())*2),'pretrigger_pairs_checked':50})
    write(root/'verification.json',{'status':'passed','jobs':56,'paired_episodes':1400,
                                    'all_initial_and_goal_states_equal':True,'rows':rows,'pairs':paired})
    text=['# PushT evaluation completed','', 'All 56 jobs completed (2,800 episode-runs). Each native/ECP comparison uses the same 50 tasks. Pre-trigger action equivalence passed for all 1,400 paired episodes.','',
          '| Checkpoint / condition / update | Native (%) | ECP (%) | Gain (pp) | Rescues | Regressions |',
          '|---|---:|---:|---:|---:|---:|']
    for p in paired:
        text.append(f"| {p['id'].removesuffix('_ecp')} | {p['native_successes']*2:.1f} | {p['ecp_successes']*2:.1f} | {p['gain_pp']:+.1f} | {p['rescues']} | {p['regressions']} |")
    (root/'summary.md').write_text('\n'.join(text)+'\n')
    dest=root/'paper_completed_v9';dest.mkdir(exist_ok=True)
    with zipfile.ZipFile(source) as z:
        assert all(not Path(n).is_absolute() and '..' not in Path(n).parts for n in z.namelist())
        z.extractall(dest)
    subprocess.run([os.sys.executable,str(dest/'scripts/update_comparison_tables.py'),'--results',str(root)],check=True)
    subprocess.run([os.sys.executable,str(dest/'scripts/build_paper.py'),'--tectonic',tectonic],check=True)
    doc=subprocess.check_output(['pdftotext',str(dest/'paper.pdf'),'-'],text=True)
    assert 'SE-JEPA' not in doc and 'SE–JEPA' not in doc
    m=json.loads((dest/'provenance/manuscript_manifest.json').read_text())
    m.update(revision='v9-completed-evaluation',primary_jobs_complete=56,
             revision_description='Automatically populated all completed matched PushT measurements. Narrative unchanged; new results await author interpretation.',
             new_experiments_run=True,pdf_sha256={n:hashlib.sha256((dest/n).read_bytes()).hexdigest() for n in ('paper.pdf','paper_main.pdf','paper_appendix.pdf')})
    write(dest/'provenance/manuscript_manifest.json',m)
    (dest/'EVALUATION_COMPLETED.md').write_text('All matched PushT measurements have been imported after verification. See ../verification.json and ../summary.md. The abstract and interpretation retain the reviewed v9 text; no automated SOTA claim was added.\n')
    return dest


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--source',type=Path,required=True);p.add_argument('--tectonic',required=True);a=p.parse_args()
    status=a.output/'finalization_status.json'
    try:
        while True:
            m=json.loads((a.output/'manifest.json').read_text())
            n=sum((a.output/j['id']/'completion.json').exists() for j in m['jobs'])
            write(status,{'status':'waiting','complete':n,'total':len(m['jobs']),'updated_utc':datetime.now(timezone.utc).isoformat()})
            if n==len(m['jobs']):break
            tail=(a.output/'launcher.log').read_text().splitlines()[-3:]
            if any(x.startswith('Finished:') for x in tail):
                raise RuntimeError(f'Launcher finished with only {n}/{len(m["jobs"])} completions; inspect failures before resuming.')
            pid=json.loads((a.output/'launcher.json').read_text())['pid']
            try:os.kill(pid,0)
            except ProcessLookupError:raise RuntimeError(f'Launcher exited with only {n}/{len(m["jobs"])} completed; inspect attempts and resume.')
            time.sleep(30)
        dest=finalize(a.output,a.source,a.tectonic)
        write(status,{'status':'complete','complete':56,'verified_pairs':1400,'paper':str(dest/'paper.pdf'),'summary':str(a.output/'summary.md')})
        print('Verified results and populated PDF:',dest,flush=True)
    except Exception as exc:
        write(status,{'status':'failed','error':repr(exc)});raise


if __name__=='__main__':main()
