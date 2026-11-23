"""Import recorded LeWM pairs and suppress boundary drawing for full-view figures."""
import argparse,hashlib,io,json,os,shutil,tarfile,types
from pathlib import Path
os.environ.update(SDL_VIDEODRIVER='dummy',SDL_AUDIODRIVER='dummy',PYGAME_HIDE_SUPPORT_PROMPT='1',MPLCONFIGDIR='/tmp/jepa-lewm-legend')
import numpy as np
from PIL import Image
ap=argparse.ArgumentParser();ap.add_argument('--repo',type=Path,required=True);ap.add_argument('--paper',type=Path,default=Path(__file__).resolve().parents[1]);a=ap.parse_args();r=a.repo;p=a.paper;v=p/'provenance';dest=p/'figures/lewm_shift_20261116';dest.mkdir(exist_ok=True)
sha=lambda f:hashlib.sha256(f.read_bytes()).hexdigest()
src=r/'reports/lewm_shift_visuals_20261116';checks=json.loads((src/'sha256.json').read_text());summary=json.loads((src/'summary.json').read_text());archive=src/'all_recorded_rollouts.tar.gz';assert sha(archive)==checks[archive.name]
manifest={'source':'reports/lewm_shift_visuals_20261116','archive_sha256':sha(archive),'view':'Full 224x224 recorded viewport; no local crop','boundary':'Source RGB unchanged; plotting hides only the 6-pixel grayscale frame with white vector strips','pusht':[]}
with tarfile.open(archive) as tar:
 for cond in ['red_agent','red_goal']:
  idx=summary['selected_targets'][cond];row=next(row for row in summary['all_episode_rows'] if row['condition']==cond and row['target_index']==idx);assert not row['native_success'] and row['ours_success']
  frames={};records={};goal=None;initial=None
  for mode,arm in [('base','native'),('ours','ours')]:
   with np.load(io.BytesIO(tar.extractfile(f'{cond}_{arm}/recorded_execution.npz').read())) as data:
    stop=row[arm+'_stop_controls'];state=data['states'][idx,stop];rgb=data['rgb'][idx,stop];g=data['state_g'][idx];start=data['state_0'][idx]
    if goal is not None:np.testing.assert_array_equal(goal,g);np.testing.assert_array_equal(initial,start)
    goal=g;initial=start
   rel=f'figures/{cond}/{arm}_frames/{stop:03d}.png';assert sha(src/rel)==checks[rel];np.testing.assert_array_equal(np.array(Image.open(src/rel)),rgb)
   border=np.concatenate([rgb[:6].reshape(-1,3),rgb[-6:].reshape(-1,3),rgb[:,:6].reshape(-1,3),rgb[:,-6:].reshape(-1,3)])
   assert np.all(border[:,0]==border[:,1]) and np.all(border[:,1]==border[:,2]), 'Non-boundary color at image perimeter'
   path=dest/f'{cond}_{mode}.png';shutil.copyfile(src/rel,path);assert sha(path)==sha(src/rel)
   frames[mode]={'path':str(path.relative_to(p)),'sha256':sha(path),'original_source':str((src/rel).relative_to(r)),'original_sha256':sha(src/rel)}
   records[mode]={'terminal_state':state.tolist(),'controls':stop,'replans':stop//5}
  case={'key':f'lewm_{cond}','task':'PushT','baseline':'LeWM','condition':cond,'sample_idx':idx,'frames':frames,'states':{'goal':goal.tolist(),'start':initial.tolist()},'replans':{k:d['replans'] for k,d in records.items()},'records':records,'recorded_result':row,'goal_scale':224/512,'hide_frame_pixels':6};manifest['pusht'].append(case)
for n in ['README.md','summary.json','paired_results.csv']:shutil.copyfile(src/n,dest/n)
(v/'lewm_figure4_visuals.json').write_text(json.dumps(manifest,indent=2)+'\n');print('Imported two LeWM rescue pairs; original RGB copied byte-for-byte; only boundary hidden by plot overlay.')
