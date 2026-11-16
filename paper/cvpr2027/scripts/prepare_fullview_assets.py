from pathlib import Path
import json,csv,hashlib,shutil,os,types
os.environ['MPLCONFIGDIR']='/tmp/jepa-fullview-mpl';os.environ['SDL_VIDEODRIVER']='dummy';os.environ['SDL_AUDIODRIVER']='dummy';os.environ['PYGAME_HIDE_SUPPORT_PROMPT']='1'
import numpy as np
from PIL import Image
import pymunk
import argparse
parser=argparse.ArgumentParser();parser.add_argument('--repo',type=Path,required=True);parser.add_argument('--paper',type=Path,default=Path(__file__).resolve().parents[1]);args=parser.parse_args();r=args.repo;p=args.paper;a=p/'figures/fullview_20261116';a.mkdir(exist_ok=True);v=p/'provenance';sha=lambda f:hashlib.sha256(f.read_bytes()).hexdigest()
# Verify all published LeWM file checksums and independently count paired outcomes.
l=r/'reports/lewm_random_matched16k_20261116';checks=json.loads((l/'SHA256SUMS.json').read_text())
for n,h in checks.items():assert sha(l/n)==h,n
rows=list(csv.DictReader((l/'evaluation/paired_tasks.csv').open()));assert len(rows)==350
counts={k:sum(int(x[k]) for x in rows) for k in ['random_native','random_ecp','augmented_native','augmented_ecp']};assert counts=={'random_native':166,'random_ecp':186,'augmented_native':215,'augmented_ecp':231}
for n in ['results.json','paired_tasks.csv','protocol_checks.json']:shutil.copyfile(l/'evaluation'/n,v/('lewm_matched_'+n))
h=json.loads((v/'headline_data_latest.json').read_text());h['series'][0]['label']='Base data';g=next(g for g in h['groups'] if g['label']=='LeWM');g['success_counts']=[166,215,231];g['evidence']['random_baseline_source']='provenance/lewm_matched_paired_tasks.csv';g['evidence']['random_baseline_sha256']=sha(v/'lewm_matched_paired_tasks.csv');g['evidence']['random_ecp_successes']=186
h['released_count_source']='Frozen/AdaJEPA: headline_data_v6.json; LeWM: lewm_matched_paired_tasks.csv'
h['protocol']=h['protocol'].replace('LeWM has no matched released baseline in this archive.','LeWM base data is newly trained matched random/source-action replay (166/350), not a released checkpoint.').replace('Brackets compare the first available bar with augmented ECP.','Brackets compare base-data native planning with augmented ECP for every model.')
(v/'headline_data_latest.json').write_text(json.dumps(h,indent=2)+'\n')
source=r/'env/pusht/pusht_env.py';code=source.read_text();assert code.count('torch')==1
m=types.ModuleType('native_pusht_renderer');exec(compile(code.replace('import torch\n',''),str(source),'exec'),m.__dict__)
native_draw=m.DrawOptions
native_fat_segment=native_draw.draw_fat_segment
boundary_segments=set()
def borderless_segment(self,a,b,radius,outline_color,fill_color):
 endpoints=frozenset((tuple(a),tuple(b)))
 if radius==2 and endpoints in boundary_segments:return
 return native_fat_segment(self,a,b,radius,outline_color,fill_color)
old={c['key']:c for c in json.loads((v/'replayed_paths_v12.json').read_text())['cases']};d=json.loads((v/'domain_visuals_20261116.json').read_text())
manifest={'remote_commit':'de78d6ec3c95dd999ad771f14ecf02d225683f70','view':'Full original camera frame for every panel; no local crops or insets.','renderer_source':str(source.relative_to(r)),'renderer_sha256':sha(source),'presentation_change':'Skip only four static Pymunk boundary-segment draw calls; physics and object geometry remain unchanged. No simulation step or inference.','pushobj':{},'pusht':[],'rope':[]}
for kind,cs in [('pushobj',[old['Z'],old['square']]),('pusht',d['pusht'])]:
 for c in cs:
  cond=c.get('condition');e=m.PushTEnv(with_velocity=True,with_target=True,render_size=512,shape=c['key'] if kind=='pushobj' else 'T',agent_color='Red' if cond=='redAgent' else 'RoyalBlue',goal_color='White' if kind=='pushobj' else ('Red' if cond=='redAnchor' else None));e._setup()
  walls=[s for s in e.space.shapes if isinstance(s,pymunk.Segment) and s.body==e.space.static_body];assert len(walls)==4
  boundary_segments={frozenset((tuple(w.a),tuple(w.b))) for w in walls}
  frames={}
  for mode in ['base','ours']:
   st=c['control_states'][mode][-1] if kind=='pushobj' else c['states'][mode]
   e.agent.position=tuple(st[:2]);e.agent.velocity=tuple(st[5:7]);e.block.angle=float(st[4]);e.block.position=tuple(st[2:4]);e.space.reindex_shapes_for_body(e.agent);e.space.reindex_shapes_for_body(e.block)
   native_draw.draw_fat_segment=native_fat_segment;original=e.render('rgb_array')
   native_draw.draw_fat_segment=borderless_segment;clean=e.render('rgb_array')
   assert np.any(original!=clean), 'Boundary removal did not change render'
   assert np.all(clean[:9]==255) and np.all(clean[-9:]==255) and np.all(clean[:,:9]==255) and np.all(clean[:,-9:]==255), 'Boundary remains'
   # Removing the debug frame must not change a single interior object pixel.
   np.testing.assert_array_equal(original[9:503,9:503],clean[9:503,9:503])
   native_draw.draw_fat_segment=native_fat_segment
   dest=a/f"{c['key']}_{mode}.png";Image.fromarray(clean).save(dest);frames[mode]={'path':str(dest.relative_to(p)),'sha256':sha(dest),'state':st}
  if kind=='pushobj':manifest['pushobj'][c['key']]={'frames':frames}
  else:cc=dict(c);cc['frames']=frames;manifest['pusht'].append(cc)
  e.close()
rope=r/'reports/rope_formal20/visuals_extra';rp=json.loads((rope/'provenance.json').read_text());frame_hashes={f['path']:f['sha256'] for s in rp['sources'] for f in s['frames']}
for c in rp['selection']:
 assert c['ours_cd']<c['native_cd'];frames={}
 for mode,arm,n in [('base','native5',c['native_pushes']),('ours','fixed3',c['ours_pushes'])]:
  rel=f"raw_frames/sample_{c['sample']:02d}/{arm}/push_{n:02d}.png";src=rope/rel;assert sha(src)==frame_hashes[rel]
  dest=a/f"rope_{c['sample']:02d}_{mode}.png";shutil.copyfile(src,dest);frames[mode]={'path':str(dest.relative_to(p)),'sha256':sha(dest),'source':str(src.relative_to(r))}
 cc=dict(c);cc.pop('crop');cc.update(frames=frames,key=f"rope_{c['sample']:02d}",excluded_from_formal20=True);manifest['rope'].append(cc)
for n in ['README.md','provenance.json','cd_by_push.csv','formal_vs_visualization.json']:shutil.copyfile(rope/n,a/('rope_extra_'+n))
(v/'fullview_visuals_20261116.json').write_text(json.dumps(manifest,indent=2)+'\n')
print('Verified LeWM counts',counts,'; rendered four full-scene pairs without boundary lines; imported two original Rope pairs.')
