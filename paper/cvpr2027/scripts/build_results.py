#!/usr/bin/env python3
"""Plot verified paired historical outcomes; no experiment execution."""
import argparse,json,hashlib
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
AP=argparse.ArgumentParser();AP.add_argument('--evidence',type=Path,default=Path('/tmp/jepa_cvpr2027/paper/provenance/evidence.json'));AP.add_argument('--out',type=Path,default=Path(__file__).resolve().parent);a=AP.parse_args();a.out.mkdir(exist_ok=True,parents=True)
d=json.loads(a.evidence.read_text());rows=d['pushobj_main30'];ordered=[next(r for r in rows if r['model']==m and r['base']==b) for m,b in [('official','frozen'),('bias16k','frozen'),('official','adajepa'),('bias16k','adajepa')]]
series={'Baseline':[r['baseline_success'] for r in ordered],'+ weighting':[r['arms']['T0P']['success'] for r in ordered],'+ weighting + memory':[r['arms']['T0SP']['success'] for r in ordered]}
for r in ordered:
 for arm in ['T0P','T0SP']:assert r['arms'][arm]['n']==350 and r['arms'][arm]['budget_replans']==30
maze=d['maze_e3']['arms']['SZ-switch']['30'];assert maze['n']==100 and maze['budget_replans']==30
plt.rcParams.update({'font.family':'serif','font.serif':['STIXGeneral'],'mathtext.fontset':'stix','font.size':9,'pdf.fonttype':42,'ps.fonttype':42,'axes.linewidth':.7})
f=plt.figure(figsize=(6.875,2.75));ax=f.add_axes([.075,.245,.625,.56]);mx=f.add_axes([.805,.245,.175,.56]);colors=['#A8AEB2','#406782','#B55C27'];xx=np.arange(4);w=.235
for j,(label,counts) in enumerate(series.items()):
 vals=np.array(counts)/350*100
 bars=ax.bar(xx+(j-1)*w,vals,w-.018,color=('white' if j==2 else colors[j]),label=label,edgecolor=(colors[j] if j==2 else 'white'),linewidth=(1.1 if j==2 else .35),zorder=3)
 for b,v in zip(bars,vals):ax.text(b.get_x()+b.get_width()/2,v+1.8,f'{v:.1f}',ha='center',va='bottom',fontsize=8)
ax.set_xticks(xx,['Released\nFrozen','Biased-data\nFrozen','Released\nAdaJEPA','Biased-data\nAdaJEPA'],fontsize=8.5);ax.set_ylabel('Success rate (%)',fontsize=9);ax.set_ylim(0,100);ax.set_yticks([0,25,50,75,100]);ax.set_xlim(-.55,3.55)
ax.legend(loc='upper left',bbox_to_anchor=(-.04,1.32),ncol=3,frameon=False,fontsize=8.5,handlelength=1.0,columnspacing=1.15,handletextpad=.4)
ax.text(.0,1.10,'(a) PushObj: 350 paired tasks',transform=ax.transAxes,fontsize=10)
v=[maze['baseline_success'],maze['success']]
bars=mx.bar([0,1],v,.54,color=[colors[0],'white'],edgecolor=['white',colors[2]],linewidth=[.35,1.1],zorder=3)
for b,z in zip(bars,v):mx.text(b.get_x()+b.get_width()/2,z+2,f'{z:.0f}',ha='center',va='bottom',fontsize=9)
mx.set_xticks([0,1],['AdaJEPA','+ switched\nmemory'],fontsize=8);mx.set_ylim(0,100);mx.set_yticks([0,25,50,75,100]);mx.tick_params(labelleft=False);mx.set_xlim(-.65,1.65)
mx.text(-.02,1.10,'(b) Maze: 100 tasks',transform=mx.transAxes,fontsize=10)
for ac in [ax,mx]:
 ac.spines[['top','right']].set_visible(False);ac.grid(axis='y',color='#E6E9EC',linewidth=.6,zorder=0);ac.tick_params(axis='both',length=2.5)
f.text(.40,.045,'Checkpoint / planner setting',ha='center',fontsize=9)
for ext in ['pdf','svg','png']:f.savefig(a.out/f'fig4_results.{ext}',dpi=200)
manifest={'evidence_path':str(a.evidence),'evidence_sha256':hashlib.sha256(a.evidence.read_bytes()).hexdigest(),'generator_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'budget_replans':30,'pushobj':{'denominator':350,'column_order':[{'checkpoint':r['model'],'planner':r['base']} for r in ordered],'series_counts':series},'maze':{'denominator':100,'baseline':'AdaJEPA','intervention':'SZ-switch (stall memory; no proprioceptive boost)','counts':v},'paired_tasks_shared_within_dataset':True,'interpretation':'Checkpoint/planner settings are conditions, not independent replicate experiments. No uncertainty bars: only the recorded runs are shown. The maze correction differs from the PushObj weighting variants.','width_inches':6.875,'height_inches':2.75,'minimum_font_pt':8}
(a.out/'results_figure_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(json.dumps(manifest,indent=2))
