#!/usr/bin/env python3
"""Compose recorded Rope observations with a registered goal annotation.

Use --import-repo to extract the selected matched execution. Normal builds read
the bundled, pixel-exact PNG observations and the provenance manifest. No new
physics, predicted frames, generative image edits, local crops, or smoothing.
"""
import argparse,collections,csv,hashlib,io,json,os,pickle,tarfile
from pathlib import Path
os.environ.setdefault('MPLCONFIGDIR','/tmp/jepa-rope-main-rollout')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image,ImageFilter

PAPER=Path(__file__).resolve().parents[1]
MANIFEST=PAPER/'provenance/rope_main_rollout_data.json'
STEPS=[5,8,14,18,19,20]
sha=lambda data:hashlib.sha256(data).hexdigest()

class ArrayOnlyUnpickler(pickle.Unpickler):
    def find_class(self,module,name):
        allowed={('numpy.core.multiarray','_reconstruct'):np._core.multiarray._reconstruct,
                 ('numpy','ndarray'):np.ndarray,('numpy','dtype'):np.dtype,
                 ('collections','OrderedDict'):collections.OrderedDict,
                 ('torch.storage','_load_from_bytes'):lambda _:None,
                 ('torch._utils','_rebuild_tensor_v2'):lambda *args:None}
        if (module,name) not in allowed:
            raise pickle.UnpicklingError(f'Unsupported archived class: {module}.{name}')
        return allowed[module,name]

def import_data(repo):
    report=repo/'reports/rope_seed101_20'
    receipts=json.loads((report/'evidence_manifest.json').read_text())
    expected={v['archive']:v['sha256'] for v in receipts.values()}
    rows={(r['arm'],int(r['sample'])):r for r in csv.DictReader((report/'per_sample.csv').open())}
    data={'source':'reports/rope_seed101_20','seed':101,'sample':5,'global_target':15,
          'steps':STEPS,'new_experiments_run':False,
          'selection':'Illustrative matched case where ECP is applied and succeeds while native planning fails at the same 20-push budget.',
          'protocol':'Fixed DINO-WM, three-push plans, per-push observations, five warm-up pushes, fixed-strength memory 2, CD < 0.3 success.',
          'archives':{},'files':{},'rows':[]}
    goals=[];starts=[];target_starts=[];identities=[];all_frames=[]
    def save(arr,name,archive,member,step):
        assert arr.shape==(224,224,3) and np.isfinite(arr).all()
        pixels=np.rint(arr).astype(np.uint8);assert np.array_equal(arr,pixels)
        rel=f'figures/recorded_rollouts/rope_main_{name}.png';f=PAPER/rel
        Image.fromarray(pixels).save(f)
        data['files'][rel]={'sha256':sha(f.read_bytes()),'archive':archive,'member':member,'step':step,
                            'full_original_viewport':True,'pixel_exact_to_recorded_observation':True}
        return rel
    for arm,label in [('native3','Native'),('fixed3','Ours')]:
        archive=report/f'{arm}_evidence.tar.gz'
        assert sha(archive.read_bytes())==expected[archive.name]
        data['archives'][archive.relative_to(repo).as_posix()]=expected[archive.name]
        row=rows[arm,5];assert row['actual_pushes']=='20'
        if arm=='fixed3':
            assert int(row['used_trigger_memories'])>0 and row['first_S_application_push']=='7'
        with tarfile.open(archive) as tar:
            base=f'{arm}/sample_05/'
            manifest=json.loads(tar.extractfile(base+'run_manifest.json').read())
            identities.append([manifest[k] for k in ['full_target_sha256','checkpoint_sha256','encoder_sha256']])
            cfg=manifest['config']['planner']
            assert cfg['chunk_size']==cfg['sub_planner']['horizon']==3
            assert cfg['ecp']['warmup_pushes']==5 and cfg['ecp']['independent_execution']
            if arm=='fixed3':assert cfg['strength_mode']=='fixed' and cfg['ecp']['kappa']==2
            member=base+'recorded_execution.npz';raw=tar.extractfile(member).read()
            with np.load(io.BytesIO(raw),allow_pickle=False) as z:
                assert z['physical_actions'].shape==(1,20,4) and z['obs_visual'].shape==(1,21,224,224,3)
                images=z['obs_visual'][0];starts.append(images[0]);all_frames.extend(images[k] for k in STEPS)
                with np.load(io.BytesIO(tar.extractfile(base+'rope_metrics.npz').read()),allow_pickle=False) as metrics:
                    assert metrics['actual_pushes'].item()==20 and metrics['executed_mask'].all()
                    cd=np.r_[metrics['initial_cd'].ravel(),metrics['cd_by_replan'].ravel()]
                    assert len(cd)==21 and cd[-1]==float(row['final_cd'])
                frames=[{'path':save(images[k],f'{arm}_{k:03}',archive.name,member,k),'step':k,'cd':float(cd[k])}
                        for k in STEPS]
                if arm=='native3':data['start']=save(images[0],'start',archive.name,member,0)
            target_member=base+('target.pkl' if arm=='native3' else 'plan_targets.pkl')
            target=ArrayOnlyUnpickler(io.BytesIO(tar.extractfile(target_member).read())).load()
            target_starts.append(target['obs_0']['visual'][0,0])
            goals.append(target['obs_g']['visual'][0,0])
            if arm=='native3':data['goal']=save(goals[-1],'goal',archive.name,target_member,None)
            data['rows'].append({'arm':arm,'label':label,'frames':frames,'final_cd':float(cd[-1]),
                                 'actual_pushes':20,'success':bool(cd[-1]<.3),
                                 'recorded_execution_sha256':sha(raw)})
    assert identities[0]==identities[1]
    assert np.array_equal(target_starts[0],target_starts[1])
    assert np.array_equal(starts[0],starts[1]) and np.array_equal(goals[0],goals[1])
    assert data['rows'][0]['final_cd']>data['rows'][1]['final_cd']
    assert not data['rows'][0]['success'] and data['rows'][1]['success']
    # Fixed-camera registration: the rendered background matches pixel-for-pixel.
    for arr in [*starts,*goals,*all_frames]:
        assert np.array_equal(arr[:40,:60],goals[0][:40,:60])
    goal=goals[0].astype(float)
    mask=(goal[:,:,0]>goal[:,:,1]+25)&(goal[:,:,0]>goal[:,:,2]+25)&(goal[:,:,0]>100)
    y,x=np.where(mask)
    assert len(x)==188 and (x.min(),y.min(),x.max(),y.max())==(76,106,136,145)
    data['identity']={'target_sha256':identities[0][0],'checkpoint_sha256':identities[0][1],
                      'encoder_sha256':identities[0][2],'first_ecp_application_push':7}
    data['goal_annotation']={'source':data['goal'],'red_rope_pixels':int(mask.sum()),
        'red_threshold':'R > G + 25, R > B + 25, R > 100','pixel_bounds':[76,106,136,145],
        'dilation_pixels':2,'display':'Pale-blue vector halo at original image coordinates.',
        'fixed_camera_background_pixel_check':True,'underlying_frames_unchanged':True}
    MANIFEST.write_text(json.dumps(data,indent=2)+'\n')
    print('Verified shared model/start/goal, 12 actual frames, equal pushes and fixed-camera registration.')
    return data

def plot(data):
    for name,record in data['files'].items():assert sha((PAPER/name).read_bytes())==record['sha256']
    goal=np.asarray(Image.open(PAPER/data['goal'])).astype(float)
    mask=(goal[:,:,0]>goal[:,:,1]+25)&(goal[:,:,0]>goal[:,:,2]+25)&(goal[:,:,0]>100)
    halo=np.asarray(Image.fromarray((mask*255).astype(np.uint8)).filter(ImageFilter.MaxFilter(5)))>0
    plt.rcParams.update({'font.family':'serif','font.serif':['STIXGeneral'],'font.size':9,
                         'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none','savefig.facecolor':'white'})
    width,height=6.875,2.10
    fig=plt.figure(figsize=(width,height))
    def text(x,y,value,**kw):return fig.text(x/width,y/height,value,**kw)
    def scene(path,x,y,annotate=False):
        side=.82
        ax=fig.add_axes([x/width,y/height,side/width,side/height])
        arr=np.asarray(Image.open(PAPER/path))
        ax.imshow(arr,interpolation='nearest',extent=(-.5,223.5,223.5,-.5))
        if annotate:
            # Contours are explicit annotations; the original RGB array is untouched.
            ax.contourf(np.arange(224),np.arange(224),halo.astype(float),levels=[.5,1.5],
                        colors=['#9BD3E6'],alpha=.32,antialiased=True)
            ax.contour(np.arange(224),np.arange(224),halo.astype(float),levels=[.5],
                       colors=['#80BED6'],linewidths=.45,alpha=.85)
        ax.set_xlim(-.5,223.5);ax.set_ylim(223.5,-.5);ax.axis('off')
    scene(data['start'],.04,1.10)
    scene(data['goal'],.04,.12)
    text(.45,1.98,'Start',ha='center',fontsize=9)
    text(.45,1.00,'Goal',ha='center',fontsize=9)
    for j,step in enumerate(data['steps']):
        text(1.43+.89*j+.41,1.98,f'Push {step}',ha='center',fontsize=8.5)
    for i,row in enumerate(data['rows']):
        y=1.10-.98*i;color='#B95B38' if i==0 else '#087D83'
        text(1.25,y+.41,row['label'],ha='center',va='center',rotation=90,fontsize=9,color=color,weight='bold')
        for j,frame in enumerate(row['frames']):
            x=1.43+.89*j
            scene(frame['path'],x,y,True)
            text(x+.41,y-.08,f"CD {frame['cd']:.3f}",ha='center',fontsize=7.7,color=color)
    fig.canvas.draw()
    renderer=fig.canvas.get_renderer()
    bounds=[(t.get_text(),t.get_window_extent(renderer)) for t in fig.texts]
    assert not [(a,b) for i,(a,x) in enumerate(bounds) for b,y in bounds[i+1:] if x.overlaps(y)]
    for suffix in ['pdf','png','svg']:
        f=PAPER/f'figures/fig_rope_main_rollout.{suffix}'
        kw={'metadata':{'CreationDate':None,'ModDate':None}} if suffix=='pdf' else {}
        fig.savefig(f,dpi=240,**kw)
        if suffix=='svg':f.write_text('\n'.join(line.rstrip() for line in f.read_text().splitlines())+'\n')
    plt.close(fig)
    print('Built full-view Rope execution strip with vector goal annotations and selectable labels.')

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--import-repo',type=Path)
    args=parser.parse_args();data=import_data(args.import_repo) if args.import_repo else json.loads(MANIFEST.read_text())
    plot(data)
