"""Check encoded frame counts, incorporate physics visibility and refresh public entry points."""
import hashlib
import json
from pathlib import Path
import subprocess

OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[2]
validation=json.loads((OUT/'validation.json').read_text())
physics=json.loads((OUT/'physics_visibility.json').read_text())
for case in validation['cases']:
    folder=OUT/case['key']
    for arm,name in [('baseline','native'),('ours','ours')]:
        n=case['native_controls' if arm=='baseline' else 'ours_controls']
        count=int(subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0','-show_entries','stream=nb_frames','-of','default=nw=1:nk=1',str(folder/arm/'execution.mp4')],text=True))
        assert count==n+1
        check=next(p for p in physics if p['condition']==case['condition'] and p['target']==case['target'] and p['arm']==name)
        assert check['maximum_replay_state_error']==0 and check['minimum_agent_center_edge_margin_px']>15
validation['physics_substeps_visibility_verified']=physics
(OUT/'validation.json').write_text(json.dumps(validation,indent=2)+'\n')
readme=(OUT/'README.md').read_text()
extra='\n物理子步复核也已通过。红色物体组agent中心到边缘的全程最小距离为78.38px，红色目标组为23.85px，均大于15px半径。使用保存动作重放后，所有控制步状态与新评测记录完全一致。详见 `physics_visibility.json`。\n'
if extra.strip() not in readme:
    readme+=extra
    readme+='\n复现示例（在新输出目录运行）\n\n```bash\nCUDA_VISIBLE_DEVICES=0 TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 eval_outputs/rope_local/venv/bin/python media/community_simulations/pusht_same_backbone/run.py --condition red_object --arm native --n-evals 50 --output /tmp/pusht_red_object_native\n```\n\n同一颜色条件将 `--arm` 改为 `ours` 并更换输出目录即可运行配对方案。三种颜色条件为 `red_object`、`red_goal`、`red_agent`。\n'
(OUT/'README.md').write_text(readme)
parent=OUT.parent
index=(parent/'README.md').read_text()
notice='**PushT 颜色素材已更新为同 backbone 的两组新评测**：[红色物体与红色目标的成功/失败配对视频](pusht_same_backbone/README.md)。\n'
if notice.strip() not in index:
    index=index.replace('# 社区视频真实仿真逐帧素材\n','# 社区视频真实仿真逐帧素材\n\n'+notice,1)
index='\n'.join(line for line in index.split('\n') if not line.startswith('| [pusht_ablation]') and not line.startswith('| [pusht_methods]') and not line.startswith('- **PushT 消融**') and not line.startswith('- **PushT 方法对比**'))
index=index.replace('每个任务提供一组机制开关对比、一组完整方法对比，展示同起点和目标下的执行差异。',
                    'PushT使用两种颜色条件的同backbone干预开关配对，其余任务提供机制开关及完整方法对比，展示同起点和目标下的执行差异。')
index=index.replace('PushT、Rope、Maze 原始 PNG 直接导出已保存的真实仿真 RGB，并逐像素核验。PushObj 以及 PushT 方法对比的 AdaJEPA 行用原始动作在原 Pymunk 仿真器重放，逐五控制步核对原实验日志端点，最大允许误差 0.00051。',
                    '当前PushT、Rope、Maze原始PNG直接导出真实仿真RGB，并逐像素核验。PushT新增两组还验证了每个物理子步的agent可见性。PushObj用原始动作在原Pymunk仿真器重放，逐五控制步核对原实验日志端点，最大允许误差0.00051。')
for case in validation['cases']:
    key=case['key']
    row=f"| [{key}](pusht_same_backbone/{key}/comparison.mp4) | LeWM 干预关闭 → 同一 LeWM + Ours | 150 / {case['ours_controls']} | 预算内未完成 → 成功 |\n"
    if row.strip() not in index:
        index=index.replace('## 剪辑入口',row+'\n## 剪辑入口',1)
# Keep the two new rows contiguous with the existing table.
index=index.replace('\n\n| [case_', '\n| [case_')
(parent/'README.md').write_text(index)
manifest=json.loads((parent/'manifest.json').read_text())
manifest['current_pusht_color_delivery']='pusht_same_backbone/README.md'
for case in manifest['cases']:
    if case['key'].startswith('pusht_'):
        case['presentation_status']='Superseded by same-backbone color pairs in pusht_same_backbone/README.md'
(parent/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
for old in ['pusht_ablation','pusht_methods']:
    (parent/old/'README.md').write_text('# PushT 素材更新\n\n当前宣传素材请使用[同 backbone 的两组新评测](../pusht_same_backbone/README.md)，两组均已核验 Ours 成功、基线失败且 agent 全程在画面内。\n')
# The original index generator keeps its historical measurements but routes PushT to the replacement.
script=parent/'verify_and_index.py'
content=script.read_text()
call="    replacement = ROOT / 'pusht_same_backbone/finalize.py'\n    if replacement.exists():\n        import runpy\n        runpy.run_path(str(replacement), run_name='__main__')\n"
if "replacement = ROOT / 'pusht_same_backbone/finalize.py'" not in content:
    content=content.replace('    print(json.dumps(audit, indent=2))',call+'    print(json.dumps(audit, indent=2))')
    script.write_text(content)
manifest['sources'][str(script.relative_to(ROOT))]=hashlib.sha256(script.read_bytes()).hexdigest()
(parent/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
hashes={str(p.relative_to(OUT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in OUT.rglob('*') if p.is_file() and p.name!='SHA256SUMS.json' and '__pycache__' not in str(p)}
(OUT/'SHA256SUMS.json').write_text(json.dumps(hashes,indent=2)+'\n')
print('Verified all four videos, physics-step visibility, and updated PushT entry points.')
