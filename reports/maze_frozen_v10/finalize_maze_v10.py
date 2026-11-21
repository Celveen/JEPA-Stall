#!/usr/bin/env python3
"""Populate a separate v10 manuscript when frozen Maze comparisons finish."""
import argparse,hashlib,json,os,subprocess,time,zipfile
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--source',type=Path,required=True);p.add_argument('--pusht',type=Path,required=True);p.add_argument('--tectonic',required=True);a=p.parse_args()
def write(d):
 tmp=a.output/'finalization_status.tmp';tmp.write_text(json.dumps(d,indent=2)+'\n');tmp.replace(a.output/'finalization_status.json')
try:
 while not (a.output/'summary.json').exists():
  pid=json.loads((a.output/'launcher.json').read_text())['pid']
  os.kill(pid,0)
  n=sum(json.loads(p.read_text()).get('status')=='complete' for p in a.output.glob('*/*/run_metadata.json'))
  write({'status':'waiting','completed_shards':n,'total_shards':20});time.sleep(30)
 dest=a.output/'paper_completed_v10';dest.mkdir(exist_ok=True)
 with zipfile.ZipFile(a.source) as z:
  assert all(not Path(n).is_absolute() and '..' not in Path(n).parts for n in z.namelist());z.extractall(dest)
 subprocess.run([os.sys.executable,str(dest/'scripts/update_maze_results.py'),'--results',str(a.output)],check=True)
 subprocess.run([os.sys.executable,str(dest/'scripts/update_comparison_tables.py'),'--results',str(a.pusht)],check=True)
 subprocess.run([os.sys.executable,str(dest/'scripts/build_paper.py'),'--tectonic',a.tectonic],check=True)
 m=json.loads((dest/'provenance/manuscript_manifest.json').read_text());m.update(revision='v10-maze-completed',maze_frozen_status='complete',pdf_sha256={n:hashlib.sha256((dest/n).read_bytes()).hexdigest() for n in ['paper.pdf','paper_main.pdf','paper_appendix.pdf']})
 (dest/'provenance/manuscript_manifest.json').write_text(json.dumps(m,indent=2)+'\n')
 write({'status':'complete','paper':str(dest/'paper.pdf'),'summary':str(a.output/'summary.json')})
except Exception as e:write({'status':'failed','error':repr(e)});raise
