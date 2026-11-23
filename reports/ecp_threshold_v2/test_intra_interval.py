import json,os,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
from intra_interval import IntraIntervalStagnationMPCPlanner
from planning.escape import EscapeMPCPlanner

class DetectorTests(unittest.TestCase):
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory();self.cwd=os.getcwd();os.chdir(self.tmp.name)
 def tearDown(self):os.chdir(self.cwd);self.tmp.cleanup()
 def planner(self,iteration=2,success=False):
  p=object.__new__(IntraIntervalStagnationMPCPlanner)
  p.iter=iteration;p.max_iter=6;p.executed_controls=25;p.history_stride_controls=5;p.warmup_controls=50;p.stag_window=3;p.stag_gamma_lo=.1;p.stag_gamma_hi=.3;p.stag_alpha=1;p.stag_prop_boost=10;p.stag_boost_replans=0;p._sample_idx=0;p.is_success=np.array([success]);p._records=[];p._reset_stag();p._obs_g={'signal':np.array([[[100.]]])}
  def encode(obs):return torch.tensor([float(x['signal'].reshape(-1)[0])for x in obs],dtype=torch.float64)[:,None],torch.zeros(len(obs),1,dtype=torch.float64)
  p._encode=encode;return p
 def run_feedback(self,p,values):
  def parent(obj,taken,obs):obj._records.append({'step':obj.iter+1});return None
  with patch.object(EscapeMPCPlanner,'_post_env_feedback',parent):p._post_env_feedback(None,{'signal':np.array(values,dtype=float)[None,:,None]})
  return [json.loads(s)for s in Path('intra_detection.jsonl').read_text().splitlines()]
 def test_interior_stall_is_retained_until_next_existing_plan(self):
  p=self.planner();x=np.linspace(0,80,76);x[40]=20;x[55]=20;x[60]=35;x[45]=0;x[65]=50;x[50]=10;x[70]=65;x[75]=80
  rows=self.run_feedback(p,x)
  self.assertEqual([r['control']for r in rows],[55,60,65,70,75]);self.assertGreater(rows[-1]['movement_ratio'],.3)
  self.assertEqual(p._first_detection_control,55);self.assertEqual(p._stag_first,3);self.assertEqual(p._first_intervention_control,75);self.assertEqual(len(p._mem_w),1)
  self.assertEqual(float(p._mem_obs[0]['signal'].reshape(-1)[0]),20);self.assertEqual(p._boost_until,3)
  self.assertEqual(p._episode_extra()['first_detection_control'],55)
  self.assertEqual(p._episode_extra()['first_intervention_control'],75)
 def test_multiple_hits_create_one_memory_per_interval(self):
  p=self.planner();rows=self.run_feedback(p,np.zeros(76));self.assertEqual(sum(r['candidate']for r in rows),5);self.assertEqual(len(p._mem_w),1);self.assertEqual(p._first_detection_control,55)
 def test_warmup_uses_physical_controls(self):
  p=self.planner(iteration=1);rows=self.run_feedback(p,np.zeros(51));self.assertEqual([r['control']for r in rows if r['eligible']],[50]);self.assertEqual(p._first_detection_control,50);self.assertEqual(p._stag_first,2)
 def test_success_and_final_block_do_not_schedule_an_intervention(self):
  for iteration,success in [(2,True),(5,False)]:
   with self.subTest(iteration=iteration,success=success):
    p=self.planner(iteration,success);rows=self.run_feedback(p,np.zeros((iteration+1)*25+1));self.assertIsNone(p._stag_first);self.assertEqual(p._mem_w,[]);self.assertFalse(any(r['eligible']for r in rows[-5:]))
 def test_all_six_blocks_keep_six_plans_and_sixteen_actionable_checks(self):
  p=self.planner(iteration=0)
  for i in range(6):p.iter=i;self.run_feedback(p,np.linspace(0,90,(i+1)*25+1))
  rows=[json.loads(s)for s in Path('intra_detection.jsonl').read_text().splitlines()]
  self.assertEqual(len(rows),30);self.assertEqual(sum(r['eligible']for r in rows),16);self.assertEqual([r['control']for r in rows if r['eligible']],list(range(50,126,5)))
  self.assertEqual(len(p._records),6)

if __name__=='__main__':unittest.main()
