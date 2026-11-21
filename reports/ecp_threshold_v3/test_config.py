import unittest
from pathlib import Path

from evaluate import configuration


class ConfigurationTests(unittest.TestCase):
    def test_adajepa_060_keeps_budget_and_adaptation(self):
        c = configuration('adajepa_aug_g060', Path('/tmp/target.pkl'), Path('/tmp/ada'))
        self.assertEqual((c.planner.max_iter, c.planner.n_taken_actions), (30, 5))
        self.assertEqual((c.planner.stag.gamma_lo, c.planner.stag.gamma_hi), (.20, .60))
        self.assertEqual(c.planner.stag.start, 10)
        self.assertEqual(c.planner.probe.seed, 0)
        self.assertFalse('mode' in c.planner.probe)

    def test_our_frozen_045_keeps_same_checkpoint_and_budget(self):
        c = configuration('frozen_aug_g045', Path('/tmp/target.pkl'), Path('/tmp/frozen'))
        self.assertEqual((c.planner.max_iter, c.planner.n_taken_actions), (30, 5))
        self.assertEqual((c.planner.stag.gamma_lo, c.planner.stag.gamma_hi), (.15, .45))
        self.assertEqual(c.planner.probe.mode, 'none')
        self.assertTrue(c.ckpt_base_path.endswith('pushobj_bias_16k_e3_seed0'))
        self.assertEqual(c.model_epoch, '3')


if __name__ == '__main__':
    unittest.main()
