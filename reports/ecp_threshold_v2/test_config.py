import unittest
from pathlib import Path

from ecp_v2_evaluate import configuration


class ConfigurationTests(unittest.TestCase):
    def test_adajepa_045_preserves_existing_thirty_by_five_budget(self):
        c = configuration('adajepa_aug_g045', Path('/tmp/target.pkl'), Path('/tmp/out'))
        self.assertEqual((c.planner.max_iter, c.planner.n_taken_actions), (30, 5))
        self.assertEqual((c.planner.stag.gamma_lo, c.planner.stag.gamma_hi), (.15, .45))
        self.assertEqual(c.planner.stag.start, 10)

    def test_dino_interval_checks_preserve_six_by_twenty_five_budget(self):
        c = configuration('dinowm_aug_intra_g030', Path('/tmp/target.pkl'), Path('/tmp/out'))
        self.assertEqual((c.planner.max_iter, c.planner.n_taken_actions), (6, 25))
        self.assertEqual(c.planner._target_, 'intra_interval.IntraIntervalStagnationMPCPlanner')
        self.assertEqual((c.planner.control_timing.history_stride_controls,
                          c.planner.stag.window), (5, 3))
        self.assertEqual((c.planner.stag.gamma_lo, c.planner.stag.gamma_hi), (.1, .3))
        self.assertEqual((c.planner.sub_planner.num_samples,
                          c.planner.sub_planner.opt_steps), (200, 10))


if __name__ == '__main__':
    unittest.main()
