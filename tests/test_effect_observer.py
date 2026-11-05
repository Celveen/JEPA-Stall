"""Critical observation/label boundaries, using synthetic RGB (no simulator)."""
import json
from pathlib import Path
import unittest
import cv2
import numpy as np
from datasets.effect_observer import measure_window, body_label, UNKNOWN, STATIC, EFFECT

ROOT = Path(__file__).resolve().parents[1]
CFG = json.loads((ROOT / "reports/effect_ab_20261103/protocol.json").read_text())["observer"]


def scene(moving=False, returning=False):
    rgb = np.full((16, 224, 224, 3), 255, dtype=np.uint8)
    prop = np.zeros((16, 4), dtype=np.float32)
    prop[:, :2] = [80, 350]
    if moving:
        prop[:, 0] += np.arange(16) * 4
        prop[:, 2] = 40
    if returning:
        prop[:, 0] += np.r_[np.arange(8), np.arange(7, -1, -1)] * 6
        prop[:, 2] = 60
    for t, im in enumerate(rgb):
        cv2.rectangle(im, (115, 85), (150, 120), (100, 120, 130), -1)
        cv2.rectangle(im, (100, 80), (120, 90), (100, 120, 130), -1)
        cv2.circle(im, tuple(np.rint(prop[t, :2] * 224/512).astype(int)), 7, (50, 90, 220), -1)
    return rgb, prop, np.ones((15, 2), dtype=np.float32) * 10


class EvidenceBoundaryTests(unittest.TestCase):
    def test_moving_pusher_static_object_is_not_static_visual(self):
        result = measure_window(*scene(moving=True), CFG)
        for pair in result["pairs"]:
            self.assertEqual(pair["labels"]["proprio"], EFFECT)
            self.assertEqual(pair["labels"]["visual"], EFFECT)
            self.assertEqual(pair["labels"]["object_diagnostic"], STATIC)

    def test_whole_scene_static_requires_attempt(self):
        rgb, prop, act = scene()
        result = measure_window(rgb, prop, act, CFG)
        for pair in result["pairs"]:
            self.assertEqual(pair["labels"]["visual"], STATIC)
            self.assertEqual(pair["labels"]["proprio"], STATIC)
        result = measure_window(rgb, prop, np.zeros_like(act), CFG)
        for pair in result["pairs"]:
            self.assertEqual(pair["labels"]["visual"], UNKNOWN)
            self.assertEqual(pair["labels"]["proprio"], UNKNOWN)

    def test_return_to_start_not_static(self):
        result = measure_window(*scene(returning=True), CFG)["pairs"][-1]
        self.assertEqual(result["labels"]["proprio"], UNKNOWN)
        self.assertNotEqual(result["labels"]["visual"], STATIC)

    def test_pre_action_alignment(self):
        rgb, prop, act = scene()
        act[:5] = 0
        result = measure_window(rgb, prop, act, CFG)
        self.assertEqual(result["pairs"][0]["body"]["action_rms"], 0)
        self.assertEqual(result["pairs"][0]["labels"]["proprio"], UNKNOWN)
        self.assertGreater(result["pairs"][1]["body"]["action_rms"], 0)


if __name__ == "__main__":
    unittest.main()
