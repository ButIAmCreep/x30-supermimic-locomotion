import tempfile
import unittest
from pathlib import Path

import torch


class HandstandEvaluationTest(unittest.TestCase):
    def test_front_only_rejects_shank_or_knee_ground_support(self):
        import numpy as np
        from go2_supermimic.robots.x30.handstand_eval import front_only_contact_mask

        feet = np.array([11, 12, 13, 14])
        robot = np.array([11, 12, 13, 14, 15])  # 15 is a shank collision geom
        floor = np.array([1])
        base = np.array([[1, 11], [1, 12]])
        worlds = np.zeros(2, dtype=int)
        active = np.ones(2, dtype=bool)
        self.assertEqual(front_only_contact_mask(base, worlds, active, feet, robot, floor, 1).tolist(), [True])
        with_shank = np.vstack((base, [1, 15]))
        self.assertEqual(front_only_contact_mask(with_shank, np.zeros(3, dtype=int),
                                                 np.ones(3, dtype=bool), feet, robot, floor, 1).tolist(), [False])

    def test_success_requires_front_only_hold_and_recovery(self):
        from go2_supermimic.robots.x30.handstand_eval import handstand_passed

        row = {'falls': 0, 'hold_samples': 300, 'front_only_fraction': .9,
               'pitch_error_rms_deg': 6., 'rear_foot_height_min_m': .12,
               'base_collision_bottom_min_m': .12, 'recovery_samples': 50,
               'recovery_tilt_max_deg': 10., 'recovery_speed_error_m_s': .05}
        self.assertTrue(handstand_passed(row))
        for key, value in [('falls', 1), ('front_only_fraction', .2),
                           ('rear_foot_height_min_m', .02), ('recovery_samples', 0)]:
            bad = row | {key: value}
            self.assertFalse(handstand_passed(bad))

    def test_wrong_checkpoint_is_rejected_before_environment_creation(self):
        from go2_supermimic.robots.x30.handstand_eval import prepare

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'wrong.pt'
            torch.save({'contract': {'task': 'X30-Locomotion-Speed2'}}, path)
            with self.assertRaises(ValueError):
                prepare(str(path), 1, 42)


if __name__ == '__main__':
    unittest.main()
