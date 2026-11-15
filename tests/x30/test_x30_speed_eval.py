import tempfile
import unittest
from pathlib import Path

import torch


class SpeedEvaluationTest(unittest.TestCase):
    def test_old_checkpoint_is_rejected_before_creating_environment(self):
        from go2_supermimic.robots.x30.speed_eval import prepare

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'wrong.pt'
            torch.save({'contract': {'task': 'X30-Locomotion-Curriculum'}}, path)
            with self.assertRaises(ValueError):
                prepare(str(path), (0., 1., 0.), 1, 42)


if __name__ == '__main__':
    unittest.main()
