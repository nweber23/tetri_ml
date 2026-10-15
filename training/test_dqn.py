"""Run from training/: python -m unittest. No server needed."""
import copy
import unittest

import numpy as np
import torch

from dqn import Replay, ValueNet, best_action, update
from tetris_env.env import NUM_ACTIONS, NUM_FEATURES


class TestDQN(unittest.TestCase):
    def test_best_action_ignores_invalid(self):
        torch.manual_seed(0)
        net = ValueNet()
        features = np.random.default_rng(0).random((NUM_ACTIONS, NUM_FEATURES), dtype=np.float32) * 10
        mask = np.zeros(NUM_ACTIONS, bool)
        mask[[3, 17]] = True
        self.assertIn(best_action(net, features, mask), (3, 17))

    def test_update_fits_terminal_reward(self):
        torch.manual_seed(0)
        net = ValueNet()
        target_net = copy.deepcopy(net)
        optimizer = torch.optim.Adam(net.parameters(), lr=1e-2)
        replay = Replay(8)
        for _ in range(8):  # terminal transitions: the target is exactly the reward
            replay.add(np.array([20, 1, 3, 0], np.float32), 2.0, True,
                       np.zeros((NUM_ACTIONS, NUM_FEATURES), np.float32), np.zeros(NUM_ACTIONS, bool))
        batch = replay.sample(8, np.random.default_rng(0))
        for _ in range(300):
            loss = update(net, target_net, optimizer, batch, 0.99)
        self.assertLess(loss, 1e-3)
        self.assertAlmostEqual(net(batch[0])[0].item(), 2.0, places=1)


if __name__ == "__main__":
    unittest.main()
