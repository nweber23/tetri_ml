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

    def test_double_target_uses_online_choice(self):
        # The online net prefers placement 0, the target net values placement 1 higher.
        # Vanilla DQN bootstraps from 1.0; Double DQN from the target's value of placement 0.
        class Fixed(torch.nn.Module):
            def __init__(self, values, own=0.0):
                super().__init__()
                self.values, self.own = torch.tensor(values), own
                self.bias = torch.nn.Parameter(torch.zeros(()))

            def forward(self, x):
                if x.dim() == 2:  # the sampled afterstate itself
                    return torch.full(x.shape[:1], self.own) + self.bias
                return self.values.expand(x.shape[:-1]) + self.bias

        mask = torch.zeros(1, NUM_ACTIONS, dtype=torch.bool)
        mask[0, :2] = True
        batch = (torch.zeros(1, NUM_FEATURES), torch.zeros(1), torch.zeros(1, dtype=torch.bool),
                 torch.zeros(1, NUM_ACTIONS, NUM_FEATURES), mask)
        online = Fixed([5.0, 1.0] + [0.0] * (NUM_ACTIONS - 2), own=5.0)
        target = Fixed([0.5, 1.0] + [0.0] * (NUM_ACTIONS - 2))
        for double, want in ((False, 1.0), (True, 0.5)):
            net = copy.deepcopy(online)
            # gamma=1, reward 0: the loss is the Huber distance from the online value (5.0) to the target
            loss = update(net, target, torch.optim.SGD(net.parameters(), lr=0.0), batch, 1.0, double)
            self.assertAlmostEqual(loss, abs(5.0 - want) - 0.5, places=4, msg=f"double={double}")


if __name__ == "__main__":
    unittest.main()
