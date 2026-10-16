"""Run from training/: python -m unittest. No server needed."""
import copy
import unittest

import numpy as np
import torch

from dqn_moves import MAX_MOVES, MOVE_FEATURES, MoveNet, Replay, best_move, pack, unpack, update


class TestMoveDQN(unittest.TestCase):
    def test_pack_round_trip(self):
        f = np.random.default_rng(0).integers(0, 2, (3, MAX_MOVES, MOVE_FEATURES), dtype=np.uint8)
        f[..., :4] = 200  # the four scalar features are bytes, not bits
        np.testing.assert_array_equal(unpack(pack(f)), f)

    def test_best_move_counts_reward_and_ignores_padding(self):
        torch.manual_seed(0)
        net = MoveNet()
        features = np.zeros((MAX_MOVES, MOVE_FEATURES), np.uint8)
        rewards = np.zeros(MAX_MOVES, np.float32)
        rewards[5] = 1200.0  # a T-spin double dwarfs whatever the untrained network thinks
        mask = np.zeros(MAX_MOVES, bool)
        mask[[2, 5]] = True
        self.assertEqual(best_move(net, features, rewards, mask), 5)

    def test_update_fits_terminal_target(self):
        torch.manual_seed(0)
        net = MoveNet()
        optimizer = torch.optim.Adam(net.parameters(), lr=1e-2)
        replay = Replay(8)
        f = np.zeros((MOVE_FEATURES,), np.uint8)
        for _ in range(8):  # terminal: the target is exactly 0
            replay.add(f, True, np.zeros((MAX_MOVES, MOVE_FEATURES), np.uint8),
                       np.zeros(MAX_MOVES, np.float32), np.zeros(MAX_MOVES, bool))
        batch = replay.sample(8, np.random.default_rng(0))
        for _ in range(200):
            loss = update(net, copy.deepcopy(net), optimizer, batch)
        self.assertLess(loss, 1e-3)

    def test_update_bootstraps_from_best_next_move(self):
        torch.manual_seed(0)
        net = MoveNet()
        replay = Replay(4)
        f = np.zeros((MOVE_FEATURES,), np.uint8)
        nf = np.zeros((MAX_MOVES, MOVE_FEATURES), np.uint8)
        rewards = np.zeros(MAX_MOVES, np.float32)
        rewards[:2] = [300.0, 1200.0]
        mask = np.zeros(MAX_MOVES, bool)
        mask[:2] = True
        replay.add(f, False, nf, rewards, mask)
        batch = replay.sample(1, np.random.default_rng(0))
        with torch.no_grad():
            u = net(torch.zeros(MOVE_FEATURES)).item()
        # both next moves have the same afterstate, so the target is the larger reward plus gamma * U
        want = 12.0 + 0.99 * u
        optimizer = torch.optim.SGD(net.parameters(), lr=0.0)
        self.assertAlmostEqual(update(net, copy.deepcopy(net), optimizer, batch),
                               torch.nn.functional.smooth_l1_loss(torch.tensor(u), torch.tensor(want)).item(), places=4)


if __name__ == "__main__":
    unittest.main()
