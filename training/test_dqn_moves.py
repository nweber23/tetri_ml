"""Run from training/: python -m unittest. No server needed."""
import copy
import unittest

import numpy as np
import torch

from dqn_moves import BOARD, MAX_MOVES, MOVE_FEATURES, MoveNet, Replay, agent_doc, best_move, pack, training_rewards, unpack, update


class TestMoveDQN(unittest.TestCase):
    def test_zero_bait_only_removes_zero_line_tspins(self):
        info = {"rewards": np.array([400, 800, 100, 0], np.float32), "bait": np.array([True, False, False, False])}
        self.assertEqual(training_rewards(info, True).tolist(), [0, 800, 100, 0])
        self.assertEqual(training_rewards(info, False).tolist(), [400, 800, 100, 0])

    def test_board_scale_is_baked_into_the_exported_weights(self):
        torch.manual_seed(0)
        net = MoveNet(warm_start=False, board_scale=0.1)
        w = torch.tensor(agent_doc(net, "x")["layers"][0]["w"])
        raw = net.mlp[0].weight
        self.assertTrue(torch.allclose(w[:, BOARD], raw[:, BOARD] * 0.1))
        self.assertTrue(torch.equal(w[:, :4], raw[:, :4]))
        # the exported plain MLP gives the same value as the scaled network
        x = torch.rand(3, MOVE_FEATURES)
        exported = torch.relu(x / torch.tensor(agent_doc(net, "x")["scale"]) @ w.T + net.mlp[0].bias)
        self.assertTrue(torch.allclose(exported, torch.relu(net.mlp[0](x / net.mlp[0].weight.new_tensor(agent_doc(net, "x")["scale"]) * net.visible)), atol=1e-5))

    def test_pack_round_trip(self):
        f = np.random.default_rng(0).integers(0, 2, (3, MAX_MOVES, MOVE_FEATURES), dtype=np.uint8)
        f[..., :4] = 200  # the four scalar features are bytes, not bits
        np.testing.assert_array_equal(unpack(pack(f)), f)

    def test_warm_start_computes_the_heuristic(self):
        net = MoveNet()
        x = torch.zeros(2, MOVE_FEATURES)
        x[:, :4] = torch.tensor([[40.0, 3.0, 12.0, 0.0], [30.0, 0.0, 6.0, 2.0]])
        x[:, 4:] = torch.rand(2, MOVE_FEATURES - 4).round()  # the board and one-hots must not matter yet
        h = [-0.510066, -0.35663, -0.184483, 0.760666]
        want = [0.1 * sum(w * v for w, v in zip(h, row[:4].tolist())) for row in x]
        for got, w in zip(net(x).tolist(), want):
            self.assertAlmostEqual(got, w, places=4)

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

    def test_game_over_is_worth_the_penalty(self):
        torch.manual_seed(0)
        net = MoveNet()
        optimizer = torch.optim.Adam(net.parameters(), lr=1e-2)
        replay = Replay(8)
        for _ in range(8):
            replay.add(np.zeros(MOVE_FEATURES, np.uint8), True, np.zeros((MAX_MOVES, MOVE_FEATURES), np.uint8),
                       np.zeros(MAX_MOVES, np.float32), np.zeros(MAX_MOVES, bool))
        batch = replay.sample(8, np.random.default_rng(0))
        for _ in range(300):
            update(net, copy.deepcopy(net), optimizer, batch, death_penalty=5.0)
        self.assertAlmostEqual(net(batch[0])[0].item(), -5.0, places=1)

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
