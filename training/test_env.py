"""Run from training/: python -m unittest. Builds and starts the Go server itself."""
import os
import subprocess
import tempfile
import time
import unittest
from pathlib import Path

import gymnasium as gym
import numpy as np

import tetris_env  # noqa: F401

ROOT = Path(__file__).resolve().parent.parent
ADDR = "127.0.0.1:50099"


def setUpModule():
    global _tmp, _server
    _tmp = tempfile.TemporaryDirectory()
    exe = os.path.join(_tmp.name, "server.exe" if os.name == "nt" else "server")
    subprocess.run(["go", "build", "-o", exe, "./cmd/server"], cwd=ROOT, check=True)
    _server = subprocess.Popen([exe, "-addr", ADDR], stderr=subprocess.DEVNULL)
    time.sleep(1)


def tearDownModule():
    _server.terminate()
    _server.wait()
    _tmp.cleanup()


class TestTetrisEnv(unittest.TestCase):
    def setUp(self):
        self.env = gym.make("Tetris-v0", addr=ADDR)
        self.addCleanup(self.env.close)

    def test_reset_observation(self):
        obs, info = self.env.reset(seed=1)
        self.assertTrue(self.env.observation_space.contains(obs))
        self.assertEqual(obs["board"].sum(), 0)
        self.assertTrue(info["action_mask"].any())
        features = info["afterstates"]
        self.assertEqual(features.shape, (48, 4))
        self.assertFalse(features[~info["action_mask"]].any())
        self.assertTrue(features[info["action_mask"]][:, 0].all())  # every placement adds height

    def test_same_seed_same_pieces(self):
        a, _ = self.env.reset(seed=7)
        b, _ = self.env.reset(seed=7)
        self.assertEqual((a["piece"], a["next"]), (b["piece"], b["next"]))

    def test_invalid_action_raises(self):
        _, info = self.env.reset(seed=1)
        with self.assertRaises(ValueError):
            self.env.step(int(np.flatnonzero(~info["action_mask"])[0]))

    def test_random_game_terminates(self):
        rng = np.random.default_rng(0)
        _, info = self.env.reset(seed=1)
        done = False
        for _ in range(1000):
            _, _, done, _, info = self.env.step(rng.choice(np.flatnonzero(info["action_mask"])))
            if done:
                break
        self.assertTrue(done)
        self.assertFalse(info["action_mask"].any())

    def test_parallel_envs(self):
        envs = gym.make_vec("Tetris-v0", num_envs=4, vectorization_mode="async", addr=ADDR)
        self.addCleanup(envs.close)
        obs, infos = envs.reset(seed=1)
        self.assertEqual(obs["board"].shape, (4, 24, 10))
        actions = [int(np.flatnonzero(m)[0]) for m in infos["action_mask"]]
        obs, *_ = envs.step(np.array(actions))
        self.assertEqual(obs["piece"].shape, (4,))


if __name__ == "__main__":
    unittest.main()
