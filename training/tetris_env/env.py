"""Gymnasium environment backed by the Go gRPC server."""
import queue

import grpc
import gymnasium as gym
import numpy as np
from gymnasium import spaces

from tetris.v1 import tetris_pb2 as pb
from tetris.v1 import tetris_pb2_grpc as pb_grpc

HEIGHT, WIDTH = 24, 10  # 4 hidden rows above the 20 visible ones
NUM_KINDS = 7
NUM_ROT = 4
X_MIN, NUM_X = -2, 12  # placement x ranges over [-2, 9]
NUM_ACTIONS = NUM_ROT * NUM_X
NUM_FEATURES = 4  # aggregate height, holes, bumpiness, lines cleared


def action_index(rot: int, x: int) -> int:
    return rot * NUM_X + (x - X_MIN)


class TetrisEnv(gym.Env):
    """Placement-based Tetris.

    An action is `rot * 12 + (x + 2)`: a rotation state and the column of the
    piece's bounding box's left edge. `info["action_mask"]` marks the valid
    actions and `info["afterstates"]` (NUM_ACTIONS x NUM_FEATURES) holds the
    board features after each one, zeros for invalid actions; sending an invalid one raises ValueError. The reward is the
    line-clear score (100/300/500/800).
    """

    metadata = {"render_modes": []}

    def __init__(self, addr: str = "127.0.0.1:50051"):
        self.observation_space = spaces.Dict({
            "board": spaces.Box(0, NUM_KINDS, (HEIGHT, WIDTH), np.uint8),
            "piece": spaces.Discrete(NUM_KINDS),
            "next": spaces.Discrete(NUM_KINDS),
        })
        self.action_space = spaces.Discrete(NUM_ACTIONS)
        self._channel = grpc.insecure_channel(addr)
        self._requests = queue.SimpleQueue()
        self._responses = pb_grpc.TetrisStub(self._channel).Play(iter(self._requests.get, None))
        self._mask = np.zeros(NUM_ACTIONS, bool)

    def _call(self, request):
        self._requests.put(request)
        res = next(self._responses)
        self._mask = np.zeros(NUM_ACTIONS, bool)
        features = np.zeros((NUM_ACTIONS, NUM_FEATURES), np.float32)
        for pl, a in zip(res.placements, res.afterstates):
            i = action_index(pl.rot, pl.x)
            self._mask[i] = True
            features[i] = (a.aggregate_height, a.holes, a.bumpiness, a.lines)
        s = res.state
        obs = {
            "board": np.frombuffer(s.board, np.uint8).reshape(HEIGHT, WIDTH),
            "piece": s.piece,
            "next": s.next,
        }
        info = {"action_mask": self._mask, "afterstates": features, "score": s.score, "lines": s.lines, "pieces": s.pieces}
        return obs, res.reward, s.done, info

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        game_seed = seed if seed is not None else int(self.np_random.integers(2**63))
        obs, _, _, info = self._call(pb.Request(start=pb.Reset(seed=game_seed)))
        return obs, info

    def step(self, action):
        action = int(action)
        if not 0 <= action < NUM_ACTIONS or not self._mask[action]:
            raise ValueError(f"invalid action {action}")
        rot, x = divmod(action, NUM_X)
        obs, reward, done, info = self._call(pb.Request(place=pb.Placement(rot=rot, x=x + X_MIN)))
        return obs, float(reward), done, False, info

    def close(self):
        self._requests.put(None)
        self._channel.close()


MAX_MOVES = 128  # the server never offers more
MOVE_FEATURES = 4 + 200 + 7 + 7  # see agent.MoveFeatures in Go


class MoveEnv(gym.Env):
    """Full-game Tetris: every way to lock a piece, including slides, tucks, spins and hold.

    An action is an index into the moves the server listed for the current state.
    `info["action_mask"]` marks the real ones, `info["features"]` (MAX_MOVES x
    MOVE_FEATURES, uint8) is the network input for the state after each move and
    `info["rewards"]` the score each move would give. `info["bait"]` marks the zero-line
    T-spins, which pay 400 without clearing anything. The reward is the score
    gained (line clears 100/300/500/800, T-spins 400-1600). `info["tspins"]` and
    `info["holds"]` count how often the episode's moves were T-spins or used hold.
    """

    metadata = {"render_modes": []}

    def __init__(self, addr: str = "127.0.0.1:50051"):
        self.observation_space = spaces.Dict({
            "board": spaces.Box(0, NUM_KINDS, (HEIGHT, WIDTH), np.uint8),
            "piece": spaces.Discrete(NUM_KINDS),
            "next": spaces.Discrete(NUM_KINDS),
        })
        self.action_space = spaces.Discrete(MAX_MOVES)
        self._channel = grpc.insecure_channel(addr)
        self._requests = queue.SimpleQueue()
        self._responses = pb_grpc.TetrisStub(self._channel).PlayMoves(iter(self._requests.get, None))
        self._moves = []
        self._tspins = self._holds = 0

    def _call(self, request):
        self._requests.put(request)
        res = next(self._responses)
        self._moves = res.moves
        mask = np.zeros(MAX_MOVES, bool)
        features = np.zeros((MAX_MOVES, MOVE_FEATURES), np.uint8)
        rewards = np.zeros(MAX_MOVES, np.float32)
        bait = np.zeros(MAX_MOVES, bool)
        for i, m in enumerate(res.moves):
            mask[i] = True
            features[i] = np.frombuffer(m.features, np.uint8)
            rewards[i] = m.reward
            bait[i] = m.t_spin and m.lines == 0
        s = res.state
        obs = {"board": np.frombuffer(s.board, np.uint8).reshape(HEIGHT, WIDTH), "piece": s.piece, "next": s.next}
        info = {"action_mask": mask, "features": features, "rewards": rewards, "bait": bait, "score": s.score,
                "lines": s.lines, "pieces": s.pieces, "tspins": self._tspins, "holds": self._holds}
        return obs, res.reward, s.done, info

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._tspins = self._holds = 0
        game_seed = seed if seed is not None else int(self.np_random.integers(2**63))
        obs, _, _, info = self._call(pb.MoveRequest(start=pb.Reset(seed=game_seed)))
        return obs, info

    def step(self, action):
        action = int(action)
        if not 0 <= action < len(self._moves):
            raise ValueError(f"invalid action {action}")
        self._tspins += self._moves[action].t_spin
        self._holds += self._moves[action].hold
        obs, reward, done, info = self._call(pb.MoveRequest(move=action))
        return obs, float(reward), done, False, info

    def close(self):
        self._requests.put(None)
        self._channel.close()
