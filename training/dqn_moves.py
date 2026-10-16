"""DQN over full-game moves (slides, tucks, spins, T-spins, hold).

U(afterstate) is the value of the state after a move, *excluding* the reward the move itself gave.
A move's value is r / 100 + gamma * U(afterstate), the agent plays the best one, and the TD target
for the U of the move it played is max over the next state's moves of that same quantity (0 if the
game ended). Keeping the reward out of U makes a T-spin's reward explicit instead of something the
network has to infer from the board.
"""
import numpy as np
import torch
from torch import nn

from tetris_env.env import MAX_MOVES, MOVE_FEATURES

GAMMA = 0.99  # the Go viewer uses the same value to rank moves
# Aggregate height, holes, bumpiness and lines are scaled; the 200 board cells and the one-hots are already 0/1.
SCALE = torch.tensor([100.0, 20.0, 50.0, 4.0] + [1.0] * (MOVE_FEATURES - 4))
BOARD = slice(4, 204)


class MoveNet(nn.Module):
    def __init__(self, hidden: int = 128):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(MOVE_FEATURES, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden // 2), nn.ReLU(),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, features):
        return self.mlp(features / SCALE).squeeze(-1)


def agent_doc(net: MoveNet, name: str) -> dict:
    linears = [m for m in net.mlp if isinstance(m, nn.Linear)]
    return {"name": name, "scale": SCALE.tolist(),
            "layers": [{"w": m.weight.tolist(), "b": m.bias.tolist()} for m in linears]}


def pack(features: np.ndarray) -> np.ndarray:
    """(..., MOVE_FEATURES) uint8 -> (..., PACKED): the 200 board cells as bits, 8x smaller."""
    return np.concatenate([features[..., :4], np.packbits(features[..., BOARD], axis=-1), features[..., 204:]], axis=-1)


def unpack(packed: np.ndarray) -> np.ndarray:
    return np.concatenate([packed[..., :4], np.unpackbits(packed[..., 4:29], axis=-1, count=200), packed[..., 29:]], axis=-1)


PACKED = 4 + 25 + 14


class Replay:
    def __init__(self, capacity: int):
        self.capacity, self.size, self.pos = capacity, 0, 0
        self.features = np.zeros((capacity, PACKED), np.uint8)
        self.done = np.zeros(capacity, bool)
        self.next_features = np.zeros((capacity, MAX_MOVES, PACKED), np.uint8)
        self.next_rewards = np.zeros((capacity, MAX_MOVES), np.float32)
        self.next_mask = np.zeros((capacity, MAX_MOVES), bool)

    def add(self, features, done, next_features, next_rewards, next_mask):
        i = self.pos
        self.features[i], self.done[i] = pack(features), done
        self.next_features[i], self.next_rewards[i], self.next_mask[i] = pack(next_features), next_rewards, next_mask
        self.pos = (i + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int, rng: np.random.Generator):
        idx = rng.integers(self.size, size=batch_size)
        mask = self.next_mask[idx]
        n = max(1, int(mask.sum(axis=1).max()))  # moves are listed from index 0, so the rest is padding
        f = lambda a: torch.from_numpy(a)  # noqa: E731
        return (f(unpack(self.features[idx]).astype(np.float32)), f(self.done[idx]),
                f(unpack(self.next_features[idx, :n]).astype(np.float32)),
                f(self.next_rewards[idx, :n]), f(mask[:, :n]))


def move_values(net, features, rewards, mask):
    """r / 100 + gamma * U for each move, -inf where there is no move."""
    return (rewards / 100.0 + GAMMA * net(features)).masked_fill(~mask, -torch.inf)


def best_move(net: MoveNet, features, rewards, mask) -> int:
    with torch.no_grad():
        return int(move_values(net, torch.from_numpy(features.astype(np.float32)),
                               torch.from_numpy(rewards), torch.from_numpy(mask)).argmax())


def update(net, target_net, optimizer, batch) -> float:
    features, done, next_features, next_rewards, next_mask = batch
    with torch.no_grad():
        best = move_values(target_net, next_features, next_rewards, next_mask).max(dim=1).values
        target = torch.where(done, torch.zeros_like(best), best)
    loss = nn.functional.smooth_l1_loss(net(features), target)
    optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(net.parameters(), 10.0)
    optimizer.step()
    return loss.item()


def evaluate(net: MoveNet, env, seeds, max_pieces: int) -> dict:
    """Greedy games on the given seeds; averages of score, lines, pieces, T-spins and holds used."""
    totals = np.zeros(5)
    for seed in seeds:
        _, info = env.reset(seed=seed)
        done = False
        while not done and info["pieces"] < max_pieces:
            a = best_move(net, info["features"], info["rewards"], info["action_mask"])
            _, _, done, _, info = env.step(a)
        totals += [info["score"], info["lines"], info["pieces"], info["tspins"], info["holds"]]
    avg = totals / len(seeds)
    return dict(zip(["score", "lines", "pieces", "tspins", "holds"], avg))
