"""DQN over placement afterstates.

The network scores the features of the board *after* a placement, V(afterstate).
The Q-value of an action is then the afterstate value, and the TD target is
r + gamma * max over the next state's placements of V_target(afterstate).
"""
import numpy as np
import torch
from torch import nn

from tetris_env.env import NUM_ACTIONS, NUM_FEATURES

# Rough upper bound of each feature (aggregate height, holes, bumpiness, lines) for scaling inputs.
FEATURE_SCALE = torch.tensor([100.0, 20.0, 50.0, 4.0])


class ValueNet(nn.Module):
    def __init__(self, hidden: int = 64):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(NUM_FEATURES, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, features):
        return self.mlp(features / FEATURE_SCALE).squeeze(-1)


class Replay:
    def __init__(self, capacity: int):
        self.capacity, self.size, self.pos = capacity, 0, 0
        self.features = np.zeros((capacity, NUM_FEATURES), np.float32)
        self.reward = np.zeros(capacity, np.float32)
        self.done = np.zeros(capacity, bool)
        self.next_features = np.zeros((capacity, NUM_ACTIONS, NUM_FEATURES), np.float32)
        self.next_mask = np.zeros((capacity, NUM_ACTIONS), bool)

    def add(self, features, reward, done, next_features, next_mask):
        i = self.pos
        self.features[i], self.reward[i], self.done[i] = features, reward, done
        self.next_features[i], self.next_mask[i] = next_features, next_mask
        self.pos = (i + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int, rng: np.random.Generator):
        idx = rng.integers(self.size, size=batch_size)
        return (torch.from_numpy(self.features[idx]), torch.from_numpy(self.reward[idx]),
                torch.from_numpy(self.done[idx]), torch.from_numpy(self.next_features[idx]),
                torch.from_numpy(self.next_mask[idx]))


def agent_doc(net: ValueNet, name: str) -> dict:
    """The network as the Go viewer's agent JSON (weights only; export.py adds curve and test vectors)."""
    linears = [m for m in net.mlp if isinstance(m, nn.Linear)]
    return {"name": name, "scale": FEATURE_SCALE.tolist(),
            "layers": [{"w": m.weight.tolist(), "b": m.bias.tolist()} for m in linears]}


def best_action(net: ValueNet, features, mask) -> int:
    with torch.no_grad():
        values = net(torch.from_numpy(features)).masked_fill(~torch.from_numpy(mask), -torch.inf)
    return int(values.argmax())


def update(net, target_net, optimizer, batch, gamma: float, double: bool = False) -> float:
    features, reward, done, next_features, next_mask = batch
    with torch.no_grad():
        target_values = target_net(next_features)
        if double:  # Double DQN: the online net picks the next placement, the target net values it
            pick = net(next_features).masked_fill(~next_mask, -torch.inf).argmax(dim=1, keepdim=True)
            next_value = target_values.gather(1, pick).squeeze(1)
        else:
            next_value = target_values.masked_fill(~next_mask, -torch.inf).max(dim=1).values
        target = reward + gamma * torch.where(done, torch.zeros_like(next_value), next_value)
    loss = nn.functional.smooth_l1_loss(net(features), target)
    optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(net.parameters(), 10.0)
    optimizer.step()
    return loss.item()


def evaluate(net: ValueNet, env, seeds, max_pieces: int) -> dict:
    """Greedy games on the given seeds; returns the average score, lines and pieces."""
    totals = np.zeros(3)
    for seed in seeds:
        _, info = env.reset(seed=seed)
        done = False
        while not done and info["pieces"] < max_pieces:
            a = best_action(net, info["afterstates"], info["action_mask"])
            _, _, done, _, info = env.step(a)
        totals += [info["score"], info["lines"], info["pieces"]]
    avg = totals / len(seeds)
    return {"score": avg[0], "lines": avg[1], "pieces": avg[2]}
