import gymnasium as gym

from .env import MoveEnv, TetrisEnv

__all__ = ["TetrisEnv", "MoveEnv"]

gym.register(id="Tetris-v0", entry_point=TetrisEnv)
gym.register(id="Tetris-v1", entry_point=MoveEnv)
