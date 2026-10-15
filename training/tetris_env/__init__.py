import gymnasium as gym

from .env import TetrisEnv

__all__ = ["TetrisEnv"]

gym.register(id="Tetris-v0", entry_point=TetrisEnv)
