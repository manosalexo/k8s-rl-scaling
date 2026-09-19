"""Q-Learning agent with tabular Q-table and epsilon-greedy exploration."""

import random
import numpy as np


class QLearningAgent:
    def __init__(
        self,
        action_size: int,
        num_bins: int = 5,
        learning_rate: float = 0.05,
        discount_factor: float = 0.9,
        exploration_rate: float = 1.0,
        exploration_decay: float = 0.999,
        min_exploration_rate: float = 0.01,
        bin_width: float = 2000.0,
    ):
        self.action_size = action_size
        self.num_bins = num_bins
        self.bin_width = bin_width
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.exploration_decay = exploration_decay
        self.min_exploration_rate = min_exploration_rate
        self.q_table = np.zeros((num_bins, action_size))

    def discretize_state(self, state: np.ndarray) -> int:
        latency = float(state[0])
        return min(int(latency / self.bin_width), self.num_bins - 1)

    def choose_action(self, state: np.ndarray) -> int:
        if np.random.rand() <= self.exploration_rate:
            return random.randrange(self.action_size)
        idx = self.discretize_state(state)
        return int(np.argmax(self.q_table[idx]))

    def learn(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
    ):
        idx = self.discretize_state(state)
        next_idx = self.discretize_state(next_state)
        best_next = np.max(self.q_table[next_idx])
        td_target = reward + self.discount_factor * best_next
        td_error = td_target - self.q_table[idx][action]
        self.q_table[idx][action] += self.learning_rate * td_error

    def decay_exploration(self):
        self.exploration_rate = max(
            self.min_exploration_rate,
            self.exploration_rate * self.exploration_decay,
        )
