"""Dyna-Q agent: model-based RL with simulated planning steps."""

import random
from collections import defaultdict

import numpy as np

from k8s_rl_scaling.agents.q_learning import QLearningAgent


class DynaQAgent(QLearningAgent):
    def __init__(self, planning_steps: int = 5, **kwargs):
        super().__init__(**kwargs)
        self.planning_steps = planning_steps
        self.model: dict[int, list[tuple[int, float, int]]] = defaultdict(list)

    def learn(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
    ):
        idx = self.discretize_state(state)
        next_idx = self.discretize_state(next_state)

        # Direct RL update
        best_next = np.max(self.q_table[next_idx])
        td_target = reward + self.discount_factor * best_next
        self.q_table[idx][action] += self.learning_rate * (
            td_target - self.q_table[idx][action]
        )

        # Store experience in model
        self.model[idx].append((action, reward, next_idx))

        # Planning: replay from model
        for _ in range(self.planning_steps):
            s_idx = random.choice(list(self.model.keys()))
            a, r, ns_idx = random.choice(self.model[s_idx])
            best_next = np.max(self.q_table[ns_idx])
            td_target = r + self.discount_factor * best_next
            self.q_table[s_idx][a] += self.learning_rate * (
                td_target - self.q_table[s_idx][a]
            )
