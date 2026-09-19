"""Dyna-Q+ agent: model-based RL with exploration bonus for non-stationary environments.

Extends Dyna-Q by adding a time-based bonus kappa * sqrt(tau) to the
planning reward, where tau is the number of steps since a state-action
pair was last visited in real experience.  This encourages the agent to
re-explore states it has not seen recently — important when the
environment's dynamics change over time (Sutton & Barto, 2018, Sec. 8.3).

During planning, untried actions from visited states are also considered:
they are assumed to transition back to the same state with reward 0, and
their large tau gives them a high exploration bonus.
"""

import random

import numpy as np

from k8s_rl_scaling.agents.q_learning import QLearningAgent


class DynaQPlusAgent(QLearningAgent):
    def __init__(
        self,
        planning_steps: int = 5,
        kappa: float = 0.001,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.planning_steps = planning_steps
        self.kappa = kappa
        self.model: dict[tuple[int, int], tuple[float, int]] = {}
        self.last_visit: dict[tuple[int, int], int] = {}
        self.visited_states: set[int] = set()
        self.time_step = 0

    def learn(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool = False,
    ):
        idx = self.discretize_state(state)
        next_idx = self.discretize_state(next_state)
        self.time_step += 1

        # Direct RL update (no exploration bonus)
        best_next = np.max(self.q_table[next_idx])
        td_target = reward + self.discount_factor * best_next
        self.q_table[idx][action] += self.learning_rate * (
            td_target - self.q_table[idx][action]
        )

        # Update model and visit tracking
        self.model[(idx, action)] = (reward, next_idx)
        self.last_visit[(idx, action)] = self.time_step
        self.visited_states.add(idx)

        # Planning with exploration bonus
        visited = list(self.visited_states)
        for _ in range(self.planning_steps):
            s_idx = random.choice(visited)
            a = random.randrange(self.action_size)

            if (s_idx, a) in self.model:
                r, ns_idx = self.model[(s_idx, a)]
            else:
                r, ns_idx = 0.0, s_idx

            tau = self.time_step - self.last_visit.get((s_idx, a), 0)
            r_plus = r + self.kappa * np.sqrt(tau)

            best_next = np.max(self.q_table[ns_idx])
            td_target = r_plus + self.discount_factor * best_next
            self.q_table[s_idx][a] += self.learning_rate * (
                td_target - self.q_table[s_idx][a]
            )
