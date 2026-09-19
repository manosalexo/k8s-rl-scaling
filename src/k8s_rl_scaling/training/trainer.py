"""Training loop that coordinates agent, environment, JMeter, and metrics collection."""

import csv
import os
import time

import numpy as np

from k8s_rl_scaling.metrics.collector import MetricsCollector
from k8s_rl_scaling.metrics.jmeter import JMeterRunner, parse_jmeter_csv


class Trainer:
    def __init__(
        self,
        env,
        agent,
        jmeter: JMeterRunner,
        metrics_collector: MetricsCollector,
        num_episodes: int = 10,
        max_steps: int = 10,
        output_csv: str = "training_metrics.csv",
    ):
        self.env = env
        self.agent = agent
        self.jmeter = jmeter
        self.collector = metrics_collector
        self.num_episodes = num_episodes
        self.max_steps = max_steps
        self.output_csv = output_csv

    def _csv_header(self) -> list[str]:
        base = ["Episode", "Step", "Time (s)", "CPU Usage", "RAM Usage",
                "Reward", "Latency", "Load Time"]
        if hasattr(self.env, "current_replicas"):
            base.insert(7, "Replicas")
        return base

    def _csv_row(self, episode, step, elapsed, cpu, ram, reward, latency,
                 load_time) -> list:
        base = [episode, step, f"{elapsed:.2f}", cpu, ram, reward, latency,
                load_time]
        if hasattr(self.env, "current_replicas"):
            base.insert(7, self.env.current_replicas)
        return base

    def train(self) -> list[float]:
        os.makedirs(os.path.dirname(self.output_csv) or ".", exist_ok=True)
        rewards_per_episode = []
        start_time = time.time()
        last_line = 0

        # Run JMeter once before training to produce initial CSV
        self.jmeter.run(wait=True)
        self.jmeter.wait_for_output()

        with open(self.output_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(self._csv_header())

            for episode in range(self.num_episodes):
                if episode > 0:
                    self.jmeter.run(wait=True)
                    self.jmeter.wait_for_output()

                _, load_times, last_line = parse_jmeter_csv(
                    self.jmeter.csv_output, last_line
                )

                state, _ = self.env.reset()
                total_reward = 0.0

                for step in range(self.max_steps):
                    action = self.agent.choose_action(state)
                    next_state, reward, terminated, truncated, info = self.env.step(action)
                    self.agent.learn(state, action, reward, next_state)

                    cpu, ram = self.collector.collect()
                    elapsed = time.time() - start_time
                    lt = load_times[step] if step < len(load_times) else "N/A"

                    writer.writerow(
                        self._csv_row(episode, step, elapsed, cpu, ram,
                                      reward, float(next_state[0]), lt)
                    )

                    state = next_state
                    total_reward += reward

                rewards_per_episode.append(total_reward)

                if hasattr(self.agent, "decay_exploration"):
                    self.agent.decay_exploration()

                print(f"Episode {episode} — total reward: {total_reward:.2f}")

        print(f"Training complete. Metrics saved to {self.output_csv}")
        return rewards_per_episode
