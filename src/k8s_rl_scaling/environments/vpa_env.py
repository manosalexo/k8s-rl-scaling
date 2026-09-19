"""Gymnasium environment for Kubernetes VPA (Vertical Pod Autoscaler) scaling."""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from kubernetes import client, config


class VPAEnv(gym.Env):
    """Scale a Kubernetes Deployment's container resource requests based on latency.

    Actions: 0 = scale down, 1 = no change, 2 = scale up.
    Observation: latency in milliseconds (shape (1,), range [0, max_latency]).
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        namespace: str = "monitoring",
        deployment_name: str = "nginx-deployment",
        container_name: str = "nginx",
        min_cpu: float = 0.025,
        max_cpu: float = 1.0,
        initial_cpu: float = 0.1,
        cpu_step: float = 0.1,
        min_memory: int = 256,
        max_memory: int = 1024,
        initial_memory: int = 512,
        memory_step: int = 128,
        max_latency: float = 10000.0,
        latency_high: float = 10.0,
        latency_low: float = 5.0,
        latency_source=None,
        kubeconfig_path: str | None = None,
        reward_cfg: dict | None = None,
    ):
        super().__init__()

        self.namespace = namespace
        self.deployment_name = deployment_name
        self.container_name = container_name
        self.min_cpu = min_cpu
        self.max_cpu = max_cpu
        self.initial_cpu = initial_cpu
        self.cpu_step = cpu_step
        self.min_memory = min_memory
        self.max_memory = max_memory
        self.initial_memory = initial_memory
        self.memory_step = memory_step
        self.current_cpu = initial_cpu
        self.current_memory = initial_memory
        self.latency_high = latency_high
        self.latency_low = latency_low
        self.latency_source = latency_source

        r = reward_cfg or {}
        self.step_penalty = r.get("step_penalty", -0.1)
        self.optimal_bonus = r.get("optimal_bonus", 2.0)
        self.underload_penalty = r.get("underload_penalty", -0.5)
        self.overload_penalty = r.get("overload_penalty", -1.0)

        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(
            low=0, high=max_latency, shape=(1,), dtype=np.float32
        )

        if kubeconfig_path:
            config.load_kube_config(config_file=kubeconfig_path)
        else:
            try:
                config.load_incluster_config()
            except config.ConfigException:
                config.load_kube_config()

        self.apps_v1 = client.AppsV1Api()

    def _get_latency(self) -> np.ndarray:
        if self.latency_source is not None:
            latency = self.latency_source()
            return np.array(
                [np.clip(latency, 0, self.observation_space.high[0])],
                dtype=np.float32,
            )
        return np.array([0.0], dtype=np.float32)

    def _scale(self, action: int):
        if action == 0:
            self.current_cpu = max(self.min_cpu, self.current_cpu - self.cpu_step)
            self.current_memory = max(
                self.min_memory, self.current_memory - self.memory_step
            )
        elif action == 2:
            self.current_cpu = min(self.max_cpu, self.current_cpu + self.cpu_step)
            self.current_memory = min(
                self.max_memory, self.current_memory + self.memory_step
            )

        cpu_request = f"{int(self.current_cpu * 1000)}m"
        memory_request = f"{self.current_memory}Mi"

        body = {
            "spec": {
                "template": {
                    "spec": {
                        "containers": [
                            {
                                "name": self.container_name,
                                "resources": {
                                    "requests": {
                                        "cpu": cpu_request,
                                        "memory": memory_request,
                                    }
                                },
                            }
                        ]
                    }
                }
            }
        }
        self.apps_v1.patch_namespaced_deployment(
            name=self.deployment_name,
            namespace=self.namespace,
            body=body,
        )

    def _compute_reward(self, latency: float) -> float:
        reward = self.step_penalty
        if self.latency_low <= latency <= self.latency_high:
            reward += self.optimal_bonus
        elif latency < self.latency_low:
            reward += self.underload_penalty
        else:
            reward += self.overload_penalty
        return reward

    def step(self, action: int):
        state = self._get_latency()
        latency = float(state[0])

        if latency > self.latency_high:
            action = 2
        elif latency < self.latency_low:
            action = 0
        else:
            action = 1

        self._scale(action)
        next_state = self._get_latency()
        reward = self._compute_reward(latency)

        return (
            next_state,
            reward,
            False,
            False,
            {"cpu": self.current_cpu, "memory": self.current_memory},
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed, options=options)
        self.current_cpu = self.initial_cpu
        self.current_memory = self.initial_memory
        return self._get_latency(), {}

    def render(self):
        print(
            f"CPU: {int(self.current_cpu * 1000)}m, "
            f"Memory: {self.current_memory}Mi"
        )
