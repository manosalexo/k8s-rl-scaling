"""CLI entry point for training RL agents on Kubernetes scaling."""

import argparse
import sys

from k8s_rl_scaling.config import load_config
from k8s_rl_scaling.agents import QLearningAgent, DynaQAgent, DynaQPlusAgent
from k8s_rl_scaling.environments import HPAEnv, VPAEnv
from k8s_rl_scaling.metrics import MetricsCollector, JMeterRunner
from k8s_rl_scaling.training import Trainer


ALGORITHMS = {"q-learning", "dyna-q", "dyna-q-plus", "dqn"}
SCALERS = {"hpa", "vpa"}


def _build_tabular_agent(algorithm: str, cfg: dict, action_size: int):
    agent_cfg = cfg["agent"]
    common = dict(
        action_size=action_size,
        num_bins=cfg["latency"]["num_bins"],
        bin_width=cfg["latency"]["bin_width"],
        learning_rate=agent_cfg["learning_rate"],
        discount_factor=agent_cfg["discount_factor"],
        exploration_rate=agent_cfg["exploration_rate"],
        exploration_decay=agent_cfg["exploration_decay"],
        min_exploration_rate=agent_cfg["min_exploration_rate"],
    )

    if algorithm == "q-learning":
        return QLearningAgent(**common)
    if algorithm == "dyna-q":
        return DynaQAgent(
            planning_steps=cfg["dyna_q"]["planning_steps"],
            **common,
        )
    if algorithm == "dyna-q-plus":
        return DynaQPlusAgent(
            planning_steps=cfg["dyna_q_plus"]["planning_steps"],
            kappa=cfg["dyna_q_plus"]["kappa"],
            **common,
        )
    raise ValueError(f"Unknown tabular algorithm: {algorithm}")


def _build_dqn_agent(cfg: dict, state_dim: int, action_size: int):
    try:
        from k8s_rl_scaling.agents.dqn import DQNAgent
    except ImportError:
        print(
            "Error: DQN requires PyTorch. Install with: pip install torch",
            file=sys.stderr,
        )
        sys.exit(1)

    agent_cfg = cfg["agent"]
    dqn_cfg = cfg["dqn"]
    return DQNAgent(
        state_dim=state_dim,
        action_size=action_size,
        hidden_dim=dqn_cfg["hidden_dim"],
        learning_rate=dqn_cfg.get("learning_rate", agent_cfg["learning_rate"]),
        discount_factor=agent_cfg["discount_factor"],
        exploration_rate=agent_cfg["exploration_rate"],
        exploration_decay=agent_cfg["exploration_decay"],
        min_exploration_rate=agent_cfg["min_exploration_rate"],
        buffer_capacity=dqn_cfg["buffer_capacity"],
        batch_size=dqn_cfg["batch_size"],
        target_update_freq=dqn_cfg["target_update_freq"],
    )


def main():
    parser = argparse.ArgumentParser(
        description="Train an RL agent for Kubernetes auto-scaling"
    )
    parser.add_argument(
        "--algorithm",
        choices=sorted(ALGORITHMS),
        required=True,
        help="RL algorithm to use",
    )
    parser.add_argument(
        "--scaler",
        choices=sorted(SCALERS),
        required=True,
        help="Scaling strategy (hpa or vpa)",
    )
    parser.add_argument(
        "--config",
        default="config/default.yaml",
        help="Path to YAML config file",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=None,
        help="Override number of training episodes",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Path for metrics CSV output",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)

    prom_cfg = cfg["prometheus"]
    collector = MetricsCollector(
        prometheus_url=prom_cfg["url"],
        total_cpu_cores=prom_cfg["total_cpu_cores"],
        total_memory_gb=prom_cfg["total_memory_gb"],
        timeout=prom_cfg.get("timeout", 10),
    )

    jmeter_cfg = cfg["jmeter"]
    jmeter = JMeterRunner(
        jmeter_path=jmeter_cfg["path"],
        jmx_template=jmeter_cfg["jmx_template"],
        output_dir=jmeter_cfg["output_dir"],
        output_file=jmeter_cfg["output_file"],
    )

    latency_fn = jmeter.get_last_latency

    if args.scaler == "hpa":
        env = HPAEnv(
            namespace=cfg["kubernetes"]["namespace"],
            deployment_name=cfg["kubernetes"]["deployment_name"],
            min_replicas=cfg["hpa"]["min_replicas"],
            max_replicas=cfg["hpa"]["max_replicas"],
            initial_replicas=cfg["hpa"]["initial_replicas"],
            latency_high=cfg["latency"]["threshold_high"],
            latency_low=cfg["latency"]["threshold_low"],
            max_latency=cfg["latency"]["max_value"],
            latency_source=latency_fn,
            kubeconfig_path=cfg["kubernetes"].get("kubeconfig_path"),
            reward_cfg=cfg["training"]["reward"],
        )
    else:
        env = VPAEnv(
            namespace=cfg["kubernetes"]["namespace"],
            deployment_name=cfg["kubernetes"]["deployment_name"],
            container_name=cfg["vpa"]["container_name"],
            min_cpu=cfg["vpa"]["min_cpu"],
            max_cpu=cfg["vpa"]["max_cpu"],
            initial_cpu=cfg["vpa"]["initial_cpu"],
            cpu_step=cfg["vpa"]["cpu_step"],
            min_memory=cfg["vpa"]["min_memory"],
            max_memory=cfg["vpa"]["max_memory"],
            initial_memory=cfg["vpa"]["initial_memory"],
            memory_step=cfg["vpa"]["memory_step"],
            latency_high=cfg["latency"]["threshold_high"],
            latency_low=cfg["latency"]["threshold_low"],
            max_latency=cfg["latency"]["max_value"],
            latency_source=latency_fn,
            kubeconfig_path=cfg["kubernetes"].get("kubeconfig_path"),
            reward_cfg=cfg["training"]["reward"],
        )

    if args.algorithm == "dqn":
        agent = _build_dqn_agent(
            cfg,
            state_dim=env.observation_space.shape[0],
            action_size=env.action_space.n,
        )
    else:
        agent = _build_tabular_agent(
            args.algorithm, cfg, action_size=env.action_space.n,
        )

    num_episodes = args.episodes or cfg["training"]["num_episodes"]
    output = (
        args.output
        or f"outputs/metrics_{args.algorithm}_{args.scaler}.csv"
    )

    trainer = Trainer(
        env=env,
        agent=agent,
        jmeter=jmeter,
        metrics_collector=collector,
        num_episodes=num_episodes,
        max_steps=cfg["training"]["max_steps_per_episode"],
        output_csv=output,
    )

    try:
        trainer.train()
    finally:
        env.close()


if __name__ == "__main__":
    main()
