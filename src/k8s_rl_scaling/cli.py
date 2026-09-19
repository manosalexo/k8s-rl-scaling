"""CLI entry point for training RL agents on Kubernetes scaling."""

import argparse
import sys

from k8s_rl_scaling.config import load_config
from k8s_rl_scaling.agents import QLearningAgent, DynaQAgent
from k8s_rl_scaling.environments import HPAEnv, VPAEnv
from k8s_rl_scaling.metrics import MetricsCollector, JMeterRunner
from k8s_rl_scaling.training import Trainer


ALGORITHMS = {"q-learning", "dyna-q"}
SCALERS = {"hpa", "vpa"}


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

    collector = MetricsCollector(
        hostname=cfg["ssh"]["hostname"],
        port=cfg["ssh"]["port"],
        username=cfg["ssh"]["username"],
        password=cfg["ssh"]["password"],
        timeout=cfg["ssh"]["timeout"],
        prometheus_endpoint=cfg["prometheus"]["endpoint"],
        total_cpu_cores=cfg["prometheus"]["total_cpu_cores"],
        total_memory_gb=cfg["prometheus"]["total_memory_gb"],
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

    agent_cfg = cfg["agent"]
    common_kwargs = dict(
        action_size=env.action_space.n,
        num_bins=cfg["latency"]["num_bins"],
        bin_width=cfg["latency"]["bin_width"],
        learning_rate=agent_cfg["learning_rate"],
        discount_factor=agent_cfg["discount_factor"],
        exploration_rate=agent_cfg["exploration_rate"],
        exploration_decay=agent_cfg["exploration_decay"],
        min_exploration_rate=agent_cfg["min_exploration_rate"],
    )

    if args.algorithm == "q-learning":
        agent = QLearningAgent(**common_kwargs)
    else:
        agent = DynaQAgent(
            planning_steps=cfg["dyna_q"]["planning_steps"],
            **common_kwargs,
        )

    num_episodes = args.episodes or cfg["training"]["num_episodes"]
    output = args.output or f"outputs/metrics_{args.algorithm}_{args.scaler}.csv"

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
