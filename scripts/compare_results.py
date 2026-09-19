#!/usr/bin/env python3
"""Compare training results across all 4 algorithm-scaler combinations.

Usage:
    python scripts/compare_results.py --input-dir outputs/
"""

import argparse
import os

import matplotlib.pyplot as plt
import pandas as pd


COMBINATIONS = [
    ("q-learning", "hpa"),
    ("q-learning", "vpa"),
    ("dyna-q", "hpa"),
    ("dyna-q", "vpa"),
]

COLORS = {
    ("q-learning", "hpa"): "#1f77b4",
    ("q-learning", "vpa"): "#ff7f0e",
    ("dyna-q", "hpa"): "#2ca02c",
    ("dyna-q", "vpa"): "#d62728",
}


def load_results(input_dir: str) -> dict[tuple, pd.DataFrame]:
    results = {}
    for algo, scaler in COMBINATIONS:
        path = os.path.join(input_dir, f"metrics_{algo}_{scaler}.csv")
        if os.path.exists(path):
            results[(algo, scaler)] = pd.read_csv(path)
            print(f"Loaded {path}")
        else:
            print(f"Not found: {path} (skipping)")
    return results


def plot_rewards(results: dict, output_dir: str):
    fig, ax = plt.subplots(figsize=(10, 6))
    for (algo, scaler), df in results.items():
        rewards = df.groupby("Episode")["Reward"].sum()
        label = f"{algo.upper()} + {scaler.upper()}"
        ax.plot(rewards.index, rewards.values, label=label,
                color=COLORS[(algo, scaler)], linewidth=2)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Total Reward")
    ax.set_title("Cumulative Reward per Episode")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "reward_comparison.png"), dpi=150)
    plt.close(fig)


def plot_latency(results: dict, output_dir: str):
    fig, ax = plt.subplots(figsize=(10, 6))
    for (algo, scaler), df in results.items():
        avg_latency = df.groupby("Episode")["Latency"].mean()
        label = f"{algo.upper()} + {scaler.upper()}"
        ax.plot(avg_latency.index, avg_latency.values, label=label,
                color=COLORS[(algo, scaler)], linewidth=2)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Average Latency (ms)")
    ax.set_title("Average Latency per Episode")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "latency_comparison.png"), dpi=150)
    plt.close(fig)


def plot_resource_usage(results: dict, output_dir: str):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    for (algo, scaler), df in results.items():
        label = f"{algo.upper()} + {scaler.upper()}"
        color = COLORS[(algo, scaler)]
        avg_cpu = df.groupby("Episode")["CPU Usage"].mean()
        avg_ram = df.groupby("Episode")["RAM Usage"].mean()
        ax1.plot(avg_cpu.index, avg_cpu.values, label=label, color=color)
        ax2.plot(avg_ram.index, avg_ram.values, label=label, color=color)

    ax1.set_xlabel("Episode")
    ax1.set_ylabel("CPU Usage (fraction)")
    ax1.set_title("Average CPU Usage")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.set_xlabel("Episode")
    ax2.set_ylabel("RAM Usage (GB)")
    ax2.set_title("Average RAM Usage")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "resource_comparison.png"), dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Compare RL training results")
    parser.add_argument("--input-dir", default="outputs/",
                        help="Directory containing metrics CSVs")
    parser.add_argument("--output-dir", default="outputs/plots/",
                        help="Directory for output plots")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    results = load_results(args.input_dir)

    if not results:
        print("No result files found.")
        return

    plot_rewards(results, args.output_dir)
    plot_latency(results, args.output_dir)
    plot_resource_usage(results, args.output_dir)
    print(f"Plots saved to {args.output_dir}")


if __name__ == "__main__":
    main()
