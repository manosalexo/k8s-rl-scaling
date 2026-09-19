#!/usr/bin/env python3
"""Compare training results across all algorithm-scaler combinations.

Usage:
    python scripts/compare_results.py --input-dir outputs/
"""

import argparse
import os

import matplotlib.pyplot as plt
import pandas as pd

PALETTE = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
    "#9467bd", "#8c564b", "#e377c2", "#17becf",
]


def load_results(input_dir: str) -> dict[tuple[str, str], pd.DataFrame]:
    results = {}
    if not os.path.isdir(input_dir):
        return results
    for f in sorted(os.listdir(input_dir)):
        if f.startswith("metrics_") and f.endswith(".csv"):
            stem = f[len("metrics_"):-len(".csv")]
            parts = stem.rsplit("_", 1)
            if len(parts) == 2 and parts[1] in ("hpa", "vpa"):
                algo, scaler = parts
                path = os.path.join(input_dir, f)
                results[(algo, scaler)] = pd.read_csv(path)
                print(f"Loaded {path}")
    return results


def _color_for(idx: int) -> str:
    return PALETTE[idx % len(PALETTE)]


def plot_rewards(results: dict, output_dir: str):
    fig, ax = plt.subplots(figsize=(10, 6))
    for i, ((algo, scaler), df) in enumerate(results.items()):
        rewards = df.groupby("Episode")["Reward"].sum()
        label = f"{algo.upper()} + {scaler.upper()}"
        ax.plot(rewards.index, rewards.values, label=label,
                color=_color_for(i), linewidth=2)
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
    for i, ((algo, scaler), df) in enumerate(results.items()):
        avg_latency = df.groupby("Episode")["Latency"].mean()
        label = f"{algo.upper()} + {scaler.upper()}"
        ax.plot(avg_latency.index, avg_latency.values, label=label,
                color=_color_for(i), linewidth=2)
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

    for i, ((algo, scaler), df) in enumerate(results.items()):
        label = f"{algo.upper()} + {scaler.upper()}"
        color = _color_for(i)
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
