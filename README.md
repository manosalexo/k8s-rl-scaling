# k8s-rl-scaling

Reinforcement Learning for Kubernetes Auto-Scaling — Q-Learning, Dyna-Q, Dyna-Q+, and Deep Q-Network (DQN) agents that learn optimal HPA (Horizontal Pod Autoscaler) and VPA (Vertical Pod Autoscaler) policies based on real-time latency metrics.

This project accompanies a master's thesis at the **University of West Attica**, Department of Informatics and Computer Engineering. It demonstrates how tabular (Q-Learning, Dyna-Q, Dyna-Q+) and deep (DQN) RL algorithms can be applied to the auto-scaling problem in Kubernetes, outperforming static threshold-based approaches.

## Architecture

```
┌──────────────┐     ┌───────────────┐     ┌──────────────────┐
│   JMeter     │────▶│   NGINX Pod   │────▶│   Prometheus     │
│  (load gen)  │     │  (K8s Depl.)  │     │  (metrics)       │
└──────────────┘     └───────────────┘     └────────┬─────────┘
                                                     │
                            ┌────────────────────────┘
                            ▼
┌──────────────────────────────────────────────────────────────┐
│                    RL Training Loop                          │
│                                                              │
│  ┌────────────────┐   ┌───────────────┐   ┌──────────────┐  │
│  │  Gymnasium Env │◀──│  RL Agent     │──▶│  K8s API     │  │
│  │  (HPA or VPA)  │   │  (Q/DQ/DQ+/DQN)│  │  (scale)     │  │
│  └────────────────┘   └───────────────┘   └──────────────┘  │
└──────────────────────────────────────────────────────────────┘
```

### Key Components

| Component | Description |
|-----------|-------------|
| **Agents** | Q-Learning (tabular, model-free), Dyna-Q (tabular, model-based), Dyna-Q+ (Dyna-Q with exploration bonus), DQN (neural network with experience replay) |
| **Environments** | Custom Gymnasium envs for HPA (replica scaling) and VPA (CPU/memory request scaling) |
| **Metrics** | Prometheus HTTP API for CPU/RAM; JMeter CSV parsing for latency |
| **State Space** | Latency — discretized into 5 bins for tabular agents, continuous for DQN |
| **Action Space** | 3 discrete actions: scale down (0), no change (1), scale up (2) |

## Experimental Setup

- **Cluster**: MicroK8s on Okeanos IaaS (4 vCPU, 8 GB RAM, Ubuntu)
- **Target Workload**: NGINX deployment in the `monitoring` namespace
- **Load Testing**: Apache JMeter with 250 / 500 / 1000 / 2000 / 4000 concurrent users
- **Monitoring**: Prometheus + cAdvisor for container-level metrics

### Scaling Strategies

| Strategy | What Scales | Range |
|----------|-------------|-------|
| **HPA** | Pod replicas | 1–10 replicas |
| **VPA** | Container resource requests | CPU: 25m–1000m, Memory: 256Mi–1024Mi |

## Project Structure

```
k8s-rl-scaling/
├── config/
│   └── default.yaml              # All configurable parameters
├── src/k8s_rl_scaling/
│   ├── agents/
│   │   ├── q_learning.py          # Tabular Q-Learning with epsilon-greedy
│   │   ├── dyna_q.py              # Dyna-Q extending Q-Learning with model-based planning
│   │   ├── dyna_q_plus.py         # Dyna-Q+ with time-based exploration bonus (κ√τ)
│   │   └── dqn.py                 # Deep Q-Network with replay buffer and target network
│   ├── environments/
│   │   ├── hpa_env.py             # Gymnasium env — scales replicas via K8s API
│   │   └── vpa_env.py             # Gymnasium env — scales CPU/memory requests via K8s API
│   ├── metrics/
│   │   ├── collector.py           # Prometheus metrics via HTTP API
│   │   └── jmeter.py              # JMeter execution and CSV parsing
│   ├── training/
│   │   └── trainer.py             # Training loop orchestration
│   ├── config.py                  # YAML loader with env-var expansion
│   └── cli.py                     # CLI entry point
├── scripts/
│   └── compare_results.py         # Plot reward/latency/resource comparisons
├── k8s/
│   ├── nginx-deployment.yaml      # Target NGINX deployment + service
│   └── monitoring/                # Full monitoring stack manifests
│       ├── namespace.yaml
│       ├── prometheus-rbac.yaml
│       ├── prometheus-config.yaml
│       ├── prometheus-deployment.yaml
│       └── grafana-deployment.yaml # Grafana + auto-provisioned Prometheus datasource
├── docs/
│   └── setup-guide.md             # Step-by-step infrastructure setup guide
├── Makefile                       # make setup-all, train-all, compare, etc.
├── requirements.txt
├── setup.py
└── LICENSE
```

## Quick Start

### Prerequisites

- Python 3.10+
- Access to a Kubernetes cluster (MicroK8s, minikube, etc.)
- Apache JMeter 5.5+ for load generation

> **Detailed setup instructions** for MicroK8s, Prometheus, Grafana, and JMeter are in [`docs/setup-guide.md`](docs/setup-guide.md).

### Installation

```bash
git clone https://github.com/manosalexo/k8s-rl-scaling.git
cd k8s-rl-scaling
pip install -e .
```

### Full Setup (Makefile)

```bash
make setup-all       # Install Python package + deploy monitoring + NGINX workload
make port-forward    # Start port-forwards for Prometheus (9091) and Grafana (3000)
```

Or step by step:

```bash
make install             # pip install -e .
make setup-monitoring    # Deploy Prometheus + Grafana to K8s
make deploy-workload     # Deploy NGINX target deployment
make port-forward        # Expose Prometheus and Grafana locally
```

### Configuration

The training script connects to Prometheus via HTTP. Make sure Prometheus is reachable:

```bash
# Option 1: kubectl port-forward (local development)
make port-forward    # Exposes Prometheus at http://localhost:9091

# Option 2: In-cluster (script runs as a Pod)
# Set prometheus.url to http://prometheus.monitoring.svc:9090 in config

# Option 3: NodePort/Ingress (remote access)
# Set prometheus.url to the external URL in config
```

Override defaults by creating a `config/local.yaml` (gitignored).

### Train an Agent

```bash
# Q-Learning with HPA
k8s-rl-train --algorithm q-learning --scaler hpa

# Dyna-Q with VPA
k8s-rl-train --algorithm dyna-q --scaler vpa

# Dyna-Q+ with HPA
k8s-rl-train --algorithm dyna-q-plus --scaler hpa

# DQN with HPA (requires PyTorch)
pip install -e '.[dqn]'
k8s-rl-train --algorithm dqn --scaler hpa

# Custom config and episode count
k8s-rl-train --algorithm dyna-q --scaler hpa --config config/local.yaml --episodes 20
```

### Run All Combinations

```bash
make train-all       # All tabular agents (Q-Learning, Dyna-Q, Dyna-Q+)
make train-dqn       # DQN (requires PyTorch)
```

### Compare Results

```bash
make compare
# or:
python scripts/compare_results.py --input-dir outputs/ --output-dir outputs/plots/
```

## Key Findings

From the thesis experiments:

1. **Dyna-Q outperforms Q-Learning** across all load levels, thanks to model-based planning that accelerates policy convergence.
2. **HPA is more effective under high load** (2000+ concurrent users) — adding replicas distributes the load horizontally.
3. **VPA is more effective under low/medium load** (250–1000 users) — increasing per-pod resources avoids the overhead of pod scheduling.
4. **Dyna-Q + HPA** achieves the best overall balance of latency reduction and resource efficiency.

## Hyperparameters

### Tabular Agents (Q-Learning, Dyna-Q, Dyna-Q+)

| Parameter | Value |
|-----------|-------|
| Learning rate (α) | 0.05 |
| Discount factor (γ) | 0.9 |
| Initial exploration (ε) | 1.0 |
| Exploration decay | 0.999 |
| Min exploration | 0.01 |
| Dyna-Q / Dyna-Q+ planning steps | 5 |
| Dyna-Q+ exploration bonus (κ) | 0.001 |
| Latency bins | 5 (0–2s, 2–4s, 4–6s, 6–8s, 8–10s) |

### DQN

| Parameter | Value |
|-----------|-------|
| Learning rate (α) | 0.001 (Adam) |
| Discount factor (γ) | 0.9 |
| Hidden layers | 2 × 64 (ReLU) |
| Replay buffer capacity | 10,000 |
| Batch size | 32 |
| Target network update | every 100 steps |
| Loss | Smooth L1 (Huber) |
| Gradient clipping | max norm 1.0 |

## Reward Function

```
reward = -0.1 (step penalty)
       + 2.0  if latency_low ≤ latency ≤ latency_high    (optimal range)
       - 0.5  if latency < latency_low                    (under-utilization)
       - 1.0  if latency > latency_high                   (overload)
       + 0.5  if replicas == initial_replicas              (stability, HPA only)
```

## License

MIT License — see [LICENSE](LICENSE).

## Citation

If you use this code in your research, please cite:

```
Alexopoulos, M. (2026). Reinforcement Learning for Kubernetes Auto-Scaling:
A Comparative Study of Q-Learning and Dyna-Q for HPA and VPA Optimization.
Master's Thesis, University of West Attica, Department of Informatics and
Computer Engineering.
```
