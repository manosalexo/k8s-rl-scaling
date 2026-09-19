# Setup Guide

Complete guide to setting up the infrastructure needed to run RL-based Kubernetes scaling experiments.

## Table of Contents

1. [Kubernetes Cluster](#1-kubernetes-cluster)
2. [Prometheus Monitoring Stack](#2-prometheus-monitoring-stack)
3. [Grafana Dashboards](#3-grafana-dashboards)
4. [NGINX Target Workload](#4-nginx-target-workload)
5. [Apache JMeter Load Testing](#5-apache-jmeter-load-testing)
6. [Python Environment](#6-python-environment)
7. [Configuration](#7-configuration)
8. [Running Experiments](#8-running-experiments)

---

## 1. Kubernetes Cluster

The experiments were originally conducted on **MicroK8s** running on an Okeanos IaaS VM (4 vCPU, 8 GB RAM, Ubuntu 22.04). Any Kubernetes distribution works.

### MicroK8s Installation

```bash
sudo snap install microk8s --classic --channel=1.28/stable
sudo usermod -aG microk8s $USER
newgrp microk8s

# Wait for cluster to be ready
microk8s status --wait-ready

# Enable required add-ons
microk8s enable dns storage metrics-server

# Create namespace for experiments
microk8s kubectl create namespace monitoring

# Export kubeconfig for the Python kubernetes client
mkdir -p ~/.kube
microk8s config > ~/.kube/config
```

### Verify Cluster

```bash
microk8s kubectl get nodes
microk8s kubectl get pods -n kube-system
```

---

## 2. Prometheus Monitoring Stack

Prometheus collects container-level CPU and memory metrics via cAdvisor (built into kubelet).

### Option A: Helm (recommended)

```bash
# Install Helm if not already available
snap install helm --classic

# Add Prometheus community chart repo
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm repo update

# Install kube-prometheus-stack (includes Prometheus, Grafana, node-exporter, kube-state-metrics)
helm install monitoring prometheus-community/kube-prometheus-stack \
  --namespace monitoring \
  --set prometheus.service.type=NodePort \
  --set prometheus.service.nodePort=30090 \
  --set grafana.service.type=NodePort \
  --set grafana.service.nodePort=30091 \
  --set grafana.adminPassword=admin
```

### Option B: Manual Manifests

Apply the manifests included in this repo:

```bash
kubectl apply -f k8s/monitoring/
```

This deploys:
- **Prometheus** (port 9090, or 9091 via port-forward)
- **ServiceAccount + ClusterRole** for scraping kubelet/cAdvisor
- **ConfigMap** with scrape configuration

### Port Forwarding (for SSH-tunneled access)

The RL training scripts access Prometheus from a local machine through an SSH tunnel. On the Kubernetes node:

```bash
# Expose Prometheus on localhost:9091
kubectl port-forward -n monitoring svc/prometheus 9091:9090 --address=0.0.0.0 &
```

Then from your local machine:

```bash
# SSH tunnel — forwards local:9091 to remote:9091
ssh -L 9091:localhost:9091 user@your-k8s-node
```

### Verify Prometheus

```bash
# On the K8s node (or via tunnel)
curl 'http://localhost:9091/api/v1/query?query=up'

# Check that cAdvisor metrics are being scraped
curl 'http://localhost:9091/api/v1/query?query=container_cpu_usage_seconds_total' | python3 -m json.tool | head -20
```

### Key PromQL Queries Used by the RL Agent

| Metric | Query |
|--------|-------|
| CPU usage (rate) | `sum(rate(container_cpu_usage_seconds_total[1m]))` |
| Memory usage (bytes) | `sum(container_memory_usage_bytes)` |

---

## 3. Grafana Dashboards

Grafana provides real-time visualization during training. It's included in the Helm kube-prometheus-stack.

### Access

```bash
# If using Helm with NodePort (as above)
# Grafana URL: http://<node-ip>:30091
# Default login: admin / admin

# If using port-forward instead:
kubectl port-forward -n monitoring svc/monitoring-grafana 3000:80 &
# Then: http://localhost:3000
```

### Recommended Dashboards

Import these dashboards from grafana.com:

| Dashboard | ID | Description |
|-----------|----|-------------|
| Kubernetes Cluster Monitoring | `315` | Node-level CPU/RAM/disk |
| Kubernetes Pod Monitoring | `6417` | Per-pod resource usage |
| NGINX Ingress Controller | `9614` | Request rate, latency |

To import: Grafana → Dashboards → Import → Enter ID → Load → Select Prometheus data source → Import.

### Custom RL Training Dashboard

Create a dashboard with these panels to monitor training in real time:

**Panel 1: CPU Usage (Gauge)**
```promql
sum(rate(container_cpu_usage_seconds_total{namespace="monitoring", pod=~"nginx-deployment.*"}[1m]))
```

**Panel 2: Memory Usage (Gauge)**
```promql
sum(container_memory_usage_bytes{namespace="monitoring", pod=~"nginx-deployment.*"}) / (1024^3)
```

**Panel 3: Pod Count (Stat)**
```promql
count(kube_pod_info{namespace="monitoring", pod=~"nginx-deployment.*"})
```

**Panel 4: Container CPU Requests (Time Series)**
```promql
sum(kube_pod_container_resource_requests{namespace="monitoring", resource="cpu", pod=~"nginx-deployment.*"})
```

---

## 4. NGINX Target Workload

Deploy the NGINX workload that the RL agent will scale:

```bash
kubectl apply -f k8s/nginx-deployment.yaml
```

This creates:
- **Deployment** `nginx-deployment` in `monitoring` namespace (3 replicas, 100m CPU / 512Mi memory requests)
- **Service** `nginx-service` (ClusterIP on port 80)

### Verify

```bash
kubectl get deployment nginx-deployment -n monitoring
kubectl get pods -n monitoring -l app=nginx
kubectl get svc nginx-service -n monitoring
```

### Expose for JMeter

JMeter needs to reach the NGINX service. Options:

```bash
# Option 1: NodePort
kubectl patch svc nginx-service -n monitoring -p '{"spec":{"type":"NodePort","ports":[{"port":80,"nodePort":30080}]}}'
# JMeter target: http://<node-ip>:30080

# Option 2: Port-forward (simpler for single-node)
kubectl port-forward -n monitoring svc/nginx-service 8080:80 --address=0.0.0.0 &
# JMeter target: http://localhost:8080
```

---

## 5. Apache JMeter Load Testing

JMeter generates HTTP load against the NGINX deployment. The RL agent uses observed latency from JMeter's output to make scaling decisions.

### Installation

```bash
# Download JMeter 5.5
wget https://archive.apache.org/dist/jmeter/binaries/apache-jmeter-5.5.tgz
tar -xzf apache-jmeter-5.5.tgz
mv apache-jmeter-5.5 ~/apache-jmeter-5.5

# Verify
~/apache-jmeter-5.5/bin/jmeter --version
```

### JMeter Test Plan

Create a test plan (`.jmx`) file or use the provided template. The test plan should:

1. **Thread Group**: Configure number of concurrent users (250, 500, 1000, 2000, or 4000)
2. **HTTP Request**: Target the NGINX service endpoint
3. **Listener**: Write results to CSV format

Example test plan structure (save as `~/apache-jmeter-5.5/bin/templates/NGINX-load.jmx`):

```xml
<ThreadGroup>
  <stringProp name="ThreadGroup.num_threads">1000</stringProp>
  <stringProp name="ThreadGroup.ramp_time">10</stringProp>
  <stringProp name="ThreadGroup.duration">60</stringProp>
</ThreadGroup>

<HTTPSamplerProxy>
  <stringProp name="HTTPSampler.domain">localhost</stringProp>
  <stringProp name="HTTPSampler.port">30080</stringProp>
  <stringProp name="HTTPSampler.path">/</stringProp>
  <stringProp name="HTTPSampler.method">GET</stringProp>
</HTTPSamplerProxy>
```

### Run JMeter Manually

```bash
mkdir -p ~/apache-jmeter-5.5/bin/output

# Run with 1000 concurrent users
~/apache-jmeter-5.5/bin/jmeter -n \
  -t ~/apache-jmeter-5.5/bin/templates/NGINX-load.jmx \
  -l ~/apache-jmeter-5.5/bin/output/jmeterlogs.csv

# Check output
head -5 ~/apache-jmeter-5.5/bin/output/jmeterlogs.csv
```

### JMeter CSV Format

The output CSV has this structure (columns used by the RL agent are marked):

| Column | Name | Used |
|--------|------|------|
| 0 | timeStamp | - |
| 1 | elapsed | **Load Time** |
| ... | ... | - |
| -3 | Latency | **Latency** |
| -2 | IdleTime | - |
| -1 | Connect | - |

---

## 6. Python Environment

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install the package in development mode
pip install -e .

# Or just install dependencies
pip install -r requirements.txt
```

---

## 7. Configuration

### Environment Variables

Set SSH credentials for accessing the Kubernetes node:

```bash
export SSH_HOSTNAME="your-k8s-node-hostname"
export SSH_USERNAME="your-ssh-username"
export SSH_PASSWORD="your-ssh-password"
```

### Config File

The default configuration is in `config/default.yaml`. To customize without modifying the tracked file:

```bash
cp config/default.yaml config/local.yaml
# Edit config/local.yaml with your values
# config/local.yaml is gitignored
```

Key parameters to adjust:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `ssh.hostname` | `${SSH_HOSTNAME}` | K8s node SSH address |
| `kubernetes.kubeconfig_path` | `~/.kube/config` | Path to kubeconfig |
| `latency.threshold_high` | `10` ms | Scale-up threshold |
| `latency.threshold_low` | `5` ms | Scale-down threshold |
| `training.num_episodes` | `10` | Training episodes |
| `training.max_steps_per_episode` | `10` | Steps per episode |
| `dyna_q.planning_steps` | `5` | Dyna-Q planning iterations |
| `jmeter.path` | `~/apache-jmeter-5.5/bin` | JMeter installation path |

---

## 8. Running Experiments

### Single Experiment

```bash
# Q-Learning with HPA scaling
k8s-rl-train --algorithm q-learning --scaler hpa

# Dyna-Q with VPA scaling
k8s-rl-train --algorithm dyna-q --scaler vpa

# Custom config and more episodes
k8s-rl-train --algorithm dyna-q --scaler hpa \
  --config config/local.yaml \
  --episodes 20 \
  --output outputs/dyna-q-hpa-20ep.csv
```

### Full Experiment Suite

Run all 4 combinations:

```bash
make train-all
```

Or manually:

```bash
for algo in q-learning dyna-q; do
  for scaler in hpa vpa; do
    echo "=== Training: $algo + $scaler ==="
    k8s-rl-train --algorithm $algo --scaler $scaler \
      --output outputs/metrics_${algo}_${scaler}.csv
  done
done
```

### Compare Results

```bash
python scripts/compare_results.py --input-dir outputs/ --output-dir outputs/plots/
```

This generates comparison plots:
- `reward_comparison.png` — Cumulative reward per episode across all combinations
- `latency_comparison.png` — Average latency per episode
- `resource_comparison.png` — CPU and RAM usage over time
