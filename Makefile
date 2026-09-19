.PHONY: install setup-monitoring deploy-workload train-all train-dqn compare clean help

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install: ## Install Python package in development mode
	pip install -e .

setup-monitoring: ## Deploy Prometheus + Grafana to Kubernetes
	kubectl apply -f k8s/monitoring/namespace.yaml
	kubectl apply -f k8s/monitoring/prometheus-rbac.yaml
	kubectl apply -f k8s/monitoring/prometheus-config.yaml
	kubectl apply -f k8s/monitoring/prometheus-deployment.yaml
	kubectl apply -f k8s/monitoring/grafana-deployment.yaml
	@echo ""
	@echo "Waiting for pods to be ready..."
	kubectl wait --for=condition=ready pod -l app=prometheus -n monitoring --timeout=120s
	kubectl wait --for=condition=ready pod -l app=grafana -n monitoring --timeout=120s
	@echo ""
	@echo "Monitoring stack deployed."
	@echo "  Prometheus: kubectl port-forward -n monitoring svc/prometheus 9091:9090"
	@echo "  Grafana:    kubectl port-forward -n monitoring svc/grafana 3000:3000"

deploy-workload: ## Deploy NGINX target workload
	kubectl apply -f k8s/nginx-deployment.yaml
	kubectl wait --for=condition=ready pod -l app=nginx -n monitoring --timeout=60s
	@echo "NGINX deployment ready."

port-forward: ## Start port-forward for Prometheus (9091) and Grafana (3000)
	@echo "Starting port-forwards (Ctrl+C to stop)..."
	kubectl port-forward -n monitoring svc/prometheus 9091:9090 &
	kubectl port-forward -n monitoring svc/grafana 3000:3000 &
	@echo "Prometheus: http://localhost:9091"
	@echo "Grafana:    http://localhost:3000 (admin/admin)"

train-all: ## Run all tabular algorithm+scaler combinations
	@mkdir -p outputs
	k8s-rl-train --algorithm q-learning --scaler hpa --output outputs/metrics_q-learning_hpa.csv
	k8s-rl-train --algorithm q-learning --scaler vpa --output outputs/metrics_q-learning_vpa.csv
	k8s-rl-train --algorithm dyna-q --scaler hpa --output outputs/metrics_dyna-q_hpa.csv
	k8s-rl-train --algorithm dyna-q --scaler vpa --output outputs/metrics_dyna-q_vpa.csv
	k8s-rl-train --algorithm dyna-q-plus --scaler hpa --output outputs/metrics_dyna-q-plus_hpa.csv
	k8s-rl-train --algorithm dyna-q-plus --scaler vpa --output outputs/metrics_dyna-q-plus_vpa.csv

train-dqn: ## Run DQN combinations (requires: pip install -e '.[dqn]')
	@mkdir -p outputs
	k8s-rl-train --algorithm dqn --scaler hpa --output outputs/metrics_dqn_hpa.csv
	k8s-rl-train --algorithm dqn --scaler vpa --output outputs/metrics_dqn_vpa.csv

compare: ## Generate comparison plots from training results
	python scripts/compare_results.py --input-dir outputs/ --output-dir outputs/plots/

clean: ## Remove training outputs
	rm -rf outputs/

setup-all: install setup-monitoring deploy-workload ## Full setup: install + monitoring + workload
	@echo ""
	@echo "Setup complete. Run 'make port-forward' then 'make train-all'."
