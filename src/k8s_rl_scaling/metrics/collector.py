"""Collect CPU and memory metrics from Prometheus via HTTP."""

from urllib.parse import quote

import numpy as np
import requests


class MetricsCollector:
    """Fetches cluster metrics from Prometheus via its HTTP API.

    Works with any Prometheus endpoint reachable over HTTP — whether via
    kubectl port-forward (localhost:9091), an in-cluster service DNS
    (http://prometheus.monitoring.svc:9090), or an external URL.
    """

    def __init__(
        self,
        prometheus_url: str = "http://localhost:9091",
        total_cpu_cores: int = 8,
        total_memory_gb: int = 8,
        timeout: int = 10,
    ):
        self.prometheus_url = prometheus_url.rstrip("/")
        self.total_cpu_cores = total_cpu_cores
        self.total_memory_gb = total_memory_gb
        self.timeout = timeout

    def _query_prometheus(self, query: str) -> dict:
        encoded_query = quote(query, safe="")
        url = f"{self.prometheus_url}/api/v1/query?query={encoded_query}"
        resp = requests.get(url, timeout=self.timeout)
        resp.raise_for_status()
        data = resp.json()
        if data.get("status") != "success":
            raise RuntimeError(f"Prometheus query failed: {data}")
        return data

    def collect(self) -> tuple[float, float]:
        """Return (cpu_fraction, memory_gb)."""
        cpu_data = self._query_prometheus(
            "sum(rate(container_cpu_usage_seconds_total[1m]))"
        )
        mem_data = self._query_prometheus(
            "sum(container_memory_usage_bytes)"
        )

        cpu_results = cpu_data["data"]["result"]
        mem_results = mem_data["data"]["result"]

        if not cpu_results or not mem_results:
            raise RuntimeError(
                "Prometheus returned empty results — "
                "check that cAdvisor metrics are being scraped"
            )

        cpu_usage = float(cpu_results[0]["value"][1])
        mem_usage = float(mem_results[0]["value"][1])

        cpu_fraction = cpu_usage / self.total_cpu_cores
        memory_gb = mem_usage / (1024**3)

        return cpu_fraction, memory_gb

    def collect_as_array(self) -> np.ndarray:
        cpu, mem = self.collect()
        return np.array([cpu, mem], dtype=np.float32)
