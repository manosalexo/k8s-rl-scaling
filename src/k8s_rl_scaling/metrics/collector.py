"""Collect CPU and memory metrics from Prometheus via SSH tunnel."""

import json
from urllib.parse import quote

import numpy as np
import paramiko


class MetricsCollector:
    """Fetches cluster metrics from Prometheus running on a remote node via SSH."""

    def __init__(
        self,
        hostname: str,
        port: int = 22,
        username: str = "user",
        password: str | None = None,
        timeout: int = 10,
        prometheus_endpoint: str = "http://localhost:9091",
        total_cpu_cores: int = 8,
        total_memory_gb: int = 8,
    ):
        self.hostname = hostname
        self.port = port
        self.username = username
        self.password = password
        self.timeout = timeout
        self.prometheus_endpoint = prometheus_endpoint
        self.total_cpu_cores = total_cpu_cores
        self.total_memory_gb = total_memory_gb

    def _connect(self) -> paramiko.SSHClient:
        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        ssh.connect(
            hostname=self.hostname,
            port=self.port,
            username=self.username,
            password=self.password,
            timeout=self.timeout,
        )
        return ssh

    def _query_prometheus(self, ssh: paramiko.SSHClient, query: str) -> dict:
        encoded_query = quote(query, safe="")
        url = f"{self.prometheus_endpoint}/api/v1/query?query={encoded_query}"
        cmd = f"curl -sf '{url}'"
        _, stdout, stderr = ssh.exec_command(cmd)
        out = stdout.read().decode("utf-8")
        err = stderr.read().decode("utf-8")
        if not out:
            raise RuntimeError(
                f"Prometheus query returned no output. "
                f"Query: {query}, stderr: {err}"
            )
        data = json.loads(out)
        if data.get("status") != "success":
            raise RuntimeError(f"Prometheus query failed: {data}")
        return data

    def collect(self) -> tuple[float, float]:
        """Return (cpu_fraction, memory_gb)."""
        ssh = self._connect()
        try:
            cpu_data = self._query_prometheus(
                ssh, "sum(rate(container_cpu_usage_seconds_total[1m]))"
            )
            mem_data = self._query_prometheus(
                ssh, "sum(container_memory_usage_bytes)"
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
        finally:
            ssh.close()

    def collect_as_array(self) -> np.ndarray:
        cpu, mem = self.collect()
        return np.array([cpu, mem], dtype=np.float32)
