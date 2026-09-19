"""JMeter load-testing integration: launch tests and parse latency results."""

import os
import subprocess
import time


def parse_jmeter_csv(
    file_path: str, start_line: int = 0
) -> tuple[list[int], list[int], int]:
    """Parse JMeter CSV output and return (latencies, load_times, lines_read)."""
    latencies = []
    load_times = []
    total_lines = 0
    with open(file_path) as f:
        lines = f.readlines()[start_line:]
        total_lines = len(lines)
        for line in lines:
            cols = line.strip().split(",")
            if len(cols) > 2:
                load_times.append(int(cols[1]))
                latencies.append(int(cols[-3]))
    return latencies, load_times, total_lines


class JMeterRunner:
    """Manage JMeter load-test execution."""

    def __init__(
        self,
        jmeter_path: str,
        jmx_template: str = "templates/NGINX-load.jmx",
        output_dir: str = "output",
        output_file: str = "jmeterlogs.csv",
    ):
        self.jmeter_bin = os.path.join(jmeter_path, "jmeter")
        self.jmx_path = os.path.join(jmeter_path, jmx_template)
        self.csv_output = os.path.join(jmeter_path, output_dir, output_file)

    def run(self, wait: bool = True) -> subprocess.Popen:
        process = subprocess.Popen(
            [self.jmeter_bin, "-n", "-t", self.jmx_path, "-l", self.csv_output]
        )
        if wait:
            process.wait()
        return process

    def wait_for_output(self, timeout: int = 60):
        start = time.time()
        while time.time() - start < timeout:
            if os.path.exists(self.csv_output) and os.path.getsize(self.csv_output) > 0:
                return
            time.sleep(1)
        raise FileNotFoundError(
            f"JMeter output {self.csv_output} not ready after {timeout}s"
        )

    def get_last_latency(self) -> float:
        """Read the most recent latency value from the JMeter CSV."""
        with open(self.csv_output) as f:
            lines = f.readlines()
            if len(lines) > 1:
                cols = lines[-1].strip().split(",")
                return float(cols[-3])
        return 0.0
