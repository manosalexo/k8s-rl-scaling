"""Configuration loader with environment variable expansion."""

import os
import re
import yaml


def _expand_env_vars(value):
    if isinstance(value, str):
        expanded = re.sub(
            r"\$\{(\w+)\}",
            lambda m: os.environ.get(m.group(1), m.group(0)),
            value,
        )
        return os.path.expanduser(expanded) if expanded.startswith("~") else expanded
    if isinstance(value, dict):
        return {k: _expand_env_vars(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_expand_env_vars(v) for v in value]
    return value  # int, float, bool pass through unchanged


def load_config(path: str = "config/default.yaml") -> dict:
    with open(path) as f:
        raw = yaml.safe_load(f)
    return _expand_env_vars(raw)
