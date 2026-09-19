"""Configuration loader with environment variable expansion."""

import os
import re
import yaml


def _expand_env_vars(value):
    if isinstance(value, str):
        return re.sub(
            r"\$\{(\w+)\}",
            lambda m: os.environ.get(m.group(1), m.group(0)),
            value,
        )
    if isinstance(value, dict):
        return {k: _expand_env_vars(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_expand_env_vars(v) for v in value]
    return value


def load_config(path: str = "config/default.yaml") -> dict:
    with open(path) as f:
        raw = yaml.safe_load(f)
    return _expand_env_vars(raw)
