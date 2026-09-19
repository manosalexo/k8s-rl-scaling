from setuptools import setup, find_packages

setup(
    name="k8s-rl-scaling",
    version="1.0.0",
    description="Reinforcement Learning for Kubernetes Auto-Scaling",
    author="Manos Alexopoulos",
    license="MIT",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.10",
    install_requires=[
        "gymnasium>=0.29.0",
        "numpy>=1.23.0",
        "kubernetes>=12.0.0",
        "requests>=2.28.0",
        "pandas>=1.4.0",
        "matplotlib>=3.5.0",
        "pyyaml>=5.4.0",
    ],
    entry_points={
        "console_scripts": [
            "k8s-rl-train=k8s_rl_scaling.cli:main",
        ],
    },
)
