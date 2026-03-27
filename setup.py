"""
setup.py
────────
Installs the `algo` CLI command system-wide (or in the active venv).

Usage:
    pip install -e .

After installation, run from anywhere:
    algo --papertrade
    algo --simulate --days 10
    algo --uptrade
"""

from setuptools import setup, find_packages

setup(
    name="algo-nifty",
    version="1.0.0",
    description="Production-grade Nifty options algorithmic trading system",
    author="Albin",
    packages=find_packages(exclude=["tests*"]),
    python_requires=">=3.9",
    install_requires=[
        "upstox-python-sdk>=2.0.0",
        "pandas>=1.5.0",
        "numpy>=1.23.0",
        "sqlalchemy>=2.0.0",
        "scikit-learn>=1.2.0",
        "joblib>=1.2.0",
        "requests>=2.28.0",
        "python-dotenv>=1.0.0",
        "click>=8.1.0",
        "rich>=13.0.0",
        "loguru>=0.7.0",
        "schedule>=1.2.0",
        "websocket-client>=1.5.0",
    ],
    entry_points={
        "console_scripts": [
            "algo = main:cli",    # `algo` command maps to the cli() function in main.py
        ],
    },
)
