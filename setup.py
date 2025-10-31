"""
Setup script za instaliranje paketa.
"""
from setuptools import setup, find_packages

setup(
    name="hvac-federated-control",
    version="1.0.0",
    description="Federated Learning for HVAC Control using Actor System",
    author="Strahinja Galic, Mihajlo Sremac",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "asyncio",
        "scikit-learn>=1.3.0", 
        "numpy>=1.21.0",
        "matplotlib>=3.5.0",
        "pandas>=1.3.0",
        "aiohttp>=3.8.0",
        "websockets>=11.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-asyncio>=0.21.0", 
            "black",
            "flake8",
        ]
    },
    entry_points={
        "console_scripts": [
            "hvac-demo=src.simulation.demo:main",
        ]
    },
)