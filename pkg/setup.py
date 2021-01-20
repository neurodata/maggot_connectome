from setuptools import find_packages, setup

REQUIRED_PACKAGES = [
    "colorcet",
    "matplotlib>=3.3.3",
    "networkx>=2.5",
    "numpy>=1.19",
    "pandas>=1.0",
    "python-catmaid>=2.0.2",
    "scikit-learn>=0.24.0",
    "scipy>=1.6.0",
    "seaborn>=0.11.0",
]

setup(
    name="pkg",
    packages=find_packages(),
    version="0.1.0",
    description="Modeling the Drosophila larva connectome",
    author="Neurodata",
    license="MIT",
    install_requires=REQUIRED_PACKAGES,
    dependency_links=[],
)