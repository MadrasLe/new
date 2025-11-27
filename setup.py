from setuptools import setup, find_packages

setup(
    name="datastar",
    version="0.1.0",
    description="High-performance GPU-first data loading library",
    packages=find_packages(),
    install_requires=[
        "torch",
        "numpy",
        "pandas",
        "pyarrow",
        "transformers",
        "datasets",
        "tqdm",
    ],
)
