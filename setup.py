"""Setup script for the Grain de Saga package."""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="grain_de_saga",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="A tiny language model for generating children's stories",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/grain_de_saga",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.13",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.13",
    install_requires=[
        "mlx>=0.24.1",
        "numpy>=1.24.0",
    ],
)
