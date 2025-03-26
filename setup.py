# Copyright (c) 2025, Tencent Inc. All rights reserved.
# Data: 2025/1/9 21:16
# Author: chenchenqin
from setuptools import setup, find_packages

with open("README.md") as f:
    long_description = f.read()

version = "0.1"

extras = {
    "tgnn": [pkg.strip() for pkg in open("requirements.txt").read().split("\n") if pkg.strip()]
}

setup(
    name="tgnn",
    version=version,
    description="Tencent Genomics Neural Network Library",
    long_description=long_description,
    author="Tencent AI For Life Science Lab",
    author_email="chenchenqin@tencent.com",
    packages=find_packages(),
    extras_require=extras,
    zip_safe=True
)
