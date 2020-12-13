import os

from setuptools import find_packages, setup

with open(os.path.join("sb3_contrib", "version.txt"), "r") as file_handler:
    __version__ = file_handler.read().strip()


long_description = """

# Stable-Baselines3 - Contrib (SB3-Contrib)

Contrib package for [Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3) - Experimental reinforcement learning (RL) code.
"sb3-contrib" for short.

### What is SB3-Contrib?

A place for RL algorithms and tools that are considered experimental, e.g. implementations of the latest publications. Goal is to keep the simplicity, documentation and style of stable-baselines3 but for less matured implementations.

### Why create this repository?

Over the span of stable-baselines and stable-baselines3, the community has been eager to contribute in form of better logging utilities, environment wrappers, extended support (e.g. different action spaces) and learning algorithms.

However sometimes these utilities were too niche to be considered for stable-baselines or
proved to be too difficult to integrate well into existing code without a mess. sb3-contrib aims to fix this by not requiring the neatest code integration with existing code and not setting limits on what is too niche: almost everything remotely useful goes! We hope this allows to extend the known quality of stable-baselines style and documentation beyond the relatively small scope of utilities of the main repository.


## Features

See documentation for the full list of included features.

**RL Algorithms**:
- [Truncated Quantile Critics (TQC)](https://arxiv.org/abs/2005.04269)


## Documentation

Documentation is available online: [https://sb3-contrib.readthedocs.io/](https://sb3-contrib.readthedocs.io/)


## Installation

**Note:** You need the `master` version of [Stable Baselines3](https://github.com/DLR-RM/stable-baselines3/).

To install Stable Baselines3 `master` version:
```
pip install git+https://github.com/DLR-RM/stable-baselines3
```

Install Stable Baselines3 - Contrib using pip:
```
pip install git+https://github.com/Stable-Baselines-Team/stable-baselines3-contrib

"""  # noqa:E501


setup(
    name="sb3_contrib",
    packages=[package for package in find_packages() if package.startswith("sb3_contrib")],
    package_data={"sb3_contrib": ["py.typed", "version.txt"]},
    install_requires=[
        "stable_baselines3[tests,docs]>=0.11.0a2",
    ],
    description="Contrib package of Stable Baselines3, experimental code.",
    author="Antonin Raffin",
    url="https://github.com/Stable-Baselines-Team/stable-baselines3-contrib",
    author_email="antonin.raffin@dlr.de",
    keywords="reinforcement-learning-algorithms reinforcement-learning machine-learning "
    "gym openai stable baselines toolbox python data-science",
    license="MIT",
    long_description=long_description,
    long_description_content_type="text/markdown",
    version=__version__,
)

# python setup.py sdist
# python setup.py bdist_wheel
# twine upload --repository-url https://test.pypi.org/legacy/ dist/*
# twine upload dist/*
