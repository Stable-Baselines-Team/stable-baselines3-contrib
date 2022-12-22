import os

from setuptools import find_packages, setup

with open(os.path.join("sb3_contrib", "version.txt")) as file_handler:
    __version__ = file_handler.read().strip()


long_description = """

# Stable-Baselines3 - Contrib (SB3-Contrib)

Contrib package for [Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3) - Experimental reinforcement learning (RL) code.
"sb3-contrib" for short.

### What is SB3-Contrib?

A place for RL algorithms and tools that are considered experimental, e.g. implementations of the latest publications. Goal is to keep the simplicity, documentation and style of stable-baselines3 but for less matured implementations.

### Why create this repository?

Over the span of stable-baselines and stable-baselines3, the community has been eager to contribute in form of better logging utilities, environment wrappers, extended support (e.g. different action spaces) and learning algorithms.

However sometimes these utilities were too niche to be considered for stable-baselines or proved to be too difficult to integrate well into the existing code without creating a mess. sb3-contrib aims to fix this by not requiring the neatest code integration with existing code and not setting limits on what is too niche: almost everything remotely useful goes!
We hope this allows us to provide reliable implementations following stable-baselines usual standards (consistent style, documentation, etc) beyond the relatively small scope of utilities in the main repository.


## Features

See documentation for the full list of included features.

**RL Algorithms**:
- [Truncated Quantile Critics (TQC)](https://arxiv.org/abs/2005.04269)
- [Quantile Regression DQN (QR-DQN)](https://arxiv.org/abs/1710.10044)
- [PPO with invalid action masking (MaskablePPO)](https://arxiv.org/abs/2006.14171)
- [Trust Region Policy Optimization (TRPO)](https://arxiv.org/abs/1502.05477)
- [Augmented Random Search (ARS)](https://arxiv.org/abs/1803.07055)

**Gym Wrappers**:
- [Time Feature Wrapper](https://arxiv.org/abs/1712.00378)

## Documentation

Documentation is available online: [https://sb3-contrib.readthedocs.io/](https://sb3-contrib.readthedocs.io/)


## Installation

**Note:** You need the `master` version of [Stable Baselines3](https://github.com/DLR-RM/stable-baselines3/).

To install Stable Baselines3 `master` version:
```
pip install git+https://github.com/DLR-RM/stable-baselines3
```

To install Stable Baselines3 contrib `master` version:
```
pip install git+https://github.com/Stable-Baselines-Team/stable-baselines3-contrib

"""  # noqa:E501


setup(
    name="sb3_contrib",
    packages=[package for package in find_packages() if package.startswith("sb3_contrib")],
    package_data={"sb3_contrib": ["py.typed", "version.txt"]},
    install_requires=[
        "stable_baselines3>=1.7.0a10",
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
    python_requires=">=3.7",
    # PyPI package information.
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
)

# python setup.py sdist
# python setup.py bdist_wheel
# twine upload --repository-url https://test.pypi.org/legacy/ dist/*
# twine upload dist/*
