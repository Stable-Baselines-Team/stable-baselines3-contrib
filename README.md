<img src="docs/\_static/img/logo.png" align="right" width="40%"/>

[![CI](https://github.com/Stable-Baselines-Team/stable-baselines3-contrib/workflows/CI/badge.svg)](https://github.com/Stable-Baselines-Team/stable-baselines3-contrib/actions) [![codestyle](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

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

**Gym Wrappers**:
- [Time Feature Wrapper](https://arxiv.org/abs/1712.00378)


## Documentation

Documentation is available online: [https://sb3-contrib.readthedocs.io/](https://sb3-contrib.readthedocs.io/)


## Installation

To install Stable Baselines3 contrib with pip, execute:

```
pip install sb3-contrib
```

We recommend to use the `master` version of [Stable Baselines3](https://github.com/DLR-RM/stable-baselines3/).

To install Stable Baselines3 `master` version:
```
pip install git+https://github.com/DLR-RM/stable-baselines3
```

Install Stable Baselines3 - Contrib using pip:
```
pip install git+https://github.com/Stable-Baselines-Team/stable-baselines3-contrib
```

## How To Contribute

If you want to contribute, please read [**CONTRIBUTING.md**](./CONTRIBUTING.md) guide first.


## Citing the Project

To cite this repository in publications (please cite SB3 directly):

```
@misc{stable-baselines3,
  author = {Raffin, Antonin and Hill, Ashley and Ernestus, Maximilian and Gleave, Adam and Kanervisto, Anssi and Dormann, Noah},
  title = {Stable Baselines3},
  year = {2019},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/DLR-RM/stable-baselines3}},
}
```
