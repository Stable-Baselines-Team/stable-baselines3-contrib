<img src="docs/\_static/img/logo.png" align="right" width="40%"/>

[![CI](https://github.com/Stable-Baselines-Team/stable-baselines3-contrib/workflows/CI/badge.svg)](https://github.com/Stable-Baselines-Team/stable-baselines3-contrib/actions) [![codestyle](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

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
- [Augmented Random Search (ARS)](https://arxiv.org/abs/1803.07055)
- [Quantile Regression DQN (QR-DQN)](https://arxiv.org/abs/1710.10044)
- [PPO with invalid action masking (MaskablePPO)](https://arxiv.org/abs/2006.14171)
- [PPO with recurrent policy (RecurrentPPO aka PPO LSTM)](https://ppo-details.cleanrl.dev//2021/11/05/ppo-implementation-details/)
- [Truncated Quantile Critics (TQC)](https://arxiv.org/abs/2005.04269)
- [Trust Region Policy Optimization (TRPO)](https://arxiv.org/abs/1502.05477)

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

To install Stable Baselines3 contrib `master` version:
```
pip install git+https://github.com/Stable-Baselines-Team/stable-baselines3-contrib
```

## How To Contribute

If you want to contribute, please read [**CONTRIBUTING.md**](./CONTRIBUTING.md) guide first.


## Citing the Project

To cite this repository in publications (please cite SB3 directly):

```bibtex
@article{stable-baselines3,
  author  = {Antonin Raffin and Ashley Hill and Adam Gleave and Anssi Kanervisto and Maximilian Ernestus and Noah Dormann},
  title   = {Stable-Baselines3: Reliable Reinforcement Learning Implementations},
  journal = {Journal of Machine Learning Research},
  year    = {2021},
  volume  = {22},
  number  = {268},
  pages   = {1-8},
  url     = {http://jmlr.org/papers/v22/20-1364.html}
}
```
