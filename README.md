[![CI](https://github.com/Stable-Baselines-Team/stable-baselines3-contrib/workflows/CI/badge.svg)](https://github.com/Stable-Baselines-Team/stable-baselines3-contrib/actions) [![codestyle](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

# Stable-Baselines3 - Contrib

Contrib package for [Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3) - Experimental code.
"sb3-contrib" for short.

A place for training algorithms and tools that are considered experimental, e.g. implementations of the latest
publications. Goal is to keep the simplicity, documentation and style of stable-baselines3 but for less matured
implementations. 

Why create this repository? Over the span of stable-baselines and stable-baselines3, the community has been eager
to contribute in form of better logging utilities, environment wrappers, extended support (e.g. different action spaces)
and learning algorithms. However sometimes these utilities were too niche to be considered for stable-baselines or
proved to be too difficult to integrate well into existing code without a mess. sb3-contrib aims to fix this by
not requiring the neatest code integration with existing code and not setting limits on what is too niche: almost everything
remotely useful goes! We hope this allows to extend the known quality of stable-baselines style and documentation beyond
the relatively small scope of utilities of the main repository.


## Features

See documentation for the full list of included features.

**Training algorithms**:
- [Truncated Quantile Critics (TQC)](https://arxiv.org/abs/2005.04269)


## Installation

**Note:** You need the `master` version of [Stable Baselines3](https://github.com/DLR-RM/stable-baselines3/).

To install Stable Baselines3 `master` version:
```
pip install git+https://github.com/DLR-RM/stable-baselines3
```

Install Stable Baselines3 - Contrib using pip:
```
pip install git+https://github.com/Stable-Baselines-Team/stable-baselines3-contrib
```


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
