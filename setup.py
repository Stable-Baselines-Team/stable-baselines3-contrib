import os

from setuptools import find_packages, setup

with open(os.path.join("sb3_contrib", "version.txt"), "r") as file_handler:
    __version__ = file_handler.read().strip()


long_description = """

# Stable-Baselines3 - Contrib

Contrib package for [Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3) - Experimental code

Implemented:
- [Truncated Quantile Critics (TQC)](https://arxiv.org/abs/2005.04269)

"""  # noqa:E501


setup(
    name="sb3_contrib",
    packages=[package for package in find_packages() if package.startswith("sb3_contrib")],
    package_data={"sb3_contrib": ["py.typed", "version.txt"]},
    install_requires=[
        "stable_baselines3[tests,docs]",
        # For progress bar when using CRR
        "tqdm"
        # Enable CMA
        # "cma",
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
