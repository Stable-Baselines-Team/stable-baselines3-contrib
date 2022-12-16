## Contributing to Stable-Baselines3 - Contrib

This contrib repository is designed for experimental implementations of various
parts of reinforcement training so that others may make use of them. This includes full
RL algorithms, different tools (e.g. new environment wrappers,
callbacks) and extending algorithms implemented in stable-baselines3.

**Before opening a pull request**, open an issue discussing the contribution.
Once we agree that the plan looks good, go ahead and implement it.

Contributions and review focuses on following three parts:
1) **Implementation quality**
  - Performance of the RL algorithms should match the one reported by the original authors (if applicable).
  - This is ensured by including a code that replicates an experiment from the original
    paper or from an established codebase (e.g. the code from authors), as well as
    a test to check that implementation works on program level (does not crash).
2) Documentation
  - Documentation quality should match that of stable-baselines3, with each feature covered
    in the documentation, in-code documentation to clarify the flow
    of logic and report of the expected results, where applicable.
3) Consistency with stable-baselines3
  - To ease readability, all contributions need to follow the code style (see below) and
    idioms used in stable-baselines3.

The implementation quality is a strict requirements with little room for changes, because
otherwise the implementation can do more harm than good (wrong results). Parts two and three
are taken into account during review but being a repository for more experimental code, these
are not very strict.

See [issues with "experimental" tag](https://github.com/DLR-RM/stable-baselines3/issues?q=is%3Aissue+is%3Aopen+label%3Aexperimental)
for suggestions of the community for new possible features to include in contrib.

## How to implement your suggestion

Implement your feature/suggestion/algorithm in following ways, using the first one that applies:
1) Environment wrapper: This can be used with any algorithm and even outside stable-baselines3.
   Place code for these under `sb3_contrib/common/wrappers` directory.
2) [Custom callback](https://stable-baselines3.readthedocs.io/en/master/guide/callbacks.html).
   Place code under `sb3_contrib/common/callbacks` directory.
3) Following the style/naming of `common` files in the stable-baseline3. If your suggestion is a specific network architecture
   for feature extraction from images, place this in `sb3_contrib/common/torch_layers.py`, for example.
4) A new learning algorithm. This is the last resort but most applicable solution.
   Even if your suggestion is a (trivial) modification to an existing algorithm, create a new algorithm for it
   unless otherwise discussed (which inherits the base algorithm). The algorithm should use same API as
   stable-baselines3 algorithms (e.g. `learn`, `load`), and the code should be placed under
   `sb3_contrib/[algorithm_name]` directory.

Look over stable-baselines3 code for the general naming of variables and try to keep this style.

If algorithm you are implementing involves more complex/uncommon equations, comment each part of these
calculations with references to the parts in paper.

## Pull Request (PR) and review

Before proposing a PR, please open an issue, where the feature will be discussed.
This prevent from duplicated PR to be proposed and also ease the code review process.

Each PR need to be reviewed and accepted by at least one of the maintainers.
A PR must pass the Continuous Integration tests to be merged with the master branch.

Along with the code, PR **must** include the following:
1) Update to documentation to include a description of the feature. If feature is a simple tool (e.g. wrapper, callback),
   this goes under respective pages in documentation. If full training algorithm, this goes under a new page with template below
   (`docs/modules/[algo_name]`).
2) If a training algorithm/improvement: results of a replicated experiment from the original paper in the documentation,
   **which must match the results from authors** unless solid arguments can be provided why they did not match.
3) If above holds: The **exact** code to run the replicated experiment (i.e. it should produce the above results), and inside the
   code information about the environment used (Python version, library versions, OS, hardware information). If small enough,
   include this in the documentation. If applicable, use [rl-baselines3-zoo](https://github.com/DLR-RM/rl-baselines3-zoo) to
   run the agent performance comparison experiments (fork repository, implement experiment in a new branch and share link to
   that branch). If above do not apply, create new code to replicate the experiment and include link to it.
4) Updated tests in `tests/test_run.py` and `tests/test_save_load.py` to test that features run as expected and serialize
   correctly. This this is **not** for testing e.g. training performance of a learning algorithm, and
   should be relatively quick to run.

Below is a template for documentation for full RL algorithms.

```rst
[Feature/Algorithm name]
========================

- Non-abbreviated  name and/or one-sentence description of the method.
- Link and reference to the original publications the present the feature, or other established source(s).
- Links to any codebases that were used for reference (e.g. authors' implementations)

Example
-------

A minimal example on how to use the feature (full, runnable code).

Results
-------

A description and comparison of results (e.g. how the change improved results over the non-changed algorithm), if
applicable.
Please link the associated pull request, e.g., `Pull Request #4 <https://github.com/Stable-Baselines-Team/stable-baselines3-contrib/pull/4>`_.

Include the expected results from the work that originally proposed the method (e.g. original paper).

How to replicate the results?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Include the code to replicate these results or a link to repository/branch where the code can be found.
Use `rl-baselines3-zoo <https://github.com/DLR-RM/rl-baselines3-zoo>`_ if possible, fork it, create a new branch
and share the code to replicate results there.

If applicable, please also provide the command to replicate the plots.

Comments
--------

Comments regarding the implementation, e.g. missing parts, uncertain parts, differences
to the original implementation.
````

If you are not familiar with creating a Pull Request, here are some guides:
- http://stackoverflow.com/questions/14680711/how-to-do-a-github-pull-request
- https://help.github.com/articles/creating-a-pull-request/


## Codestyle

We are using [black codestyle](https://github.com/psf/black) (max line length of 127 characters) together with [isort](https://github.com/timothycrosley/isort) to sort the imports.

**Please run `make format`** to reformat your code. You can check the codestyle using `make check-codestyle` and `make lint`.

Please document each function/method and [type](https://google.github.io/pytype/user_guide.html) them using the following template:

```python

def my_function(arg1: type1, arg2: type2) -> returntype:
    """
    Short description of the function.

    :param arg1: describe what is arg1
    :param arg2: describe what is arg2
    :return: describe what is returned
    """
    ...
    return my_variable
```

## Tests

All new features and algorithms must add tests in the `tests/` folder ensuring that everything works fine (on program level).
We use [pytest](https://pytest.org/).
Also, when a bug fix is proposed, tests should be added to avoid regression.

To run tests with `pytest`:

```
make pytest
```

Type checking with `pytype` and `mypy`:

```
make type
```

Codestyle check with `black`, `isort` and `flake8`:

```
make check-codestyle
make lint
```

To run `type`, `format` and `lint` in one command:
```
make commit-checks
```

Build the documentation:

```
make doc
```

## Changelog and Documentation

Please do not forget to update the changelog (`docs/misc/changelog.rst`).

Credits: this contributing guide is based on the [PyTorch](https://github.com/pytorch/pytorch/) one.
