## Contributing to Stable-Baselines3 - Contrib

This contrib repository is designed for experimental implementations of various
training algorithms so that others may make use of them. This includes full
training algorithms, different tools (e.g. new environment wrappers,
callbacks) and extending algorithms implemented in stable-baselines3.

**Before opening a pull request**, open an issue discussing the contribution.
Once we agree that the plan looks good, go ahead and implement it.

Contributions and review focuses on following three parts:
1) **Implementation quality**
  - Performance of the training algorithms should match what proposed authors reported.
  - This is ensured by including a code that replicates an experiment from the original
    paper or from an established codebase (e.g. the code from authors), as well as 
    a test to check that implementation works on program level (does not crash).
2) Documentation
  - Documentation quality should match that of stable-baselines3, with each algorithm
    containing its own README file, changelog, in-code documentation to clarify the flow
    of logic and report the expected results.
3) Consistency with stable-baselines3
  - To ease readibility, all contributions need to follow the code style (see below) and
    ideoms used in stable-baselines3. 

The implementation quality is a strict requirements with little room for changes, because
otherwise the implementation can do more harm than good (wrong results). Parts two and three
are taken into account during review but being a repository for more experimental code, these
are not very strict.

## How to implement your suggestion

All code will go under `sb3_contrib/[feature_name]` directory, regardless of what they implement.
The idea is to keep different contributions separate from each other and only over time combine/mature
them into a shared package.

Implement your feature/suggestion/algorithm in following ways, using the first one that applies:
1) Environment wrapper: This can be used with any algorithm and even outside stable-baselines3.
2) [Custom callback](https://stable-baselines3.readthedocs.io/en/master/guide/callbacks.html)
3) Following the style/naming of `common` files in the stable-baseline3. If your suggestion is a specific network architecture
   for feature extraction from images, place this under `sb3_contrib/[feature_name]/torch_layers.py`, for example.
4) A new learning algorithm. This is the last resort but most applicable solution.
   Even if your suggestion is a (trivial) modification to an existing algorithm, create a new algorithm for it
   unless otherwise discussed (which inherits the base algorithm). The algorithm should use same API as
   stable-baselines3 algorithms (e.g. `learn`, `load`)

Look over stable-baselines3 code for the general naming of variables and try to keep this style.

If algorithm you are implementing involves more complex/uncommon equations, comment each part of these
calculations with references to the parts in paper.

## Pull Request (PR) and review

Before proposing a PR, please open an issue, where the feature will be discussed.
This prevent from duplicated PR to be proposed and also ease the code review process.

Each PR need to be reviewed and accepted by at least one of the maintainers.
A PR must pass the Continuous Integration tests to be merged with the master branch.

Along with the code, PR **must** include the following:
1) `README.md` file for the feature (see template below). This is placed in the algorithm's directory.
2) Results of a replicated experiment from the original paper, **which must match the results from authors**
   unless solid arguments can be provided why they did not match. 
3) The **exact** code to run the replicated experiment (i.e. it should produce the above results), and inside the
   code information about the environment used (Python version, library versions, OS, hardware information).
4) A new test file under `tests/`, which tests the implementation for functional errors. This this is **not** for
   testing e.g. training performance of a learning algorithm, and should be relatively quick to run.

README.md template:

```markdown
# [Feature/Algorithm name]

* Non-abreviated name and/or one-sentence description of the method.
* Link and reference to the original publications the present the feature, or other established source(s).
* Links to any codebases that were used for reference (e.g. authors' implementations)

## Example

A minimal example on how to use the feature (full, runnable code).

## Results

A copy of results reported in the original paper and results obtained by your replicate of the experiments, as well
as an overview of the experiment setup (full details are in the code you will provide).

## Comments

Comments regarding the implementation, e.g. missing parts, uncertain parts, differences
to the original implementation.

## Changelog

Per-algorithm changelog, in format "dd/mm/yyyy username: comment". E.g:
* 05.10.2020 Miffyli: Adding missing instructions for contrib repo
* 04.10.2020 Miffyli: Initial commit
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

    :param arg1: (type1) describe what is arg1
    :param arg2: (type2) describe what is arg2
    :return: (returntype) describe what is returned
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

Type checking with `pytype`:

```
make type
```

Codestyle check with `black`, `isort` and `flake8`:

```
make check-codestyle
make lint
```

To run `pytype`, `format` and `lint` in one command:
```
make commit-checks
```

## Changelog and Documentation

Please do not forget to update the changelog (`CHANGELOG.rst`).

Credits: this contributing guide is based on the [PyTorch](https://github.com/pytorch/pytorch/) one.
