.. _changelog:

Changelog
==========

Release 2.3.0 (2024-03-31)
--------------------------

**New defaults hyperparameters for QR-DQN**

Breaking Changes:
^^^^^^^^^^^^^^^^^
- Upgraded to Stable-Baselines3 >= 2.3.0
- The default ``learning_starts`` parameter of ``QRDQN`` have been changed to be consistent with the other offpolicy algorithms


.. code-block:: python

  # SB3 < 2.3.0 default hyperparameters, 50_000 corresponded to Atari defaults hyperparameters
  # model = QRDQN("MlpPolicy", env, learning_starts=50_000)
  # SB3 >= 2.3.0:
  model = QRDQN("MlpPolicy", env, learning_starts=100)


New Features:
^^^^^^^^^^^^^
- Added ``rollout_buffer_class`` and ``rollout_buffer_kwargs`` arguments to MaskablePPO
- Log success rate ``rollout/success_rate`` when available for on policy algorithms


Bug Fixes:
^^^^^^^^^^

Deprecations:
^^^^^^^^^^^^^

Others:
^^^^^^^
- Fixed ``train_freq`` type annotation for tqc and qrdqn (@Armandpl)
- Fixed ``sb3_contrib/common/maskable/*.py`` type annotations
- Fixed ``sb3_contrib/ppo_mask/ppo_mask.py`` type annotations
- Fixed ``sb3_contrib/common/vec_env/async_eval.py`` type annotations

Documentation:
^^^^^^^^^^^^^^
- Add some additional notes about ``MaskablePPO`` (evaluation and multi-process) (@icheered)


Release 2.2.1 (2023-11-17)
--------------------------

Breaking Changes:
^^^^^^^^^^^^^^^^^
- Upgraded to Stable-Baselines3 >= 2.2.1
- Switched to ``ruff`` for sorting imports (isort is no longer needed), black and ruff version now require a minimum version
- Dropped ``x is False`` in favor of ``not x``, which means that callbacks that wrongly returned None (instead of a boolean) will cause the training to stop (@iwishiwasaneagle)

New Features:
^^^^^^^^^^^^^
- Added ``set_options`` for ``AsyncEval``
- Added ``rollout_buffer_class`` and ``rollout_buffer_kwargs`` arguments to TRPO

Bug Fixes:
^^^^^^^^^^

Deprecations:
^^^^^^^^^^^^^

Others:
^^^^^^^
- Fixed ``ActorCriticPolicy.extract_features()`` signature by adding an optional ``features_extractor`` argument
- Update dependencies (accept newer Shimmy/Sphinx version and remove ``sphinx_autodoc_typehints``)

Documentation:
^^^^^^^^^^^^^^


Release 2.1.0 (2023-08-17)
--------------------------

Breaking Changes:
^^^^^^^^^^^^^^^^^
- Removed Python 3.7 support
- SB3 now requires PyTorch > 1.13
- Upgraded to Stable-Baselines3 >= 2.1.0

New Features:
^^^^^^^^^^^^^
- Added Python 3.11 support

Bug Fixes:
^^^^^^^^^^
- Fixed MaskablePPO ignoring ``stats_window_size`` argument

Deprecations:
^^^^^^^^^^^^^

Others:
^^^^^^^

Documentation:
^^^^^^^^^^^^^^


Release 2.0.0 (2023-06-22)
--------------------------

**Gymnasium support**

.. warning::

  Stable-Baselines3 (SB3) v2.0 will be the last one supporting python 3.7 (end of life in June 2023).
  We highly recommended you to upgrade to Python >= 3.8.


Breaking Changes:
^^^^^^^^^^^^^^^^^
- Switched to Gymnasium as primary backend, Gym 0.21 and 0.26 are still supported via the ``shimmy`` package (@carlosluis, @arjun-kg, @tlpss)
- Upgraded to Stable-Baselines3 >= 2.0.0

New Features:
^^^^^^^^^^^^^

Bug Fixes:
^^^^^^^^^^
- Fixed QRDQN update interval for multi envs

Deprecations:
^^^^^^^^^^^^^

Others:
^^^^^^^
- Fixed ``sb3_contrib/tqc/*.py`` type hints
- Fixed ``sb3_contrib/trpo/*.py`` type hints
- Fixed ``sb3_contrib/common/envs/invalid_actions_env.py`` type hints


Documentation:
^^^^^^^^^^^^^^
- Update documentation, switch from Gym to Gymnasium

Release 1.8.0 (2023-04-07)
--------------------------

.. warning::

  Stable-Baselines3 (SB3) v1.8.0 will be the last one to use Gym as a backend.
  Starting with v2.0.0, Gymnasium will be the default backend (though SB3 will have compatibility layers for Gym envs).
  You can find a migration guide here: https://gymnasium.farama.org/content/migration-guide/.
  If you want to try the SB3 v2.0 alpha version, you can take a look at `PR #1327 <https://github.com/DLR-RM/stable-baselines3/pull/1327>`_.


Breaking Changes:
^^^^^^^^^^^^^^^^^
- Removed shared layers in ``mlp_extractor`` (@AlexPasqua)
- Upgraded to Stable-Baselines3 >= 1.8.0

New Features:
^^^^^^^^^^^^^
- Added ``stats_window_size`` argument to control smoothing in rollout logging (@jonasreiher)

Bug Fixes:
^^^^^^^^^^

Deprecations:
^^^^^^^^^^^^^

Others:
^^^^^^^
- Moved to pyproject.toml
- Added github issue forms
- Fixed Atari Roms download in CI
- Fixed ``sb3_contrib/qrdqn/*.py`` type hints
- Switched from ``flake8`` to ``ruff``

Documentation:
^^^^^^^^^^^^^^
- Added warning about potential crashes caused by ``check_env`` in the ``MaskablePPO`` docs (@AlexPasqua)


Release 1.7.0 (2023-01-10)
--------------------------

.. warning::

  Shared layers in MLP policy (``mlp_extractor``) are now deprecated for PPO, A2C and TRPO.
  This feature will be removed in SB3 v1.8.0 and the behavior of ``net_arch=[64, 64]``
  will create **separate** networks with the same architecture, to be consistent with the off-policy algorithms.


Breaking Changes:
^^^^^^^^^^^^^^^^^
- Removed deprecated ``create_eval_env``, ``eval_env``, ``eval_log_path``, ``n_eval_episodes`` and ``eval_freq`` parameters,
  please use an ``EvalCallback`` instead
- Removed deprecated ``sde_net_arch`` parameter
- Upgraded to Stable-Baselines3 >= 1.7.0

New Features:
^^^^^^^^^^^^^
- Introduced mypy type checking
- Added support for Python 3.10
- Added ``with_bias`` parameter to ``ARSPolicy``
- Added option to have non-shared features extractor between actor and critic in on-policy algorithms (@AlexPasqua)
- Features extractors now properly support unnormalized image-like observations (3D tensor)
  when passing ``normalize_images=False``

Bug Fixes:
^^^^^^^^^^
- Fixed a bug in ``RecurrentPPO`` where the lstm states where incorrectly reshaped for ``n_lstm_layers > 1`` (thanks @kolbytn)
- Fixed ``RuntimeError: rnn: hx is not contiguous`` while predicting terminal values for ``RecurrentPPO`` when ``n_lstm_layers > 1``

Deprecations:
^^^^^^^^^^^^^
- You should now explicitely pass a ``features_extractor`` parameter when calling ``extract_features()``
- Deprecated shared layers in ``MlpExtractor`` (@AlexPasqua)

Others:
^^^^^^^
- Fixed flake8 config
- Fixed ``sb3_contrib/common/utils.py`` type hint
- Fixed ``sb3_contrib/common/recurrent/type_aliases.py`` type hint
- Fixed ``sb3_contrib/ars/policies.py`` type hint
- Exposed modules in `__init__.py` with `__all__` attribute (@ZikangXiong)
- Removed ignores on Flake8 F401 (@ZikangXiong)
- Upgraded GitHub CI/setup-python to v4 and checkout to v3
- Set tensors construction directly on the device
- Standardized the use of ``from gym import spaces``

Release 1.6.2 (2022-10-10)
--------------------------

**Progress bar and upgrade to latest SB3 version**

Breaking Changes:
^^^^^^^^^^^^^^^^^
- Upgraded to Stable-Baselines3 >= 1.6.2

New Features:
^^^^^^^^^^^^^
- Added ``progress_bar`` argument in the ``learn()`` method, displayed using TQDM and rich packages

Bug Fixes:
^^^^^^^^^^

Deprecations:
^^^^^^^^^^^^^
- Deprecate parameters ``eval_env``, ``eval_freq`` and ``create_eval_env``

Others:
^^^^^^^
- Fixed the return type of ``.load()`` methods so that they now use ``TypeVar``


Release 1.6.1 (2022-09-29)
-------------------------------

**Bug fix release**

Breaking Changes:
^^^^^^^^^^^^^^^^^
- Fixed the issue that ``predict`` does not always return action as ``np.ndarray`` (@qgallouedec)
- Upgraded to Stable-Baselines3 >= 1.6.1

New Features:
^^^^^^^^^^^^^

Bug Fixes:
^^^^^^^^^^
- Fixed the issue of wrongly passing policy arguments when using CnnLstmPolicy or MultiInputLstmPolicy with ``RecurrentPPO`` (@mlodel)
- Fixed division by zero error when computing FPS when a small number of time has elapsed in operating systems with low-precision timers.
- Fixed calling child callbacks in MaskableEvalCallback (@CppMaster)
- Fixed missing verbose parameter passing in the ``MaskableEvalCallback`` constructor (@burakdmb)
- Fixed the issue that when updating the target network in QRDQN, TQC, the ``running_mean`` and ``running_var`` properties of batch norm layers are not updated (@honglu2875)

Deprecations:
^^^^^^^^^^^^^

Others:
^^^^^^^
- Changed the default buffer device from ``"cpu"`` to ``"auto"``


Release 1.6.0 (2022-07-11)
--------------------------

**Add RecurrentPPO (aka PPO LSTM)**

Breaking Changes:
^^^^^^^^^^^^^^^^^
- Upgraded to Stable-Baselines3 >= 1.6.0
- Changed the way policy "aliases" are handled ("MlpPolicy", "CnnPolicy", ...), removing the former
  ``register_policy`` helper, ``policy_base`` parameter and using ``policy_aliases`` static attributes instead (@Gregwar)
- Renamed ``rollout/exploration rate`` key to ``rollout/exploration_rate`` for QRDQN (to be consistent with SB3 DQN)
- Upgraded to python 3.7+ syntax using ``pyupgrade``
- SB3 now requires PyTorch >= 1.11
- Changed the default network architecture when using ``CnnPolicy`` or ``MultiInputPolicy`` with TQC,
  ``share_features_extractor`` is now set to False by default and the ``net_arch=[256, 256]`` (instead of ``net_arch=[]`` that was before)


New Features:
^^^^^^^^^^^^^
- Added ``RecurrentPPO`` (aka PPO LSTM)

Bug Fixes:
^^^^^^^^^^
- Fixed a bug in ``RecurrentPPO`` when calculating the masked loss functions (@rnederstigt)
- Fixed a bug in ``TRPO`` where kl divergence was not implemented for ``MultiDiscrete`` space

Deprecations:
^^^^^^^^^^^^^

Release 1.5.0 (2022-03-25)
-------------------------------

Breaking Changes:
^^^^^^^^^^^^^^^^^
- Switched minimum Gym version to 0.21.0.
- Upgraded to Stable-Baselines3 >= 1.5.0

New Features:
^^^^^^^^^^^^^
- Allow PPO to turn of advantage normalization (see `PR #61 <https://github.com/Stable-Baselines-Team/stable-baselines3-contrib/pull/61>`_) (@vwxyzjn)


Bug Fixes:
^^^^^^^^^^
- Removed explict calls to ``forward()`` method as per pytorch guidelines

Deprecations:
^^^^^^^^^^^^^

Others:
^^^^^^^

Documentation:
^^^^^^^^^^^^^^

Release 1.4.0 (2022-01-19)
-------------------------------
**Add Trust Region Policy Optimization (TRPO) and Augmented Random Search (ARS) algorithms**

Breaking Changes:
^^^^^^^^^^^^^^^^^
- Dropped python 3.6 support
- Upgraded to Stable-Baselines3 >= 1.4.0
- ``MaskablePPO`` was updated to match latest SB3 ``PPO`` version (timeout handling and new method for the policy object)

New Features:
^^^^^^^^^^^^^
- Added ``TRPO`` (@cyprienc)
- Added experimental support to train off-policy algorithms with multiple envs (note: ``HerReplayBuffer`` currently not supported)
- Added Augmented Random Search (ARS) (@sgillen)

Bug Fixes:
^^^^^^^^^^

Deprecations:
^^^^^^^^^^^^^

Others:
^^^^^^^
- Improve test coverage for ``MaskablePPO``


Documentation:
^^^^^^^^^^^^^^

Release 1.3.0 (2021-10-23)
-------------------------------

**Add Invalid action masking for PPO**

.. warning::

  This version will be the last one supporting Python 3.6 (end of life in Dec 2021).
  We highly recommended you to upgrade to Python >= 3.7.


Breaking Changes:
^^^^^^^^^^^^^^^^^
- Removed ``sde_net_arch``
- Upgraded to Stable-Baselines3 >= 1.3.0

New Features:
^^^^^^^^^^^^^
- Added ``MaskablePPO`` algorithm (@kronion)
- ``MaskablePPO`` Dictionary Observation support (@glmcdona)


Bug Fixes:
^^^^^^^^^^

Deprecations:
^^^^^^^^^^^^^

Others:
^^^^^^^


Documentation:
^^^^^^^^^^^^^^


Release 1.2.0 (2021-09-08)
-------------------------------

**Train/Eval mode support**

Breaking Changes:
^^^^^^^^^^^^^^^^^
- Upgraded to Stable-Baselines3 >= 1.2.0

Bug Fixes:
^^^^^^^^^^
- QR-DQN and TQC updated so that their policies are switched between train and eval mode at the correct time (@ayeright)

Deprecations:
^^^^^^^^^^^^^

Others:
^^^^^^^
- Fixed type annotation
- Added python 3.9 to CI

Documentation:
^^^^^^^^^^^^^^


Release 1.1.0 (2021-07-01)
-------------------------------

**Dictionary observation support and timeout handling**

Breaking Changes:
^^^^^^^^^^^^^^^^^
- Added support for Dictionary observation spaces (cf. SB3 doc)
- Upgraded to Stable-Baselines3 >= 1.1.0
- Added proper handling of timeouts for off-policy algorithms (cf. SB3 doc)
- Updated usage of logger (cf. SB3 doc)

Bug Fixes:
^^^^^^^^^^
- Removed unused code in ``TQC``

Deprecations:
^^^^^^^^^^^^^

Others:
^^^^^^^
- SB3 docs and tests dependencies are no longer required for installing SB3 contrib

Documentation:
^^^^^^^^^^^^^^

- updated QR-DQN docs checkmark typo (@minhlong94)


Release 1.0 (2021-03-17)
-------------------------------

Breaking Changes:
^^^^^^^^^^^^^^^^^
- Upgraded to Stable-Baselines3 >= 1.0

Bug Fixes:
^^^^^^^^^^
- Fixed a bug with ``QR-DQN`` predict method when using ``deterministic=False`` with image space


Pre-Release 0.11.1 (2021-02-27)
-------------------------------

Bug Fixes:
^^^^^^^^^^
- Upgraded to Stable-Baselines3 >= 0.11.1


Pre-Release 0.11.0 (2021-02-27)
-------------------------------

Breaking Changes:
^^^^^^^^^^^^^^^^^
- Upgraded to Stable-Baselines3 >= 0.11.0

New Features:
^^^^^^^^^^^^^
- Added ``TimeFeatureWrapper`` to the wrappers
- Added ``QR-DQN`` algorithm (`@ku2482`_)

Bug Fixes:
^^^^^^^^^^
- Fixed bug in ``TQC`` when saving/loading the policy only with non-default number of quantiles
- Fixed bug in ``QR-DQN`` when calculating the target quantiles (@ku2482, @guyk1971)

Deprecations:
^^^^^^^^^^^^^

Others:
^^^^^^^
- Updated ``TQC`` to match new SB3 version
- Updated SB3 min version
- Moved ``quantile_huber_loss`` to ``common/utils.py`` (@ku2482)

Documentation:
^^^^^^^^^^^^^^



Pre-Release 0.10.0 (2020-10-28)
-------------------------------

**Truncated Quantiles Critic (TQC)**

Breaking Changes:
^^^^^^^^^^^^^^^^^

New Features:
^^^^^^^^^^^^^
- Added ``TQC`` algorithm (@araffin)

Bug Fixes:
^^^^^^^^^^
- Fixed features extractor issue (``TQC`` with ``CnnPolicy``)

Deprecations:
^^^^^^^^^^^^^

Others:
^^^^^^^

Documentation:
^^^^^^^^^^^^^^
- Added initial documentation
- Added contribution guide and related PR templates


Maintainers
-----------

Stable-Baselines3 is currently maintained by `Antonin Raffin`_ (aka `@araffin`_), `Ashley Hill`_ (aka @hill-a),
`Maximilian Ernestus`_ (aka @ernestum), `Adam Gleave`_ (`@AdamGleave`_) and `Anssi Kanervisto`_ (aka `@Miffyli`_).

.. _Ashley Hill: https://github.com/hill-a
.. _Antonin Raffin: https://araffin.github.io/
.. _Maximilian Ernestus: https://github.com/ernestum
.. _Adam Gleave: https://gleave.me/
.. _@araffin: https://github.com/araffin
.. _@AdamGleave: https://github.com/adamgleave
.. _Anssi Kanervisto: https://github.com/Miffyli
.. _@Miffyli: https://github.com/Miffyli
.. _@ku2482: https://github.com/ku2482

Contributors:
-------------

@ku2482 @guyk1971 @minhlong94 @ayeright @kronion @glmcdona @cyprienc @sgillen @Gregwar @rnederstigt @qgallouedec
@mlodel @CppMaster @burakdmb @honglu2875 @ZikangXiong @AlexPasqua @jonasreiher @icheered @Armandpl
