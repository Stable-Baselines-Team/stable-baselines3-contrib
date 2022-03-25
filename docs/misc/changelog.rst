.. _changelog:

Changelog
==========

Release 1.5.0 (2022-03-25)
-------------------------------

Breaking Changes:
^^^^^^^^^^^^^^^^^
- Switched minimum Gym version to 0.21.0.
- Upgraded to Stable-Baselines3 >= 1.5.0

New Features:
^^^^^^^^^^^^^
- Allow PPO to turn of advantage normalization (see `PR #61 <https://github.com/Stable-Baselines-Team/stable-baselines3-contrib/pull/61>`_) @vwxyzjn

Bug Fixes:
^^^^^^^^^^
- Removed explict calls to ``forward()`` method as per pytorch guidelines

Deprecations:
^^^^^^^^^^^^^

Others:
^^^^^^^

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

@ku2482 @guyk1971 @minhlong94 @ayeright @kronion @glmcdona @cyprienc @sgillen
