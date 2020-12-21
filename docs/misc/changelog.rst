.. _changelog:

Changelog
==========


Pre-Release 0.11.0a3 (WIP)
-------------------------------

Breaking Changes:
^^^^^^^^^^^^^^^^^

New Features:
^^^^^^^^^^^^^
- Added ``TimeFeatureWrapper`` to the wrappers
- Added ``QR-DQN`` algorithm (`@ku2482`_)

Bug Fixes:
^^^^^^^^^^
- Fixed bug in ``TQC`` when saving/loading the policy only with non-default number of quantiles

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

@ku2482
