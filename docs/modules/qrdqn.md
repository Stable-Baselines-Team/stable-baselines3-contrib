(qrdqn)=

```{eval-rst}
.. automodule:: sb3_contrib.qrdqn

```

# QR-DQN

[Quantile Regression DQN (QR-DQN)](https://arxiv.org/abs/1710.10044) builds on [Deep Q-Network (DQN)](https://arxiv.org/abs/1312.5602)
and make use of quantile regression to explicitly model the [distribution over returns](https://arxiv.org/abs/1707.06887),
instead of predicting the mean return (DQN).

```{eval-rst}
.. rubric:: Available Policies
```

```{eval-rst}
.. autosummary::
    :nosignatures:

    MlpPolicy
    CnnPolicy
    MultiInputPolicy

```

## Notes

- Original paper: <https://arxiv.org/abs/1710.10044>
- Distributional RL (C51): <https://arxiv.org/abs/1707.06887>
- Further reference: <https://github.com/amy12xx/ml_notes_and_reports/blob/master/distributional_rl/QRDQN.pdf>

## Can I use?

- Recurrent policies: ❌
- Multi processing: ✔️
- Gym spaces:

| Space         | Action | Observation |
| ------------- | ------ | ----------- |
| Discrete      | ✔️     | ✔️          |
| Box           | ❌     | ✔️          |
| MultiDiscrete | ❌     | ✔️          |
| MultiBinary   | ❌     | ✔️          |
| Dict          | ❌     | ✔️          |

## Example

```python
import gymnasium as gym

from sb3_contrib import QRDQN

env = gym.make("CartPole-v1", render_mode="human")

policy_kwargs = dict(n_quantiles=50)
model = QRDQN("MlpPolicy", env, policy_kwargs=policy_kwargs, verbose=1)
model.learn(total_timesteps=10_000, log_interval=4)
model.save("qrdqn_cartpole")

del model # remove to demonstrate saving and loading

model = QRDQN.load("qrdqn_cartpole")

obs, _ = env.reset()
while True:
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    env.render()
    if terminated or truncated:
      obs, _ = env.reset()
```

## Results

Result on Atari environments (10M steps, Pong and Breakout) and classic control tasks using 3 and 5 seeds.

The complete learning curves are available in the [associated PR](https://github.com/Stable-Baselines-Team/stable-baselines3-contrib/pull/13).

:::{note}
QR-DQN implementation was validated against [Intel Coach](https://github.com/IntelLabs/coach/tree/master/benchmarks/qr_dqn) one
which roughly compare to the original paper results (we trained the agent with a smaller budget).
:::

| Environments | QR-DQN     | DQN        |
| ------------ | ---------- | ---------- |
| Breakout     | 413 +/- 21 | ~300       |
| Pong         | 20 +/- 0   | ~20        |
| CartPole     | 386 +/- 64 | 500 +/- 0  |
| MountainCar  | -111 +/- 4 | -107 +/- 4 |
| LunarLander  | 168 +/- 39 | 195 +/- 28 |
| Acrobot      | -73 +/- 2  | -74 +/- 2  |

### How to replicate the results?

Clone RL-Zoo fork and checkout the branch `feat/qrdqn`:

```bash
git clone https://github.com/ku2482/rl-baselines3-zoo/
cd rl-baselines3-zoo/
git checkout feat/qrdqn
```

Run the benchmark (replace `$ENV_ID` by the envs mentioned above):

```bash
python train.py --algo qrdqn --env $ENV_ID --eval-episodes 10 --eval-freq 10000
```

Plot the results:

```bash
python scripts/all_plots.py -a qrdqn -e Breakout Pong -f logs/ -o logs/qrdqn_results
python scripts/plot_from_file.py -i logs/qrdqn_results.pkl -latex -l QR-DQN
```

## Parameters

```{eval-rst}
.. autoclass:: QRDQN
  :members:
  :inherited-members:
```

(qrdqn_policies)=

## QR-DQN Policies

```{eval-rst}
.. autoclass:: MlpPolicy
  :members:
  :inherited-members:
```

```{eval-rst}
.. autoclass:: sb3_contrib.qrdqn.policies.QRDQNPolicy
  :members:
  :noindex:
```

```{eval-rst}
.. autoclass:: CnnPolicy
  :members:
```

```{eval-rst}
.. autoclass:: MultiInputPolicy
  :members:
```
