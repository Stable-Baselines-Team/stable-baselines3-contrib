(sacd)=

```{eval-rst}
.. automodule:: sb3_contrib.sacd

```

# SACD

[Soft Actor Critic Discrete (SACD)](https://arxiv.org/abs/1910.07207) is a modification of the original Soft Actor Critic Algorithm for discrete action spaces.

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

- Original paper: <https://arxiv.org/abs/1910.07207>
- Original Implementation: <https://github.com/p-christ/Deep-Reinforcement-Learning-Algorithms-with-PyTorch>

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

from sb3_contrib import SACD

env = gym.make("CartPole-v1", render_mode="rgb_array")

model = SACD("MlpPolicy", env, verbose=1, policy_kwargs=dict(net_arch=[64,64]))
model.learn(total_timesteps=20_000)
model.save("sacd_cartpole")

del model # remove to demonstrate saving and loading

model = SACD.load("sac_cartpole")

obs, info = env.reset()
while True:
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        obs, info = env.reset()
```

## Parameters

```{eval-rst}
.. autoclass:: SACD
  :members:
  :inherited-members:
```

(sac-policies)=

## SACD Policies

```{eval-rst}
.. autoclass:: MlpPolicy
  :members:
  :inherited-members:
```

```{eval-rst}
.. autoclass:: stable_baselines3.sac.policies.SACPolicy
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
