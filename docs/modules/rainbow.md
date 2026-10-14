(rainbow)=

```{eval-rst}
.. automodule:: sb3_contrib.rainbow
```

# Rainbow DQN

Rainbow DQN combines six extensions to Deep Q-Networks in a single agent:
Double DQN, prioritized experience replay, dueling networks, multi-step
returns, categorical distributional reinforcement learning and NoisyNets.

The [original paper](https://arxiv.org/abs/1710.02298) studies how these
extensions interact and reports improved data efficiency and final performance
on the Atari 2600 benchmark.

This implementation uses a memory-efficient replay buffer for stacked image
observations. It stores individual frames separately while replay entries
contain pointers to the frames and rewards required to reconstruct each
transition.

```{eval-rst}
.. rubric:: Available Policies
```

## Notes

- Original paper: [Rainbow: Combining Improvements in Deep Reinforcement Learning](https://arxiv.org/abs/1710.02298)
- Rainbow combines:
  - Double DQN
  - prioritized experience replay
  - dueling networks
  - multi-step returns
  - categorical distributional reinforcement learning
  - factorized NoisyNets
- True terminations prevent bootstrapping. Time-limit truncations retain a
  bootstrap discount based on the number of transitions completed before
  truncation.
- The replay ratio is expressed as gradient updates per environment transition.
  Training frequency is adjusted when using parallel environments.
- Prioritized replay beta is annealed according to the number of environment
  transitions collected.

## Can I use?

- Recurrent policies: ❌
- Multi processing: ✔️
- Gym spaces:

| Space         | Action | Observation |
| ------------- | ------ | ----------- |
| Discrete      | ✔️     | ❌          |
| Box           | ❌     | ✔️          |
| MultiDiscrete | ❌     | ❌          |
| MultiBinary   | ❌     | ❌          |
| Dict          | ❌     | ❌          |

Rainbow currently supports discrete action spaces and image observations.
Vector, discrete and dictionary observations are not currently supported.

## Example

```python
import gymnasium as gym

from sb3_contrib.rainbow import Rainbow

env = gym.make(
    "ALE/Pong-v5",
    frameskip=1,
    repeat_action_probability=0.0,
)

env = gym.wrappers.AtariPreprocessing(
    env,
    terminal_on_life_loss=False,
)
env = gym.wrappers.FrameStackObservation(env, stack_size=4)

model = Rainbow(
    "CnnPolicy",
    env,
    verbose=1,
)

model.learn(total_timesteps=1_000_000)
model.save("rainbow_pong")

del model

model = Rainbow.load("rainbow_pong", env=env)
```

### Results

The implementation was evaluated on BattleZone, NameThisGame and Phoenix.
For each environment, one agent was trained for approximately 200 million
Atari frames. The experiments were run as GPU jobs on a high-performance
computing cluster.

The final policy was evaluated over 20 episodes. The table reports the mean,
standard deviation and median of the returns from these episodes. To
characterise performance near the end of training, it also reports the mean
return across the final ten evaluation checkpoints.

For comparison, the table includes the no-op evaluation scores reported by
[Hessel et al. (2018)](https://arxiv.org/abs/1710.02298) for the original
Rainbow implementation.

| Environment | Final policy, mean ± SD | Final median | Mean over final ten evaluations | Original Rainbow no-op score |
|:------------|------------------------:|-------------:|--------------------------------:|------------------------------:|
| BattleZone | 93,050 ± 30,104 | 87,500 | 86,650 | 62,010 |
| NameThisGame | 14,498.5 ± 1,741.6 | 14,410 | 14,644.1 | 13,136 |
| Phoenix | 139,112 ± 78,316 | 119,905 | 141,740.7 | 108,528.6 |

The evaluation returns are comparable to the no-op scores reported for the
original Rainbow implementation. For all three environments, both the final
mean return and the mean over the final ten evaluation checkpoints exceed the
corresponding score reported by Hessel et al. Phoenix exhibits substantially
greater variation between evaluation episodes than BattleZone or
NameThisGame.

The learning curves below show the mean evaluation return throughout training.
The shaded regions are approximate 95% confidence intervals calculated from
the 20 evaluation episodes at each checkpoint. They describe variation between
evaluation episodes for a single trained agent and do not represent variation
across independent training runs.

![Test Validation Learning Curves](https://raw.githubusercontent.com/jbuerman/stable-baselines3-contrib/refs/heads/rainbow-dqn-replicate/sb3_contrib/rainbow/analysis/rainbow_validation_learning_curves.png)

These experiments provide evidence that the implementation reproduces the
expected learning behaviour and reaches evaluation returns comparable to those
reported for the original Rainbow implementation. The experiments are a
focused implementation validation rather than a reproduction of the complete
Atari benchmark.

## How to reproduce the results

The container definition, HPC submission script and analysis notebook used for
the validation are provided in the Rainbow replication directory in the
contributor's SB3-Contrib fork.

The container definition builds the Apptainer image containing the required
software environment. The HPC script submits one Slurm GPU job for each
environment and invokes the supplied `main.py` experiment script through the
Apptainer container. The analysis notebook reads the resulting evaluation
files and generates the summary table and learning curves.

Clone the fork and switch to the replication branch:

```bash
git clone https://github.com/jbuerman/stable-baselines3-contrib.git
cd stable-baselines3-contrib
git switch rainbow-dqn-replicate
```

Build the Apptainer image from the supplied definition file:

```bash
apptainer build rainbow.sif rainbow.def
```

Before submitting the experiments, configure the paths and cluster-specific
Slurm options in `execute_rainbow_experiments.sh`, including the Apptainer
image, working directory, partition and account.

Submit the experiments:

```bash
bash execute_rainbow_experiments.sh
```

For each environment, the submission script runs a command of the following
form:

```bash
apptainer run --nv rainbow.sif \
    --game <environment> \
    --repeat 0
```

The container uses `main.py` as its entry point.

```bash
python -m sb3_contrib.rainbow.main \
    --game <environment> \
    --repeat 0
```

The script trains the agent and records the experiment and evaluation outputs.

After placing the experiment outputs in the directory expected by the
notebook, generate the summary and learning curves with:

```bash
jupyter notebook rainbow_experiment_analysis.ipynb
```

## Implementation details

Rainbow uses a categorical value distribution with 51 atoms over a fixed
support from -10 to 10. The online network selects the next action and the
target network supplies the categorical distribution for that action.

Transitions are sampled proportionally to their priorities using a sum tree.
Importance-sampling weights correct the loss for prioritized sampling. Sample
priorities are updated using the unweighted per-transition categorical loss.

The replay buffer maintains separate pending frame and reward sequences for
each parallel environment. Multi-step returns stop at true terminations.
Time-limit truncations retain the appropriate bootstrap discount.

Factorized Gaussian noise is applied to the value and advantage streams. Noise
is resampled during training and non-deterministic prediction. Deterministic
prediction disables the noise.

## Parameters

```{eval-rst}
.. autoclass:: Rainbow
  :members:
  :inherited-members:
```

(rainbow_policies)=

## Rainbow Policies

```{eval-rst}
.. autoclass:: sb3_contrib.rainbow.policy.RainbowPolicy
  :members:
  :noindex:
```
