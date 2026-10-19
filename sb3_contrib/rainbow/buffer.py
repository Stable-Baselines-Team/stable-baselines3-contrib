from typing import TYPE_CHECKING, Any, NamedTuple

import numpy as np
import torch
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.type_aliases import ReplayBufferSamples
from stable_baselines3.common.vec_env import VecNormalize


class SumTree:
    """
    Binary sum tree used for prioritised experience replay.

    Each parent node stores the sum of its children, allowing transitions to
    be sampled efficiently according to their priorities.
    """

    def __init__(self, size: int) -> None:
        """
        Initialise the sum tree.

        :param size: Maximum number of priorities stored in the tree.
        """
        self.index = 0
        self.size = size
        self.full = False  # Used to track actual capacity
        leaf_count = 2 ** (size - 1).bit_length()  # Put all used node leaves on last tree level
        self.tree_start = leaf_count - 1
        self.sum_tree = np.zeros(2 * leaf_count - 1, dtype=np.float32)
        self.max = 1.0  # Initial max value to return (1 = 1^ω)

    def _update_nodes(self, indices: np.ndarray) -> None:
        """
        Update parent nodes with the sums of their children.

        :param indices: Indices of the parent nodes to update.
        """
        children_indices = indices * 2 + np.expand_dims([1, 2], axis=1)
        self.sum_tree[indices] = np.sum(self.sum_tree[children_indices], axis=0)

    def _propagate(self, indices: np.ndarray) -> None:
        """
        Propagate priority changes from child nodes towards the root.

        :param indices: Indices of the nodes whose changes should be propagated.
        """
        parents = (indices - 1) // 2
        unique_parents = np.unique(parents)
        self._update_nodes(unique_parents)
        if parents[0] != 0:
            self._propagate(parents)

    def _propagate_index(self, index: int) -> None:
        """
        Propagate a priority change from a single node towards the root.

        :param index: Index of the node whose change should be propagated.
        """
        parent = (index - 1) // 2
        left, right = 2 * parent + 1, 2 * parent + 2
        self.sum_tree[parent] = self.sum_tree[left] + self.sum_tree[right]
        if parent != 0:
            self._propagate_index(parent)

    def update(self, indices: np.ndarray, values: np.ndarray) -> None:
        """
        Update priorities at the given tree indices.

        :param indices: Tree indices of the priorities to update.
        :param values: New priority values.
        """
        self.sum_tree[indices] = values
        self._propagate(indices)
        current_max_value = np.max(values)
        self.max = max(current_max_value, self.max)

    def _update_index(self, index: int, value: float) -> None:
        """
        Update a single priority in the sum tree.

        :param index: Tree index of the priority to update.
        :param value: New priority value.
        """
        self.sum_tree[index] = value
        self._propagate_index(index)
        self.max = max(value, self.max)

    def append(self, value: float) -> None:
        """
        Append a priority to the sum tree.

        :param value: Priority value to append.
        """
        self._update_index(self.index + self.tree_start, value)
        self.index = (self.index + 1) % self.size
        self.full = self.full or self.index == 0

    def _retrieve(self, indices: np.ndarray, values: np.ndarray) -> np.ndarray:
        """
        Find leaf indices corresponding to cumulative priority values.

        :param indices: Current tree indices in the recursive search.
        :param values: Cumulative priority values to locate in the tree.
        :return: Leaf indices corresponding to the cumulative priority values.
        """
        children_indices = indices * 2 + np.expand_dims([1, 2], axis=1)  # Make matrix of children indices
        # If indices correspond to leaf nodes, return them
        if children_indices[0, 0] >= self.sum_tree.shape[0]:
            return indices
        # If children indices correspond to leaf nodes, bound rare outliers in case total slightly overshoots
        if children_indices[0, 0] >= self.tree_start:
            children_indices = np.minimum(children_indices, self.sum_tree.shape[0] - 1)
        left_children_values = self.sum_tree[children_indices[0]]
        # Classify which values are in left or right branches
        successor_choices = np.greater(values, left_children_values).astype(np.int32)
        # Use classification to index into the indices matrix
        successor_indices = children_indices[successor_choices, np.arange(indices.size)]
        # Subtract the left branch values when searching in the right branch
        successor_values = values - successor_choices * left_children_values
        return self._retrieve(successor_indices, successor_values)

    def find(self, values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Find priorities and indices for cumulative priority values.

        :param values: Cumulative priority values to locate in the sum tree.
        :return: Priorities, data indices and tree indices for the located leaves.
        """
        indices = self._retrieve(np.zeros(values.shape, dtype=np.int32), values)
        data_indices = indices - self.tree_start
        return self.sum_tree[indices], data_indices, indices

    def total(self) -> float:
        """
        Return the total priority stored in the sum tree.

        :return: Sum of all priorities stored in the tree.
        """
        return float(self.sum_tree[0])


if TYPE_CHECKING:

    class PERReplayBufferSamples(NamedTuple):
        observations: torch.Tensor
        actions: torch.Tensor
        next_observations: torch.Tensor
        dones: torch.Tensor
        rewards: torch.Tensor
        idxs: np.ndarray
        weights: torch.Tensor
        discounts: torch.Tensor

else:
    # Extend ReplayBufferSamples with PER-specific fields.
    replay_buffer_samples_fields = list(ReplayBufferSamples.__annotations__.items())
    # Add PER-specific fields only
    replay_buffer_samples_fields.append(("idxs", np.ndarray))
    replay_buffer_samples_fields.append(("weights", torch.Tensor))
    PERReplayBufferSamples = NamedTuple(
        "PERReplayBufferSamples",
        replay_buffer_samples_fields,
    )
    # Preserve the ReplayBufferSamples defaults and add defaults for the
    # Rainbow-specific fields.
    base_defaults = ReplayBufferSamples.__new__.__defaults__ or ()
    PERReplayBufferSamples.__new__.__defaults__ = (
        *base_defaults,
        None,  # idxs
        None,  # weights
    )


class PER(ReplayBuffer):

    def __init__(
        self,
        size: int,
        device: torch.device | str,
        n_step: int,
        n_envs: int,
        gamma: float,
        alpha: float = 0.5,
        beta: float = 0.4,
        framestack: int = 4,
        image_width: int = 84,
        image_height: int = 84,
        rgb: bool = False,
    ) -> None:
        """
        Initialise the prioritised experience replay buffer.

        :param size: Maximum number of transitions stored in the replay buffer.
        :param device: PyTorch device used for sampled tensors.
        :param n_step: Number of steps used for multi-step returns.
        :param n_envs: Number of parallel environments.
        :param gamma: Discount factor.
        :param alpha: Priority exponent controlling the strength of prioritisation.
        :param beta: Initial importance-sampling exponent.
        :param framestack: Number of frames in each stacked observation.
        :param image_width: Observation image width.
        :param image_height: Observation image height.
        :param rgb: Whether observations contain RGB frames.
        """
        self.buffer_size = size
        self.n_envs = n_envs
        self.pos = 0
        self.full = False

        self.st = SumTree(size)
        self.index = 0

        # this is the number of frames, not the number of transitions
        # the technical size to ensure there are errors with overwritten memory in theory is very high-
        # (2*framestack - overlap) * first_states + non_first_states
        # with N=3, framestack=4, size=1M, average ep length 20, we need a total frame storage of around 1.35M
        # this however is still pretty light given it uses discrete memory.
        # Careful when using RGB though, as we don't need as much memory
        if rgb:
            self.storage_size = int(size * 4)
        else:
            self.storage_size = int(size * 1.25)
        self.gamma = gamma
        self.capacity = 0

        self.point_mem_idx = 0

        self.state_mem_idx = 0
        self.reward_mem_idx = 0

        self.image_width = image_width
        self.image_height = image_height

        self.max_prio = 1.0

        self.framestack = framestack

        self.alpha = alpha
        self.beta = beta
        # per-add() annealing step; set by Rainbow._setup_learn once total_timesteps is known
        self.beta_increment = 0.0
        self.eps = 1e-6  # small constant to stop 0 probability
        self.device = torch.device(device)

        self.last_terminal = [True for i in range(n_envs)]
        self.tstep_counter = [0 for i in range(n_envs)]

        self.n_step = n_step
        self.state_buffer: list[list[int]] = [[] for _ in range(n_envs)]
        self.reward_buffer: list[list[int]] = [[] for _ in range(n_envs)]

        if rgb:
            self.state_mem = np.zeros((self.storage_size, 3, self.image_width, self.image_height), dtype=np.uint8)
        else:
            self.state_mem = np.zeros((self.storage_size, self.image_width, self.image_height), dtype=np.uint8)
        # One extra slot at index storage_size acts as a zero-reward sentinel for n-step padding.
        # Ring indices wrap modulo storage_size, so this extra slot is never overwritten.
        self.action_mem = np.zeros(self.storage_size + 1, dtype=np.int64)
        self.reward_mem = np.zeros(self.storage_size + 1, dtype=float)
        self.done_mem = np.zeros(self.storage_size + 1, dtype=bool)
        self.trun_mem = np.zeros(self.storage_size + 1, dtype=bool)

        self.pad_idx = self.storage_size
        self.trun_mem[self.pad_idx] = True

        # everything here is stored as ints as they are just pointers to the actual memory
        # reward contains N values. The first value contains the action. The set of N contains the pointers for both
        # the reward and dones
        self.trans_dtype = np.dtype(
            [("state", int, self.framestack), ("n_state", int, self.framestack), ("reward", int, self.n_step)]
        )

        self.blank_trans = (
            np.zeros(self.framestack, dtype=int),
            np.zeros(self.framestack, dtype=int),
            np.zeros(self.n_step, dtype=int),
        )

        self.pointer_mem = np.array([self.blank_trans] * size, dtype=self.trans_dtype)

        self.overlap = self.framestack - self.n_step

    def add(
        self,
        obs: np.ndarray,
        next_obs: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
        infos: list[dict[str, Any]],
    ) -> None:
        """
        Add a batch of transitions to the replay buffer.

        :param obs: Current observations for each environment.
        :param next_obs: Next observations for each environment.
        :param action: Actions taken in each environment.
        :param reward: Rewards received in each environment.
        :param done: Episode termination flags for each environment.
        :param infos: Additional information returned by each environment.
        """
        batch_size = len(action)

        for i in range(batch_size):
            state = obs[i]
            next_state = next_obs[i]
            act = action[i]
            rew = reward[i]
            trun = infos[i].get("TimeLimit.truncated", False)
            dn = bool(done[i]) and not trun

            self.append(state, act, rew, next_state, dn, trun, stream=i)

        self.beta = min(self.beta + self.beta_increment, 1.0)

    def append(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        n_state: np.ndarray,
        done: bool,
        trun: bool,
        stream: int,
    ) -> None:
        """
        Append a transition to an environment stream.

        :param state: Current stacked observation.
        :param action: Action taken in the environment.
        :param reward: Reward received for the transition.
        :param n_state: Next stacked observation.
        :param done: Whether the transition terminates the episode.
        :param trun: Whether the transition truncates the episode.
        :param stream: Index of the parallel environment.
        """
        self.append_memory(state, action, reward, n_state, done, trun, stream)

        self.append_pointer(stream)

        if done or trun:
            self.finalize_experiences(stream)
            self.state_buffer[stream] = []
            self.reward_buffer[stream] = []

        self.last_terminal[stream] = done or trun

    def append_pointer(self, stream: int) -> None:
        """
        Store complete n-step transitions for an environment stream.

        :param stream: Index of the parallel environment.
        """
        while (
            len(self.state_buffer[stream]) >= self.framestack + self.n_step and len(self.reward_buffer[stream]) >= self.n_step
        ):
            # First array in the experience
            state_array = self.state_buffer[stream][: self.framestack]

            # Second array in the experience (starts after N frames)
            n_state_array = self.state_buffer[stream][self.n_step : self.n_step + self.framestack]

            # Reward array (first N rewards)
            reward_array = self.reward_buffer[stream][: self.n_step]

            # Add the experience to the list
            self.pointer_mem[self.point_mem_idx] = (
                np.array(state_array, dtype=int),
                np.array(n_state_array, dtype=int),
                np.array(reward_array, dtype=int),
            )

            # update the sumtree with the priority
            self.st.append(self.max_prio**self.alpha)

            self.capacity = min(self.buffer_size, self.capacity + 1)
            self.point_mem_idx = (self.point_mem_idx + 1) % self.buffer_size

            # Remove the first state and reward from the buffers to slide the window
            self.state_buffer[stream].pop(0)
            self.reward_buffer[stream].pop(0)

    def finalize_experiences(self, stream: int) -> None:
        """
        Store remaining n-step transitions at the end of an episode.

        Incomplete n-step reward sequences are padded with the sentinel index so
        that transitions near an episode boundary can still be sampled.

        :param stream: Index of the parallel environment.
        """
        # Process remaining states and rewards at the end of an episode
        while len(self.state_buffer[stream]) >= self.framestack and len(self.reward_buffer[stream]) > 0:
            # First array in the experience
            first_array = self.state_buffer[stream][: self.framestack]

            # Second array in the experience (Final `framestack` elements)
            second_array = self.state_buffer[stream][-self.framestack :]

            reward_array = self.reward_buffer[stream][:]
            while len(reward_array) < self.n_step:
                reward_array.append(self.pad_idx)

            self.pointer_mem[self.point_mem_idx] = (
                np.array(first_array, dtype=int),
                np.array(second_array, dtype=int),
                np.array(reward_array, dtype=int),
            )

            self.st.append(self.max_prio**self.alpha)

            self.point_mem_idx = (self.point_mem_idx + 1) % self.buffer_size
            self.capacity = min(self.buffer_size, self.capacity + 1)

            # Remove the first state and reward from the buffers to slide the window
            self.state_buffer[stream].pop(0)
            if len(self.reward_buffer[stream]) > 0:
                self.reward_buffer[stream].pop(0)

    def append_memory(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        n_state: np.ndarray,
        done: bool,
        trun: bool,
        stream: int,
    ) -> None:
        """
        Store transition data and frame pointers for an environment stream.

        :param state: Current stacked observation.
        :param action: Action taken in the environment.
        :param reward: Reward received for the transition.
        :param n_state: Next stacked observation.
        :param done: Whether the transition terminates the episode.
        :param trun: Whether the transition truncates the episode.
        :param stream: Index of the parallel environment.
        """
        if self.last_terminal[stream]:
            # add full transition
            for i in range(self.framestack):
                self.state_mem[self.state_mem_idx] = state[i]
                self.state_buffer[stream].append(self.state_mem_idx)
                self.state_mem_idx = (self.state_mem_idx + 1) % self.storage_size

            # remember n_step is not applied in this memory
            self.state_mem[self.state_mem_idx] = n_state[self.framestack - 1]
            self.state_buffer[stream].append(self.state_mem_idx)
            self.state_mem_idx = (self.state_mem_idx + 1) % self.storage_size

            self.action_mem[self.reward_mem_idx] = action
            self.reward_mem[self.reward_mem_idx] = reward
            self.done_mem[self.reward_mem_idx] = done
            self.trun_mem[self.reward_mem_idx] = trun

            self.reward_buffer[stream].append(self.reward_mem_idx)
            self.reward_mem_idx = (self.reward_mem_idx + 1) % self.storage_size

            self.tstep_counter[stream] = 0

        else:
            # just add relevant info
            self.state_mem[self.state_mem_idx] = n_state[self.framestack - 1]
            self.state_buffer[stream].append(self.state_mem_idx)
            self.state_mem_idx = (self.state_mem_idx + 1) % self.storage_size

            self.action_mem[self.reward_mem_idx] = action
            self.reward_mem[self.reward_mem_idx] = reward
            self.done_mem[self.reward_mem_idx] = done
            self.trun_mem[self.reward_mem_idx] = trun

            self.reward_buffer[stream].append(self.reward_mem_idx)
            self.reward_mem_idx = (self.reward_mem_idx + 1) % self.storage_size

    def sample(  # type: ignore[override]
        self,
        batch_size: int,
        env: VecNormalize | None = None,
    ) -> PERReplayBufferSamples:
        """
        Sample a batch of transitions using prioritised experience replay.

        :param batch_size: Number of transitions to sample.
        :param env: Optional VecNormalize environment, retained for compatibility
        with the SB3 replay buffer interface.
        :return: Sampled transitions, priorities, importance-sampling weights and
        bootstrap discounts.
        """
        # get total sumtree priority
        p_total = self.st.total()

        # first use sumtree prios to get the indices
        segment_length = p_total / batch_size
        segment_starts = np.arange(batch_size) * segment_length
        samples = np.random.uniform(0.0, segment_length, [batch_size]) + segment_starts

        prios, idxs, tree_idxs = self.st.find(samples)

        probs = prios / p_total

        # fetch the pointers by using indices
        pointers = self.pointer_mem[idxs]

        # Extract the pointers into separate arrays
        state_pointers = np.array([p[0] for p in pointers])
        n_state_pointers = np.array([p[1] for p in pointers])
        reward_pointers = np.array([p[2] for p in pointers])
        if self.n_step > 1:
            action_pointers = np.array([p[2][0] for p in pointers])
        else:
            action_pointers = np.array([p[2] for p in pointers])

        # get state info
        states = torch.tensor(self.state_mem[state_pointers], dtype=torch.uint8)
        n_states = torch.tensor(self.state_mem[n_state_pointers], dtype=torch.uint8)

        # Rewards, terminations and actions are retrieved through their pointers.
        rewards_array = self.reward_mem[reward_pointers]
        dones_array = self.done_mem[reward_pointers]
        truns_array = self.trun_mem[reward_pointers]
        actions_array = self.action_mem[action_pointers]

        # Apply n-step accumulation to rewards and terminations.
        if self.n_step > 1:
            (
                rewards_array,
                dones_array,
                discounts_array,
            ) = self.compute_discounted_rewards_batch(
                rewards_array,
                dones_array,
                truns_array,
            )
        else:
            rewards_array = rewards_array.reshape(-1)
            dones_array = dones_array.reshape(-1)
            actions_array = actions_array.reshape(-1)
            discounts_array = np.full(
                len(rewards_array),
                self.gamma,
                dtype=np.float64,
            )

        # Compute normalised importance-sampling weights.
        weights_array = (self.capacity * probs) ** -self.beta
        weights_array = weights_array / weights_array.max()

        # Convert the sampled data to tensors.
        states_tensor = states.to(
            dtype=torch.float32,
            device=self.device,
        )
        next_states_tensor = n_states.to(
            dtype=torch.float32,
            device=self.device,
        )
        rewards_tensor = torch.as_tensor(
            rewards_array,
            dtype=torch.float32,
            device=self.device,
        )
        dones_tensor = torch.as_tensor(
            dones_array,
            dtype=torch.bool,
            device=self.device,
        )
        actions_tensor = torch.as_tensor(
            actions_array,
            dtype=torch.int64,
            device=self.device,
        )
        weights_tensor = torch.as_tensor(
            weights_array,
            dtype=torch.float32,
            device=self.device,
        )
        discounts_tensor = torch.as_tensor(
            discounts_array,
            dtype=torch.float32,
            device=self.device,
        )

        batch = PERReplayBufferSamples(
            observations=states_tensor,
            actions=actions_tensor,
            next_observations=next_states_tensor,
            dones=dones_tensor,
            rewards=rewards_tensor,
            idxs=tree_idxs,
            weights=weights_tensor,
            discounts=discounts_tensor,
        )
        return batch

    def compute_discounted_rewards_batch(
        self,
        rewards_batch: np.ndarray,
        dones_batch: np.ndarray,
        truns_batch: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute n-step discounted returns for a batch of transitions.

        :param rewards_batch: Rewards with shape ``(batch_size, n_step)``.
        :param dones_batch: Termination flags with shape ``(batch_size, n_step)``.
        :param truns_batch: Truncation flags with shape ``(batch_size, n_step)``.
        :return: Discounted returns, cumulative termination flags and bootstrap
        discounts for each transition.
        """
        batch_size, n_step = rewards_batch.shape
        discounted_rewards = np.zeros(
            batch_size,
            dtype=np.float64,
        )
        cumulative_dones = np.zeros(
            batch_size,
            dtype=bool,
        )
        discounts = np.full(
            batch_size,
            self.gamma**n_step,
            dtype=np.float64,
        )

        for i in range(batch_size):
            cumulative_discount = 1.0
            for j in range(n_step):
                discounted_rewards[i] += cumulative_discount * rewards_batch[i, j]
                if dones_batch[i, j] == 1:
                    cumulative_dones[i] = True
                    break
                elif truns_batch[i, j] == 1:
                    # Truncated after j + 1 real steps: n_state is the final observation,
                    # so bootstrap with gamma^(j + 1) rather than gamma^n.
                    discounts[i] = cumulative_discount * self.gamma
                    break
                cumulative_discount *= self.gamma

        return discounted_rewards, cumulative_dones, discounts

    def update_priorities(
        self,
        idxs: np.ndarray,
        priorities: np.ndarray,
    ) -> None:
        """
        Update the priorities of sampled transitions.

        :param idxs: Sum-tree indices of the sampled transitions.
        :param priorities: New priority values for the sampled transitions.
        """
        priorities = priorities + self.eps
        self.max_prio = max(self.max_prio, np.max(priorities))
        self.st.update(idxs, priorities**self.alpha)
