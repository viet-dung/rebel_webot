import copy
import warnings
from typing import Any, Optional, Union

import numpy as np
import torch as th
from gymnasium import spaces

from stable_baselines3.common.buffers import DictReplayBuffer
from stable_baselines3.common.type_aliases import DictReplayBufferSamples
from stable_baselines3.common.vec_env import VecEnv, VecNormalize
from stable_baselines3.her.goal_selection_strategy import KEY_TO_GOAL_STRATEGY, GoalSelectionStrategy
from stable_baselines3.her import HerReplayBuffer

class CustomHerReplayBuffer(HerReplayBuffer):
    # def __init__(
    #     self,
    #     buffer_size: int,
    #     observation_space: spaces.Dict, # This is the observation_space from the env
    #     action_space: spaces.Space,     # This is the action_space from the env
    #     env: VecEnv,                    # This is the VecEnv instance the agent uses
    #     device: Union[th.device, str] = "auto",
    #     n_envs: int = 1,
    #     optimize_memory_usage: bool = False,
    #     handle_timeout_termination: bool = True,
    #     n_sampled_goal: int = 4,
    #     goal_selection_strategy: Union[GoalSelectionStrategy, str] = "future",
    #     copy_info_dict: bool = False,
    #     box_move_tolerance: float = 1e-4, # Your custom argument
    # ):
    #     # This calls HerReplayBuffer.__init__
    #     # Ensure ALL arguments expected by HerReplayBuffer.__init__ are present
    #     # and in the correct order if positional, or correctly named if keyword.
    #     super().__init__(
    #         buffer_size=buffer_size,                # Positional in HerReplayBuffer
    #         observation_space=observation_space,    # Positional in HerReplayBuffer
    #         action_space=action_space,              # Positional in HerReplayBuffer
    #         env=env,                                # Positional in HerReplayBuffer
    #         device=device,                          # Keyword in HerReplayBuffer
    #         n_envs=n_envs,                          # Keyword in HerReplayBuffer
    #         optimize_memory_usage=optimize_memory_usage, # Keyword
    #         handle_timeout_termination=handle_timeout_termination, # Keyword
    #         n_sampled_goal=n_sampled_goal,          # Keyword
    #         goal_selection_strategy=goal_selection_strategy, # Keyword
    #         copy_info_dict=copy_info_dict           # Keyword
    #     )

    #     # Your custom attributes
    #     self.episode_has_box_movement = np.zeros((self.buffer_size, self.n_envs), dtype=bool)
    #     self.box_move_tolerance = box_move_tolerance

    # def add(  # type: ignore[override]
    #     self,
    #     obs: dict[str, np.ndarray],
    #     next_obs: dict[str, np.ndarray],
    #     action: np.ndarray,
    #     reward: np.ndarray,
    #     done: np.ndarray,
    #     infos: list[dict[str, Any]],
    # ) -> None:
    #     # When the buffer is full, we rewrite on old episodes. When we start to
    #     # rewrite on an old episodes, we want the whole old episode to be deleted
    #     # (and not only the transition on which we rewrite). To do this, we set
    #     # the length of the old episode to 0, so it can't be sampled anymore.
    #     for env_idx in range(self.n_envs):
    #         episode_start = self.ep_start[self.pos, env_idx]
    #         episode_length = self.ep_length[self.pos, env_idx]
    #         if episode_length > 0:
    #             episode_end = episode_start + episode_length
    #             episode_indices = np.arange(self.pos, episode_end) % self.buffer_size
    #             self.ep_length[episode_indices, env_idx] = 0
    #             # when you detect an overwrite and reset ep_length for the old episode, you also need to reset the episode_has_box_movement flag for those indices.
    #             self.episode_has_box_movement[episode_indices, env_idx] = False

    #     # Update episode start
    #     self.ep_start[self.pos] = self._current_ep_start.copy()

    #     if self.copy_info_dict:
    #         self.infos[self.pos] = infos  # type: ignore[assignment]
    #     # Store the transition
    #     super().add(obs, next_obs, action, reward, done, infos)

    #     # When episode ends, compute and store the episode length
    #     for env_idx in range(self.n_envs):
    #         if done[env_idx]:
    #             self._compute_episode_length(env_idx)

    # def _compute_episode_length(self, env_idx: int) -> None:
    #     """
    #     Compute and store the episode length and check for box movement
    #     between the start and end of the episode for environment with index env_idx.
    #     """
    #     episode_start_idx = self._current_ep_start[env_idx] # Index in buffer for the start of the episode
    #     # The current `self.pos` is the index *after* the last transition of the episode.
    #     # So, the last transition's index is `(self.pos - 1 + self.buffer_size) % self.buffer_size`.
    #     episode_last_transition_idx = (self.pos - 1 + self.buffer_size) % self.buffer_size

    #     actual_episode_end_buffer_idx = self.pos # Where the next episode will start writing
    #     if actual_episode_end_buffer_idx < episode_start_idx:
    #         # Buffer has wrapped around
    #         current_episode_length = (actual_episode_end_buffer_idx + self.buffer_size) - episode_start_idx
    #     else:
    #         current_episode_length = actual_episode_end_buffer_idx - episode_start_idx

    #     # --- Check for Box Movement (First vs. Last Step) ---
    #     box_moved_in_episode = False
    #     if current_episode_length > 1: # Only check if episode has more than one step
    #         # Achieved goal at the START of the episode (from the first transition's next_observation)
    #         # self.next_observations["achieved_goal"] stores s_1, s_2, ..., s_T
    #         # If episode_start_idx is the index of (o_0, a_0, r_0, o_1), then o_1["achieved_goal"] is at this index.
    #         first_step_box_pos = self.next_observations["achieved_goal"][episode_start_idx, env_idx][-1]

    #         # Achieved goal at the END of the episode (from the last transition's next_observation)
    #         last_step_box_pos = self.next_observations["achieved_goal"][episode_last_transition_idx, env_idx][-1]

    #         # Calculate Euclidean distance between the first and last box positions
    #         # Ensure they are numpy arrays for norm calculation
    #         distance = np.linalg.norm(np.array(last_step_box_pos) - np.array(first_step_box_pos))
    #         if distance > self.box_move_tolerance:
    #             box_moved_in_episode = True
    #     # --- End Check ---

    #     # Calculate episode indices for setting length and movement flag
    #     # These are all the indices in the buffer that belong to the just-finished episode
    #     episode_buffer_indices = np.arange(episode_start_idx, actual_episode_end_buffer_idx) % self.buffer_size
    #     if actual_episode_end_buffer_idx < episode_start_idx: # Handle wrap-around for arange
    #          episode_buffer_indices = np.concatenate([
    #              np.arange(episode_start_idx, self.buffer_size),
    #              np.arange(0, actual_episode_end_buffer_idx)
    #          ])


    #     # Store episode length for all transitions in this episode
    #     self.ep_length[episode_buffer_indices, env_idx] = current_episode_length

    #     # Store the movement flag for all transitions in this episode
    #     self.episode_has_box_movement[episode_buffer_indices, env_idx] = box_moved_in_episode

    #     # Update the current episode start pointer for the *next* episode in this env
    #     self._current_ep_start[env_idx] = self.pos


    # def _get_virtual_samples(
    #     self,
    #     batch_indices: np.ndarray,
    #     env_indices: np.ndarray,
    #     env: Optional[VecNormalize] = None,
    # ) -> DictReplayBufferSamples:
    #     """
    #     Get the samples, sample new desired goals and compute new rewards.

    #     :param batch_indices: Indices of the transitions
    #     :param env_indices: Indices of the environments
    #     :param env: associated gym VecEnv to normalize the
    #         observations/rewards when sampling, defaults to None
    #     :return: Samples, with new desired goals and new rewards
    #     """
    #     # Get infos and obs
    #     obs = {key: obs[batch_indices, env_indices, :] for key, obs in self.observations.items()}
    #     next_obs = {key: obs[batch_indices, env_indices, :] for key, obs in self.next_observations.items()}
    #     if self.copy_info_dict:
    #         # The copy may cause a slow down
    #         infos = copy.deepcopy(self.infos[batch_indices, env_indices])
    #     else:
    #         infos = [{} for _ in range(len(batch_indices))]

    #     # --- Get ORIGINAL goals and rewards for potential fallback ---
    #     original_desired_goals = self.observations["desired_goal"][batch_indices, env_indices]
    #     original_rewards = self.rewards[batch_indices, env_indices]
    #     original_dones = self.dones[batch_indices, env_indices] * (1 - self.timeouts[batch_indices, env_indices])
    #     # --- End Original Fetch ---

    #     # --- Determine which samples should be relabeled ---
    #     # Check the flag for the episodes these transitions belong to
    #     should_relabel_mask = self.episode_has_box_movement[batch_indices, env_indices]
    #     # --- End Determination ---

    #     # --- Always sample potential new goals (needed for indexing consistency) ---
    #     # Even if not used for all samples, sampling maintains array shapes.
    #     sampled_new_goals = self._sample_goals(batch_indices, env_indices)
    #     # --- End Sampling ---

    #     # Choose goals: new ones if should_relabel, otherwise original
    #     final_desired_goals = np.where(
    #         should_relabel_mask[..., None], # Expand mask dims to match goal dims
    #         sampled_new_goals,
    #         original_desired_goals
    #     )

    #     # Update the observation dictionary with the final desired goals
    #     obs["desired_goal"] = final_desired_goals
    #     # The desired goal for the next observation must be the same
    #     next_obs["desired_goal"] = final_desired_goals

    #     # --- Observation modification based on new_goals (40-element structure) ---
    #     # Components from the NEW desired goal
    #     new_box_xyz_from_goal = obs["desired_goal"][..., :3]
    #     new_target_box_center_rot_from_goal = obs["desired_goal"][..., 3]

    #     # Derived property from the NEW desired goal's box_xyz
    #     new_theta_for_box = np.arctan2(new_box_xyz_from_goal[..., 1], new_box_xyz_from_goal[..., 0])
    #     new_box_center_rot = -new_theta_for_box / np.pi # This is the recalculated box_center_rot

    #     # --- Update obs["observation"] (current timestep t) ---
    #     current_pinch_pose_t_xyz = obs["observation"][..., 14:17]  # XYZ part of current_pinch_pose from obs
    #     normalized_joint_pos0_t = obs["observation"][..., 28]       # First joint pos from obs

    #     # Update ONLY the XYZ part (0:3) of box_gripper_delta. Elements 3:7 remain as sampled.
    #     obs["observation"][..., 0:3] = (new_box_xyz_from_goal - current_pinch_pose_t_xyz) / 2.0

    #     obs["observation"][..., 7] = (new_target_box_center_rot_from_goal - new_box_center_rot) / 2.0 # target_delta
    #     obs["observation"][..., 8] = (new_box_center_rot - normalized_joint_pos0_t) / 2.0      # joint_box_delta
    #     obs["observation"][..., 9] = new_box_center_rot                                        # box_center_rot (recalculated)
    #     obs["observation"][..., 10] = np.linalg.norm(new_box_xyz_from_goal, axis=-1)/2                # box_center_l2_norm (of new goal's box_xyz)
    #     obs["observation"][..., 11] = np.linalg.norm(new_box_xyz_from_goal - current_pinch_pose_t_xyz, axis=-1)/2 # box_gripper_l2_norm (XYZ based)
    #     obs["observation"][..., 21:24] = new_box_xyz_from_goal                                      # Update XYZ part of box_pose in observation

    #     # --- Update next_obs["observation"] (next timestep t+1) ---
    #     current_pinch_pose_tplus1_xyz = next_obs["observation"][..., 14:17] # XYZ part of current_pinch_pose from next_obs
    #     normalized_joint_pos0_tplus1 = next_obs["observation"][..., 28]     # First joint pos from next_obs

    #     # Update ONLY the XYZ part (0:3) of box_gripper_delta. Elements 3:7 remain as sampled.
    #     next_obs["observation"][..., 0:3] = (new_box_xyz_from_goal - current_pinch_pose_tplus1_xyz) / 2.0

    #     next_obs["observation"][..., 7] = (new_target_box_center_rot_from_goal - new_box_center_rot) / 2.0 # target_delta
    #     next_obs["observation"][..., 8] = (new_box_center_rot - normalized_joint_pos0_tplus1) / 2.0  # joint_box_delta
    #     next_obs["observation"][..., 9] = new_box_center_rot                                         # box_center_rot
    #     next_obs["observation"][..., 10] = np.linalg.norm(new_box_xyz_from_goal, axis=-1)/2                 # box_center_l2_norm
    #     next_obs["observation"][..., 11] = np.linalg.norm(new_box_xyz_from_goal - current_pinch_pose_tplus1_xyz, axis=-1)/2 # box_gripper_l2_norm (XYZ based)
    #     next_obs["observation"][..., 21:24] = new_box_xyz_from_goal  

    #     assert (
    #         self.env is not None
    #     ), "You must initialize HerReplayBuffer with a VecEnv so it can compute rewards for virtual transitions"
    #     # Compute new reward
    #     # --- Compute potential new rewards (only needed if relabeling) ---
    #     # We'll compute them based on the sampled_new_goals, then select later.
    #     # Temporarily set desired goals in obs to the sampled ones for reward computation
    #     obs_for_reward_calc = obs.copy() # Avoid modifying the main 'obs' dict yet
    #     obs_for_reward_calc["desired_goal"] = sampled_new_goals

    #     # Compute rewards as if all samples were relabeled
    #     computed_rewards = self.env.env_method(
    #         "compute_reward",
    #         next_obs["achieved_goal"],
    #         obs_for_reward_calc["desired_goal"], # Use the sampled goals here
    #         infos,
    #         indices=[0],
    #     )[0].astype(np.float32)
    #     # --- End Reward Computation ---
        
    #     # Choose rewards: computed ones if should_relabel, otherwise original
    #     rewards = np.where(
    #         should_relabel_mask,
    #         computed_rewards,
    #         original_rewards
    #     )

    #     obs = self._normalize_obs(obs, env)  # type: ignore[assignment]
    #     next_obs = self._normalize_obs(next_obs, env)  # type: ignore[assignment]

    #     # Convert to torch tensor
    #     observations = {key: self.to_torch(obs) for key, obs in obs.items()}
    #     next_observations = {key: self.to_torch(obs) for key, obs in next_obs.items()}

    #     return DictReplayBufferSamples(
    #         observations=observations,
    #         actions=self.to_torch(self.actions[batch_indices, env_indices]),
    #         next_observations=next_observations,
    #         # Only use dones that are not due to timeouts
    #         # deactivated by default (timeouts is initialized as an array of False)
    #         dones=self.to_torch(
    #             self.dones[batch_indices, env_indices] * (1 - self.timeouts[batch_indices, env_indices])
    #         ).reshape(-1, 1),
    #         rewards=self.to_torch(self._normalize_reward(rewards.reshape(-1, 1), env)),  # type: ignore[attr-defined]
    #     )

    def _get_virtual_samples(
        self,
        batch_indices: np.ndarray,
        env_indices: np.ndarray,
        env: Optional[VecNormalize] = None,
    ) -> DictReplayBufferSamples:
        """
        Get the samples, sample new desired goals and compute new rewards.

        :param batch_indices: Indices of the transitions
        :param env_indices: Indices of the environments
        :param env: associated gym VecEnv to normalize the
            observations/rewards when sampling, defaults to None
        :return: Samples, with new desired goals and new rewards
        """
        # Get infos and obs
        obs = {key: obs[batch_indices, env_indices, :] for key, obs in self.observations.items()}
        next_obs = {key: obs[batch_indices, env_indices, :] for key, obs in self.next_observations.items()}
        if self.copy_info_dict:
            # The copy may cause a slow down
            infos = copy.deepcopy(self.infos[batch_indices, env_indices])
        else:
            infos = [{} for _ in range(len(batch_indices))]
        # Sample and set new goals
        new_goals = self._sample_goals(batch_indices, env_indices)
        obs["desired_goal"] = new_goals
        # The desired goal for the next observation must be the same as the previous one
        next_obs["desired_goal"] = new_goals

        # --- Observation modification based on new_goals (40-element structure) ---
        # Components from the NEW desired goal
        new_box_xyz_from_goal = obs["desired_goal"][..., :3]
        new_target_box_center_rot_from_goal = obs["desired_goal"][..., 3]

        # Derived property from the NEW desired goal's box_xyz
        new_theta_for_box = np.arctan2(new_box_xyz_from_goal[..., 1], new_box_xyz_from_goal[..., 0])
        new_box_center_rot = -new_theta_for_box / np.pi # This is the recalculated box_center_rot

        # --- Update obs["observation"] (current timestep t) ---
        current_pinch_pose_t_xyz = obs["observation"][..., 14:17]  # XYZ part of current_pinch_pose from obs
        normalized_joint_pos0_t = obs["observation"][..., 28]       # First joint pos from obs

        # Update ONLY the XYZ part (0:3) of box_gripper_delta. Elements 3:7 remain as sampled.
        obs["observation"][..., 0:3] = (new_box_xyz_from_goal - current_pinch_pose_t_xyz) / 2.0

        obs["observation"][..., 7] = (new_target_box_center_rot_from_goal - new_box_center_rot) / 2.0 # target_delta
        obs["observation"][..., 8] = (new_box_center_rot - normalized_joint_pos0_t) / 2.0      # joint_box_delta
        obs["observation"][..., 9] = new_box_center_rot                                        # box_center_rot (recalculated)
        obs["observation"][..., 10] = np.linalg.norm(new_box_xyz_from_goal, axis=-1)/2                # box_center_l2_norm (of new goal's box_xyz)
        obs["observation"][..., 11] = np.linalg.norm(new_box_xyz_from_goal - current_pinch_pose_t_xyz, axis=-1)/2 # box_gripper_l2_norm (XYZ based)
        obs["observation"][..., 21:24] = new_box_xyz_from_goal                                      # Update XYZ part of box_pose in observation

        # --- Update next_obs["observation"] (next timestep t+1) ---
        current_pinch_pose_tplus1_xyz = next_obs["observation"][..., 14:17] # XYZ part of current_pinch_pose from next_obs
        normalized_joint_pos0_tplus1 = next_obs["observation"][..., 28]     # First joint pos from next_obs

        # Update ONLY the XYZ part (0:3) of box_gripper_delta. Elements 3:7 remain as sampled.
        next_obs["observation"][..., 0:3] = (new_box_xyz_from_goal - current_pinch_pose_tplus1_xyz) / 2.0

        next_obs["observation"][..., 7] = (new_target_box_center_rot_from_goal - new_box_center_rot) / 2.0 # target_delta
        next_obs["observation"][..., 8] = (new_box_center_rot - normalized_joint_pos0_tplus1) / 2.0  # joint_box_delta
        next_obs["observation"][..., 9] = new_box_center_rot                                         # box_center_rot
        next_obs["observation"][..., 10] = np.linalg.norm(new_box_xyz_from_goal, axis=-1)/2                 # box_center_l2_norm
        next_obs["observation"][..., 11] = np.linalg.norm(new_box_xyz_from_goal - current_pinch_pose_tplus1_xyz, axis=-1)/2 # box_gripper_l2_norm (XYZ based)
        next_obs["observation"][..., 21:24] = new_box_xyz_from_goal  

        assert (
            self.env is not None
        ), "You must initialize HerReplayBuffer with a VecEnv so it can compute rewards for virtual transitions"
        # Compute new reward
        rewards = self.env.env_method(
            "compute_reward",
            # the new state depends on the previous state and action
            # s_{t+1} = f(s_t, a_t)
            # so the next achieved_goal depends also on the previous state and action
            # because we are in a GoalEnv:
            # r_t = reward(s_t, a_t) = reward(next_achieved_goal, desired_goal)
            # therefore we have to use next_obs["achieved_goal"] and not obs["achieved_goal"]
            next_obs["achieved_goal"],
            # here we use the new desired goal
            obs["desired_goal"],
            infos,
            # we use the method of the first environment assuming that all environments are identical.
            indices=[0],
        )
        rewards = rewards[0].astype(np.float32)  # env_method returns a list containing one element
        obs = self._normalize_obs(obs, env)  # type: ignore[assignment]
        next_obs = self._normalize_obs(next_obs, env)  # type: ignore[assignment]

        # Convert to torch tensor
        observations = {key: self.to_torch(obs) for key, obs in obs.items()}
        next_observations = {key: self.to_torch(obs) for key, obs in next_obs.items()}

        return DictReplayBufferSamples(
            observations=observations,
            actions=self.to_torch(self.actions[batch_indices, env_indices]),
            next_observations=next_observations,
            # Only use dones that are not due to timeouts
            # deactivated by default (timeouts is initialized as an array of False)
            dones=self.to_torch(
                self.dones[batch_indices, env_indices] * (1 - self.timeouts[batch_indices, env_indices])
            ).reshape(-1, 1),
            rewards=self.to_torch(self._normalize_reward(rewards.reshape(-1, 1), env)),  # type: ignore[attr-defined]
        )