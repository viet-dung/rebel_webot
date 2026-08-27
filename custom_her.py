import warnings
from typing import Any, Optional, Union

import numpy as np
import torch as th
from gymnasium import spaces
import copy

# Import the original HerReplayBuffer to inherit from it
from stable_baselines3.her.her_replay_buffer import HerReplayBuffer
from stable_baselines3.common.type_aliases import DictReplayBufferSamples
from stable_baselines3.common.vec_env import VecNormalize

def default_push_reward_fn(achieved_goal: np.ndarray, desired_goal: np.ndarray) -> np.ndarray:
    """
    Computes a sparse reward for a pushing task. Reward is 1.0 if the object (in achieved_goal)
    is close to the target (in desired_goal), -1.0 otherwise.
    Assumes goal format is [..., object_pos_x, object_pos_y, ...].
    """
    current_box_xy = achieved_goal[:, 7:9]
    target_box_xy = desired_goal[:, 7:9]
    check_xy = np.all(np.abs(target_box_xy - current_box_xy) <= 0.02, axis=1)

    # Return a batch of rewards: 1.0 for success, -1.0 for failure.
    reward = np.where(check_xy, 1.0, -1.0).astype(np.float32)
    return reward

def recompute_observation(observation_batch: np.ndarray, new_box_target_batch: np.ndarray) -> np.ndarray:
    modified_obs_batch = observation_batch.copy()
    
    target_delta_idx = 7
    box_pose_start_idx = 21
    target_vector_start_idx = 59
    box_target_dist_idx = 62
    theta_box_target_idx = 63

    box_pose_batch = observation_batch[:, box_pose_start_idx : box_pose_start_idx + 7]
    box_pos_batch = box_pose_batch[:, :3]

    # Vector from current box to new target
    delta_vector = new_box_target_batch - box_pos_batch
    delta_x = delta_vector[:, 0]
    delta_y = delta_vector[:, 1]
    
    # --- Recompute all features based on this single delta_vector ---
    # 1. 'target_vector'
    modified_obs_batch[:, target_vector_start_idx : target_vector_start_idx + 3] = delta_vector / 2

    # 2. 'box_target_distance'
    box_target_distance = np.linalg.norm(delta_vector[:, :2], axis=1) / 2
    modified_obs_batch[:, box_target_dist_idx] = box_target_distance
    
    # 3. 'theta_box_target'
    # This will now correctly be 0 if distance is 0.
    theta_box_target = -np.arctan2(delta_y, delta_x) / np.pi
    modified_obs_batch[:, theta_box_target_idx] = theta_box_target

    # 4. 'target_delta' (rely on the same inputs)
    box_y_orig, box_x_orig = box_pos_batch[:, 1], box_pos_batch[:, 0]
    box_center_rot_batch = -np.arctan2(box_y_orig, box_x_orig) / np.pi
    target_y, target_x = new_box_target_batch[:, 1], new_box_target_batch[:, 0]
    target_box_center_rot_batch = -np.arctan2(target_y, target_x) / np.pi
    modified_obs_batch[:, target_delta_idx] = (target_box_center_rot_batch - box_center_rot_batch) / 2

    return modified_obs_batch


class CustomHerReplayBuffer(HerReplayBuffer):
    """
    A corrected custom HER replay buffer that inherits from the original SB3 HerReplayBuffer.

    It makes two critical modifications:
    1.  **Episode Filtering:** In the `add()` method, it checks `info["box_was_pushed"]`
        before marking an episode as sampleable.
    2.  **Consistent Observations:** In `_get_virtual_samples()`, it calls the environment's
        `recompute_observation()` method to make the `observation` vector consistent
        with the new hindsight goal.
    """

    def add(
        self,
        obs: dict[str, np.ndarray],
        next_obs: dict[str, np.ndarray],
        action: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
        infos: list[dict[str, Any]],
    ) -> None:
        # This is the original logic from the parent SB3 HerReplayBuffer.
        # It's crucial for correctly handling buffer wrapping.
        for env_idx in range(self.n_envs):
            if self.ep_length[self.pos, env_idx] > 0:
                episode_start = self.ep_start[self.pos, env_idx]
                episode_length = self.ep_length[self.pos, env_idx]
                episode_end = episode_start + episode_length
                episode_indices = np.arange(self.pos, episode_end) % self.buffer_size
                self.ep_length[episode_indices, env_idx] = 0

        # Store the start of the episode for the current transition
        self.ep_start[self.pos] = self._current_ep_start.copy()

        # Store the transition data by calling the grandparent's add method
        # This is what the original HerReplayBuffer does.
        self.infos[self.pos] = infos
        super(HerReplayBuffer, self).add(obs, next_obs, action, reward, done, infos)

        # Handle episode finalization with our filtering logic
        for env_idx in range(self.n_envs):
            if done[env_idx]:
                last_actionable_step = infos[env_idx].get("last_actionable_step", -1)

                if last_actionable_step == -1:
                    # Case 1: INVALID episode. Box never moved. Orphan it.
                    self._current_ep_start[env_idx] = self.pos
                else:
                    # Case 2: VALID episode. Truncate and set length.                    
                    # The "true" length of the useful part of the episode
                    # is up to and including the step *after* the last action.
                    # Episode steps are 0-indexed, length is 1-indexed.
                    true_length = last_actionable_step + 1

                    episode_start = self._current_ep_start[env_idx]
                    
                    # Get the absolute buffer indices for the USEFUL part of the episode
                    true_episode_indices = (episode_start + np.arange(true_length)) % self.buffer_size
                    
                    # Set the length for only these useful transitions
                    self.ep_length[true_episode_indices, env_idx] = true_length

                    self._current_ep_start[env_idx] = self.pos
                    
    def sample(self, batch_size: int, env: Optional[VecNormalize] = None) -> DictReplayBufferSamples:
        """
        Sample elements from the replay buffer.
        """
        # Determine how many real vs. virtual samples we want
        nb_virtual = int(self.her_ratio * batch_size)
        nb_real = batch_size - nb_virtual

        # Sample indices for real and potential virtual transitions
        # Note: We over-sample virtual candidates in case some fail
        potential_virtual_indices, real_indices = self._sample_indices(nb_virtual, nb_real)
        
        # --- Get Real Samples (this part is standard) ---
        real_batch_indices, real_env_indices = np.unravel_index(real_indices, self.ep_length.shape)
        real_data = self._get_real_samples(real_batch_indices, real_env_indices, env)

        # --- Get Virtual Samples (with "discard and resample" logic) ---
        virtual_batch_indices, virtual_env_indices = np.unravel_index(potential_virtual_indices, self.ep_length.shape)
        
        # Try to create virtual samples
        virtual_data, success_mask = self._get_virtual_samples(virtual_batch_indices, virtual_env_indices, env)
        
        # Figure out how many virtual samples failed and need to be replaced
        nb_failures = nb_virtual - np.sum(success_mask)
        
        if nb_failures > 0:
            # We need to sample more REAL transitions to fill the batch
            # print(f"Resampling {nb_failures} real transitions to replace failed virtual ones.")
            failure_indices, _ = self._sample_indices(nb_failures)
            fail_batch_indices, fail_env_indices = np.unravel_index(failure_indices, self.ep_length.shape)
            replacement_data = self._get_real_samples(fail_batch_indices, fail_env_indices, env)

            # Concatenate the successful virtual data with the new real data
            if virtual_data is not None:
                observations = {k: th.cat((virtual_data.observations[k], replacement_data.observations[k])) for k in virtual_data.observations.keys()}
                actions = th.cat((virtual_data.actions, replacement_data.actions))
                next_observations = {k: th.cat((virtual_data.next_observations[k], replacement_data.next_observations[k])) for k in virtual_data.next_observations.keys()}
                dones = th.cat((virtual_data.dones, replacement_data.dones))
                rewards = th.cat((virtual_data.rewards, replacement_data.rewards))
            else: # All virtual samples failed
                observations = replacement_data.observations
                actions = replacement_data.actions
                next_observations = replacement_data.next_observations
                dones = replacement_data.dones
                rewards = replacement_data.rewards
        else:
            # All virtual samples were successful
            observations = virtual_data.observations
            actions = virtual_data.actions
            next_observations = virtual_data.next_observations
            dones = virtual_data.dones
            rewards = virtual_data.rewards

        # Concatenate the final virtual batch with the initial real batch
        final_observations = {key: th.cat((real_data.observations[key], observations[key])) for key in real_data.observations.keys()}
        final_actions = th.cat((real_data.actions, actions))
        final_next_observations = {key: th.cat((real_data.next_observations[key], next_observations[key])) for key in real_data.next_observations.keys()}
        final_dones = th.cat((real_data.dones, dones))
        final_rewards = th.cat((real_data.rewards, rewards))

        return DictReplayBufferSamples(
            observations=final_observations,
            actions=final_actions,
            next_observations=final_next_observations,
            dones=final_dones,
            rewards=final_rewards,
        )
    
    def _sample_indices(self, nb_virtual, nb_real=0):
        """Helper method to sample valid indices from the buffer."""
        is_valid = self.ep_length > 0
        if not np.any(is_valid):
            raise RuntimeError("No valid episodes in buffer.")
        
        valid_indices = np.flatnonzero(is_valid)
        total_samples = nb_virtual + nb_real
        
        sampled_indices = np.random.choice(valid_indices, size=total_samples, replace=True)
        
        if nb_real > 0:
            return np.split(sampled_indices, [nb_virtual])
        else:
            return sampled_indices, None

    
    def _get_virtual_samples(
        self,
        batch_indices: np.ndarray,
        env_indices: np.ndarray,
        env: Optional[VecNormalize] = None,
    ) -> tuple[Optional[DictReplayBufferSamples], np.ndarray]:
        """
        Tries to get virtual samples by finding actionable goals in a fully vectorized manner.
        """
        # Set up logging
    
        # self.has_logged_example = False
        # self.log_path = "her_log_vectorized.txt"
        
        # --- 1. Construct Future Timeline and Actionable Mask ---
        ep_starts = self.ep_start[batch_indices, env_indices]
        ep_lengths = self.ep_length[batch_indices, env_indices]
        current_indices_in_episode = (batch_indices - ep_starts) % self.buffer_size

        # Find the maximum possible trajectory length in the batch to shape our arrays
        max_ep_len = np.max(ep_lengths)

        # Create a matrix of episode steps [0, 1, 2, ...]
        # Shape: (batch_size, max_ep_len)
        rel_indices_matrix = np.arange(max_ep_len).reshape(1, -1)

        # Create a mask to identify valid future steps for each sample in the batch
        # A step is a valid future if it's >= the current step and < the episode length
        valid_future_mask = (rel_indices_matrix >= current_indices_in_episode[:, np.newaxis]) & \
                            (rel_indices_matrix < ep_lengths[:, np.newaxis])

        # Get the 'info' dicts for every step in every episode in the batch
        # Shape: (batch_size, max_ep_len)
        abs_indices_matrix = (ep_starts[:, np.newaxis] + rel_indices_matrix) % self.buffer_size
        all_infos = self.infos[abs_indices_matrix, env_indices[:, np.newaxis]]

        # Create a boolean mask of where the box moved
        # Shape: (batch_size, max_ep_len)
        box_moved_mask = np.array([info.get("box_moved_this_step", False) for info in all_infos.flat]).reshape(all_infos.shape)

        # The final actionable mask: a step must be a valid future AND the box must have moved
        actionable_mask = valid_future_mask & box_moved_mask

        # --- 2. Vectorized Random Choice of Actionable Goal ---
        
        # Check which samples have at least one actionable goal
        success_mask = np.any(actionable_mask, axis=1)

        # If no samples were successful at all, we can exit early.
        if not np.any(success_mask):
            return None, success_mask

        # For rows with at least one True, we want to pick one of them randomly.
        # Trick: assign random numbers to True elements, 0 to False, then find the argmax.
        rand_matrix = np.random.rand(*actionable_mask.shape)
        masked_rand = np.where(actionable_mask, rand_matrix, -1)
        # chosen_rel_indices will contain the column index of the max random number for each row
        chosen_rel_indices = np.argmax(masked_rand, axis=1)

        # Use the chosen column indices to get the absolute buffer index for the goal
        goal_indices = abs_indices_matrix[np.arange(len(batch_indices)), chosen_rel_indices]

        # --- 3. Filter and Process Successful Samples ---
        successful_batch_idx = np.where(success_mask)[0]

        # Fetch original data for the successful transitions
        obs = {key: self.observations[key][batch_indices[successful_batch_idx], env_indices[successful_batch_idx]] for key in self.observation_space.keys()}
        next_obs = {key: self.next_observations[key][batch_indices[successful_batch_idx], env_indices[successful_batch_idx]] for key in self.observation_space.keys()}
        actions = self.actions[batch_indices[successful_batch_idx], env_indices[successful_batch_idx]]
        dones = self.dones[batch_indices[successful_batch_idx], env_indices[successful_batch_idx]]

        # Get the new goals for the successful transitions
        new_goals = self.next_observations["achieved_goal"][goal_indices[successful_batch_idx], env_indices[successful_batch_idx]]

        # Recompute observations and rewards
        obs["desired_goal"] = new_goals
        next_obs["desired_goal"] = new_goals
        new_box_target = new_goals[:, 7:10]
        obs["observation"] = recompute_observation(obs["observation"], new_box_target)
        next_obs["observation"] = recompute_observation(next_obs["observation"], new_box_target)
        rewards = default_push_reward_fn(next_obs["achieved_goal"], obs["desired_goal"])
        
        # --- 4. Logging ---
        # if not self.has_logged_example and len(successful_batch_idx) > 0:
        #     # We log the first successful transition we found
        #     log_idx_in_batch = successful_batch_idx[0]
        #     # Get original data for that specific transition before it was modified
        #     original_obs_for_log = {k: self.observations[k][batch_indices[log_idx_in_batch], env_indices[log_idx_in_batch]] for k in self.observation_space.keys()}
        #     original_next_obs_for_log = {k: self.next_observations[k][batch_indices[log_idx_in_batch], env_indices[log_idx_in_batch]] for k in self.observation_space.keys()}
        #     original_rewards_for_log = self.rewards[batch_indices[log_idx_in_batch], env_indices[log_idx_in_batch]]
        #     # Get the modified data for that same transition
        #     virt_obs_for_log = {k: v[0] for k, v in obs.items()}
        #     virt_next_obs_for_log = {k: v[0] for k, v in next_obs.items()}
        #     virt_rewards_for_log = rewards[0]

        #     self.log_first_transition_comparison(
        #         batch_indices[log_idx_in_batch], env_indices[log_idx_in_batch],
        #         original_obs_for_log, original_next_obs_for_log, original_rewards_for_log,
        #         virt_obs_for_log, virt_next_obs_for_log, virt_rewards_for_log,
        #         True, # We know it was successful because we're in this block
        #         goal_indices[log_idx_in_batch]
        #     )
        #     self.has_logged_example = True

        # --- 5. Final Processing to PyTorch Tensors ---
        obs_norm = self._normalize_obs(obs, env)
        next_obs_norm = self._normalize_obs(next_obs, env)
        observations_th = {k: self.to_torch(o) for k, o in obs_norm.items()}
        next_observations_th = {k: self.to_torch(no) for k, no in next_obs_norm.items()}
        actions_th = self.to_torch(actions)
        dones_th = self.to_torch(dones).reshape(-1, 1)
        rewards_th = self.to_torch(self._normalize_reward(rewards.reshape(-1, 1), env))
        
        successful_samples = DictReplayBufferSamples(observations_th, actions_th, next_observations_th, dones_th, rewards_th)
        
        return successful_samples, success_mask

    # The log function needs a slight modification to handle single dictionary inputs for obs
    def log_first_transition_comparison(
        self, 
        batch_idx: int, 
        env_idx: int, 
        real_obs: dict, 
        real_next_obs: dict, 
        real_rewards: float, 
        virt_obs: dict, 
        virt_next_obs: dict, 
        virt_rewards: float, 
        actionable_found: bool, 
        goal_idx: int
    ):
        """
        Logs a detailed comparison between an original transition and its 
        hindsight-modified version. It highlights changes in goals, rewards,
        and specific observation features that are relative to the target.
        """
        # --- Define indices of target-relative features based on get_observation() ---
        # This makes the log robust to changes in the observation space.
        TARGET_DELTA_IDX = 7
        BOX_VELOCITY_START_IDX = 57
        TARGET_VECTOR_START_IDX = 60
        BOX_TARGET_DIST_IDX = 63
        THETA_BOX_TARGET_IDX = 64
        # For goals, the box position is in indices 7:10 of the 'achieved_goal' and 'desired_goal'
        GOAL_BOX_POS_SLICE = slice(7, 10)
        # For the achieved goal at t+1, we only care about the xy position for reward calculation
        ACHIEVED_BOX_POS_SLICE = slice(7, 9)
        achieved_goal_of_goal_state = self.next_observations["achieved_goal"][goal_idx, env_idx]

        # --- Gather Contextual Information ---
        ep_start = self.ep_start[batch_idx, env_idx]
        ep_length = self.ep_length[batch_idx, env_idx]
        current_transition_in_episode = (batch_idx - ep_start) % self.buffer_size
        goal_transition_in_episode = (goal_idx - ep_start) % self.buffer_size
        
        # Get info for the chosen goal state
        chosen_goal_info = self.infos[goal_idx, env_idx]
        goal_box_moved = chosen_goal_info.get("box_moved_this_step", "N/A")

        # Get the full timeline of when the box moved in the episode
        episode_box_moved_flags = [
            self.infos[(ep_start + step) % self.buffer_size, env_idx].get("box_moved_this_step", "N/A")
            for step in range(ep_length)
        ]

        with open(self.log_path, 'a') as f:
            f.write("="*60 + "\n")
            f.write("--- HINDSIGHT EXPERIENCE REPLAY (HER) TRANSITION LOG ---\n")
            f.write("="*60 + "\n\n")

            # --- 1. CONTEXT ---
            f.write("--- 1. SAMPLING CONTEXT ---\n")
            f.write(f"Sampled Transition (s_t):  Buffer Index={batch_idx}, Episode Step={current_transition_in_episode}\n")
            f.write(f"Chosen Hindsight Goal (g): Buffer Index={goal_idx}, Episode Step={goal_transition_in_episode}\n")
            f.write(f"  - Was 'box_moved_this_step' True for goal state? -> {goal_box_moved}\n\n")

            # --- 2. HIGH-LEVEL COMPARISON (GOALS & REWARDS) ---
            f.write("--- 2. GOAL & REWARD COMPARISON ---\n")
            f.write(f"Original Desired Goal (g_orig):   {np.round(real_obs['desired_goal'][7:10], 3)}\n")
            f.write(f"Hindsight Desired Goal (g_new):   {np.round(virt_obs['desired_goal'][7:10], 3)} <-- MODIFIED\n")
            f.write(f"Achieved Goal at t (s_ag):     {np.round(real_obs['achieved_goal'][7:9], 3)}\n\n")
            f.write(f"Achieved Goal at t+1 (s'_ag):     {np.round(real_next_obs['achieved_goal'][7:9], 3)}\n\n")

            f.write("--- 2. GOAL & REWARD COMPARISON ---\n")
            f.write(f"  A) Original Desired Goal (g_orig):      {np.round(real_obs['desired_goal'][GOAL_BOX_POS_SLICE], 3)} (What the agent was supposed to do)\n")
            f.write(f"  B) Achieved Goal of Goal State:         {np.round(achieved_goal_of_goal_state[GOAL_BOX_POS_SLICE], 3)} (The future state we picked)\n")
            f.write(f"  C) Hindsight Desired Goal (g_new):      {np.round(virt_obs['desired_goal'][GOAL_BOX_POS_SLICE], 3)} <-- MODIFIED (Should match B)\n")
            f.write(f"  D) Achieved Goal at t+1 (s'_ag):        {np.round(real_next_obs['achieved_goal'][ACHIEVED_BOX_POS_SLICE], 3)} (What the agent actually did in this step)\n\n")
            
            f.write(f"Original Reward (r_orig):         {real_rewards:.4f}\n")
            f.write(f"Hindsight Reward (r_new):         {virt_rewards:.4f} <-- RECOMPUTED\n\n")

            # --- DETAILED OBSERVATION FEATURE COMPARISON (UPDATED) ---
            f.write("--- 3. OBSERVATION FEATURE COMPARISON ---\n")
            orig_obs_vec = real_obs['observation']
            virt_obs_vec = virt_obs['observation']
            
            f.write("HER-Safe Features (should be UNCHANGED):\n")
            orig_vel = orig_obs_vec[BOX_VELOCITY_START_IDX:BOX_VELOCITY_START_IDX+3]
            virt_vel = virt_obs_vec[BOX_VELOCITY_START_IDX:BOX_VELOCITY_START_IDX+3]
            f.write(f"  - 'box_velocity':          {np.round(orig_vel, 3)}  ->  {np.round(virt_vel, 3)}\n\n")

            f.write("HER-Fixable Features (should be MODIFIED):\n")
            f.write(f"  - 'target_delta':          {orig_obs_vec[TARGET_DELTA_IDX]:.4f}  ->  {virt_obs_vec[TARGET_DELTA_IDX]:.4f}\n")
            orig_vec = orig_obs_vec[TARGET_VECTOR_START_IDX:TARGET_VECTOR_START_IDX+3]
            virt_vec = virt_obs_vec[TARGET_VECTOR_START_IDX:TARGET_VECTOR_START_IDX+3]
            f.write(f"  - 'target_vector':         {np.round(orig_vec, 3)}  ->  {np.round(virt_vec, 3)}\n")
            f.write(f"  - 'box_target_distance':   {orig_obs_vec[BOX_TARGET_DIST_IDX]:.4f}  ->  {virt_obs_vec[BOX_TARGET_DIST_IDX]:.4f}\n")
            f.write(f"  - 'theta_box_target':      {orig_obs_vec[THETA_BOX_TARGET_IDX]:.4f}  ->  {virt_obs_vec[THETA_BOX_TARGET_IDX]:.4f}\n")

            # --- 4. EPISODE TIMELINE ---
            f.write("--- 4. EPISODE TIMELINE ('box_moved_this_step' flags) ---\n")
            for i, flag in enumerate(episode_box_moved_flags):
                marker = ""
                if i == current_transition_in_episode:
                    marker += " <-- CURRENT TRANSITION (s_t)"
                if i == goal_transition_in_episode:
                    marker += " <-- CHOSEN GOAL (g)"
                f.write(f"  Step {i:02d}: {str(flag):<5}{marker}\n")

            f.write("\n" + "="*60 + "\n\n")