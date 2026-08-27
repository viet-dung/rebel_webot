import rclpy
from geometry_msgs.msg import Pose, Point, Quaternion
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from sensor_msgs.msg import JointState
from std_msgs.msg import String, Bool
import threading
from baseEnv import BaseEnv
import numpy as np
from ros2_message_interfaces.msg import ResetSeed 
from utils import *
import os
import time
import shutil
import math
from stable_baselines3 import TD3, SAC

DEBUG = False

class SimulationEnv(BaseEnv):
    def __init__(self, seed=0, port='1234',env_id=0):
        super().__init__(seed=seed, port=port, env_id=env_id)

        rclpy.init()
        self.__node = rclpy.create_node('simulation_env', namespace=f'igus_rebel_{self.port}')

        self.create_publisher()
        self.create_subscription()

        self.noise_levels = {
                'gripper_pos_std': 0.05,  # meters
                'gripper_orient_std': 0.01, # unitless (applied to quaternion components)
                'box_pos_std': 0.01,      # meters
                'box_orient_std': 0.1,    # unitless
                'joint_pos_std': 0.1,     # radians
                'joint_vel_std': 0.1,     # radians/sec
                'delay_std': 0.05
        }
        
        # Configure for repeatability
        self.seed = 0
        set_seed(self.seed)
        self.reset_pub.publish(ResetSeed(reset=True, seed=self.seed))

        # Action delay for sim
        self.last_action = np.zeros(7)
        self.action_smoothing_factor = 0.7
        self.frame_skip = 1

        self.original_box_pose = None
        self.push_vector = np.zeros(3)
        self.target_vector = np.zeros(3)
        self.target_box_pos_custom = None
        self.has_reset_box = False
        self.last_actionable_step = -1
        self.box_velocity = np.zeros(2)
        self.last_distance_to_target = 0.0

        self.wait_period = 1.0/62.5

        # code_dir = os.path.dirname(os.path.abspath(__file__))
        # best_model_path = os.path.join(code_dir, "sac_sim_reach_best", "sac_reach_4200000_stepsgdown.zip")
        # self.reach_model = SAC.load(best_model_path, device="cuda")
        
        log_file = f"env_{self.env_id}_step_log.txt"
        if os.path.exists(log_file):
            os.remove(log_file)
        
        cwd = os.getcwd()
        self.save_path = os.path.abspath(os.path.join(cwd, "imgs"))
        if os.path.exists(self.save_path):
            shutil.rmtree(self.save_path)
            
        self.executor = rclpy.executors.MultiThreadedExecutor()
        self.executor.add_node(self.__node)
        self.executor_thread = threading.Thread(target=self.executor.spin, daemon=True)
        self.executor_thread.start()

    def create_publisher(self):
        # ROS Communication Setup
        self.arm_pub = self.__node.create_publisher(
            JointTrajectory,
            'arm_controller/command',
            10
        )
        self.gripper_pub = self.__node.create_publisher(
            JointTrajectory,
            'gripper_controller/command',
            10
        )
        self.target_box_pub = self.__node.create_publisher(
            String,
            'target_box',
            10
        )
        self.world_reset_pub = self.__node.create_publisher(Bool, 'reset_world', 10)

        self.reset_pub = self.__node.create_publisher(ResetSeed, 'reset_arm', 10)
        self.reset_ack_pub = self.__node.create_publisher(Bool, 'reset_ack', 10)
        self.image_pub = self.__node.create_publisher(ResetSeed, 'capture_image', 10)
        self.reset_box_pub = self.__node.create_publisher(ResetSeed, 'reset_boxes', 10)
        self.box_pub_viz = self.__node.create_publisher(Pose, 'box_pos_viz', 10)

    def create_subscription(self):
        self.__node.create_subscription(Pose, 'pose/gripper', self.gripper_pose_callback, 1)
        self.__node.create_subscription(JointState, 'gripper/joint_states', self.gripper_joint_states_callback, 1)
        self.__node.create_subscription(JointState, 'custom_joint_states', self.custom_joint_states_callback, 1)
        self.__node.create_subscription(Bool, 'reset_arm_done', self.reset_arm_done_callback, 1)
        self.__node.create_subscription(Bool,"reset_box_ack",self.reset_box_ack_callback,1)
        self.__node.create_subscription(Point, 'pos/link7', self.pos_link7_callback, 1)
        self.__node.create_subscription(Point, 'target_box_pos', self.target_pos_callback, 1)

        #self.target_name = 1
        for i in range(self.box_count):
            self.__node.create_subscription(
                Pose,
                f'box_{i}/pose',
                lambda msg, box_id=i: self.create_box_pose_callback(box_id,msg),
                1
            )
            #self.__node.create_subscription(Point, 'pos/target', self.target_pos_callback, 1)


        #self.__node.create_subscription(JointState, 'joint_states', self.joint_states_callback, 1) from ROS controller manager
       
    def close(self):
        self.executor.shutdown()
        self.__node.destroy_node()

    def _apply_noise(self, value: np.ndarray, std_dev: float, is_quaternion: bool = False) -> np.ndarray:
        """Applies Gaussian noise to a value and normalizes if it's a quaternion."""
       
        noisy_value = value.copy() + np.random.normal(0, std_dev, size=value.shape)
        
        if is_quaternion:
            # For quaternions (x, y, z, w) or (w, x, y, z), normalize after adding noise
            norm = np.linalg.norm(noisy_value)
            if norm > 1e-9: # Avoid division by zero
                noisy_value /= norm
            else: # Should be rare, but reset to a default orientation
                if noisy_value.shape[0] == 4: # Assuming [x,y,z,w] or [w,x,y,z]
                    # Your code uses [x,y,z,w] for self.gripper_orient
                    # And [x,y,z,w] for box_orient based on Pose message
                    noisy_value = np.array([0.0, 0.0, 0.0, 1.0]) 
        return noisy_value

    def reset(self, seed=None):
        print(f"Reset {self.current_step} env {self.env_id}")
        
        # Publish reset message with integer seed from init()
        reset_msg = ResetSeed()
        reset_msg.reset = True
        reset_msg.seed = self.seed

        self.grasped_boxes = [False] * self.box_count
        self.current_step = 0
        self.success_mask = False

        self.reset_arm_done = False
        self.reset_box_done = False

        self.last_action = np.zeros(self.action_space.shape)
     
        ## wait for reset to finish
        self.reset_box_pub.publish(reset_msg)  # We not allow after the reset to obtain the same box position
        self.reset_pub.publish(reset_msg)       # Additionally set the self.__flag to false

        # Domain randomization
        #self.frame_skip = np.random.randint(1, 5)  # Random frame skip between 1 and 4
        #self.action_smoothing_factor = np.random.uniform(0.7, 1)  # Random smoothing factor between 0.7 and 1.0

        start_time = time.time()
        while not self.reset_arm_done:
            if (time.time() - start_time) >= 0.0009: 
                self.reset_pub.publish(reset_msg)
                start_time = time.time()
            rclpy.spin_once(self.__node, timeout_sec=self.wait_period)
            
        self.last_sync_timestamp = 0
        self.joint_state_updated = False
        
        # For webot box
        # while not self.reset_box_done: # or np.sum(np.abs(last_box_pose - self.box_poses[self.target_name].copy())) <= 0.01:
        #     rclpy.spin_once(self.__node, timeout_sec=0)

        # limit = 0.5
        # if np.any(self.box_poses[self.target_name].copy()[:2] > limit):
        #     self.reset()

        # for i in range(50):
        #     # 1. Get the current full observation from the environment
        #     full_obs_dict = self.get_observation()
        #     full_observation_vector = full_obs_dict["observation"]

        #     # 3. Adapt the observation for the "reach" model
        #     reach_observation = full_observation_vector[:-7]
        #     achieved_goal = full_obs_dict["achieved_goal"]
        #     desired_goal = full_obs_dict["desired_goal"]

        #     # 2. Check for success condition
        #     current_pinch_pos = achieved_goal[:3]
        #     current_box_pos = achieved_goal[7:10]
        #     distance = np.linalg.norm(current_pinch_pos - current_box_pos)

        #     if distance < 0.15:
        #         break

        #     reach_observation_dict = {
        #         "observation": reach_observation,
        #         "achieved_goal": achieved_goal,
        #         "desired_goal": desired_goal
        #     }

        #     action, _ = self.reach_model.predict(reach_observation_dict, deterministic=True)
        #     self.last_action = action

        #     full_obs_dict, _, _, _, _  = self._step(action)

        #     if i == 49:
        #         action[:] = 0
        #         full_obs_dict, _, _, _, _   = self._step(action)

        #rclpy.spin_once(self.__node, timeout_sec=0)
        self.set_box_pose = True
        final_observation = self.get_observation()
        self.last_actionable_step = -1
        self.current_step = 0
        self.last_sync_timestamp = 0
        self.last_action = np.zeros(self.action_space.shape)
        self.success_mask = False

        if DEBUG:
            log_file = f"env_{self.env_id}_step_log.txt"
            with open(log_file, "a") as f:
                f.write(f"\n\nReset Duration {time.time() - start_time}\n")
                f.write(f"Reset Gripper Position: {self.gripper_pos}\n")
                f.write(f"Reset Box Positions: {[pose[:3] for pose in self.box_poses]}\n")

        return final_observation, {}
    
    def reset_world(self):
        self.world_reset_pub.publish(Bool(data=True))

    def send_action(self, joint_vels, gripper_vel):
        ### ENABLE GRAVITY FOR BOXES
        self.target_box_pub.publish(String(data=f'box_{self.target_name}'))
        
        # Publish arm velocities
        arm_msg = JointTrajectory()
        arm_msg.joint_names = ['joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6']
        arm_msg.points.append(JointTrajectoryPoint(
            velocities=joint_vels.tolist(),
            time_from_start=rclpy.time.Duration(seconds=0.1).to_msg(),
            effort=[self.last_sync_timestamp, float(self.frame_skip), float(self.frame_elapse)] # add the current step number
        ))
        self.arm_pub.publish(arm_msg)

        # Publish gripper velocity
        gripper_msg = JointTrajectory()
        gripper_msg.joint_names = ['gripper_joint']
        gripper_msg.points.append(JointTrajectoryPoint(
            velocities=[gripper_vel],
            time_from_start=rclpy.time.Duration(seconds=0.1).to_msg(),
            effort=(self.last_sync_timestamp*np.ones(6)).tolist() # add the current step number
        ))
        self.gripper_pub.publish(gripper_msg)

    def sample_goal_state(self,phase="grasping"):
        target_pos = []
        if phase == 'grasping':
            workspace_bounds = {
                'x': (0.15, 0.35),  # X-axis bounds (left/right)
                'y': (0.15, 0.35),  # Y-axis bounds (forward/backward)
            }
             
            target_pos = np.array([np.random.uniform(workspace_bounds['x'][0], workspace_bounds['x'][1]),
                            np.random.uniform(workspace_bounds['y'][0], workspace_bounds['y'][1]),
                            0])
        if phase == "stacking":
            target_pos = np.array([0.5,0.5,0.055*self.stack_height + 1])
        return target_pos
    
    def get_observation(self):
         # --- Part 1: Get Current Absolute States (HER-Safe) ---
        gripper_pos = self.gripper_pos.copy()
        gripper_orient = self.gripper_orient.copy()
        gripper_rot_matrix = quaternion_to_rotation_matrix(gripper_orient)
        current_pinch_pose = self.compute_gripper_pinch_pos(gripper_pos, gripper_rot_matrix, gripper_orient)
        
        link7_pos = self.link7_pos.copy()
        # box_pose = self.box_poses[self.target_name].copy()
        # box_pose[2] += 0.1
        box_pose = self.random_box_pose() if self.set_box_pose else self.current_box_pose
        point = Point(x=float(box_pose[0]), y=float(box_pose[1]), z=float(box_pose[2]))
        quaternion = Quaternion(x=float(box_pose[3]), y=float(box_pose[4]), z=float(box_pose[5]), w=float(box_pose[6]))
        self.box_pub_viz.publish(Pose(position=point, orientation=quaternion))

        # Joint states (already normalized, which is good)
        dist_to_limits = []
        normalized_joint_positions = []
        joint_keys = list(self.joint_limits.keys())
        for i, joint_name in enumerate(joint_keys):
            position = self.joint_positions[i]
            min_limit, max_limit = self.joint_limits[joint_name]
            dist_to_min = np.clip(1.0 - (position - min_limit) / ((max_limit - min_limit) / 2), 0, 1)
            dist_to_max = np.clip(1.0 - (max_limit - position) / ((max_limit - min_limit) / 2), 0, 1)
            dist_to_limits.extend([dist_to_min, dist_to_max])
            normalized_position = 2 * (position - min_limit) / (max_limit - min_limit) - 1
            normalized_joint_positions.append(normalized_position)
        
        normalized_joint_positions = np.array(normalized_joint_positions)
        dist_to_limits = np.array(dist_to_limits)

         # Action and gripper states
        action_velocities = self.current_action.copy()
        normalized_gripper_joint_positions = self.gripper_joint_positions.copy()[0]
        normalized_gripper_joint_velocities = action_velocities[-1]

        # --- Part 2: Calculate Goal-Relative Features (HER-Fixable) ---
    
        # These are the core vectors needed for goal-relative features.
        vec_gripper_to_box = box_pose[:3] - current_pinch_pose[:3]
        vec_box_to_target = self.target_box_pos - box_pose[:3]

        # Calculate Push Alignment (new feature)
        norm_g2b = np.linalg.norm(vec_gripper_to_box)
        norm_b2t = np.linalg.norm(vec_box_to_target)
        # This encourages the gripper to point in the direction it needs to push.
        gripper_forward_vec = gripper_rot_matrix[:, 2]  # Assumes Z-axis is the gripper's forward direction
        if norm_g2b > 1e-6 and norm_b2t > 1e-6:
            # Dot product of normalized vectors. -1 means perfect alignment for pushing.
            push_alignment = np.dot(vec_gripper_to_box / norm_g2b, vec_box_to_target / norm_b2t)
            desired_push_dir = vec_box_to_target / norm_b2t
            push_direction_alignment = np.dot(gripper_forward_vec, desired_push_dir)
        else:
            push_alignment = 0.0
            push_direction_alignment = 0.0

        theta = np.arctan2(box_pose[1], box_pose[0])
        box_center_rot = -theta / np.pi
        target_box_center_rot = -np.arctan2(self.target_box_pos[1], self.target_box_pos[0]) / np.pi
        target_delta = (target_box_center_rot - box_center_rot) / 2.0
        joint_box_delta = (box_center_rot - normalized_joint_positions[0]) / 2.0
        box_gripper_delta = (box_pose - current_pinch_pose) / 2.0
        box_center_l2_norm = np.linalg.norm(box_pose[:3]) / 2.0
        box_gripper_l2_norm = norm_g2b / 2.0
        gripper_center_l2_norm = np.linalg.norm(current_pinch_pose[:3]) / 2.0
        theta_g = np.arctan2(current_pinch_pose[1], current_pinch_pose[0])
        gripper_center_rot = -theta_g / np.pi
        target_vector = vec_box_to_target / 2.0
        box_target_distance = norm_b2t / 2.0
        theta_box_target = -np.arctan2(vec_box_to_target[1], vec_box_to_target[0]) / np.pi

        # Cannot has absolute value of things that not controll by the robot, because using HER these values need to be change accordingly.
        observation = np.concatenate([
            # --- Part 1: Relative features (gripper, box, joints) ---
            box_gripper_delta,              # Size: 7, Indices: 0-6
            [target_delta],                 # Size: 1, Index: 7      (HER-Fixable)
            [joint_box_delta],              # Size: 1, Index: 8
            [box_center_rot],               # Size: 1, Index: 9
            [box_center_l2_norm],           # Size: 1, Index: 10
            [box_gripper_l2_norm],          # Size: 1, Index: 11
            [gripper_center_l2_norm],       # Size: 1, Index: 12
            [gripper_center_rot],           # Size: 1, Index: 13
            
            # --- Part 2: Absolute physical states (agent and box) ---
            current_pinch_pose,             # Size: 7, Indices: 14-20
            box_pose,                       # Size: 7, Indices: 21-27
            normalized_joint_positions,     # Size: 6, Indices: 28-33
            dist_to_limits,                 # Size: 12, Indices: 34-45
            [normalized_gripper_joint_positions], # Size: 1, Index: 46
            [normalized_gripper_joint_velocities],# Size: 1, Index: 47
            action_velocities[:-1],         # Size: 6, Indices: 48-53
            link7_pos,                      # Size: 3, Indices: 54-56

            # --- Part 3: Dynamics and Goal-Relative Features (at the end) ---
            self.box_velocity,              # Size: 2, Indices: 57-58 (HER-safe)
            target_vector,                  # Size: 3, Indices: 59-61 (HER-Fixable)
            [box_target_distance],          # Size: 1, Index: 62      (HER-Fixable)
            [theta_box_target],              # Size: 1, Index: 63      (HER-Fixable)
            [push_alignment],                 # Size: 1, Index: 64      (HER-Fixable)
            [push_direction_alignment]     # Size: 1, Index: 65      (HER-Fixable)

        ])

        achieved_goal = np.concatenate([
            current_pinch_pose,        # Current pinch point position (x, y, z)
            box_pose[:3],
            link7_pos
        ])

        desired_goal = np.concatenate([
            box_pose,        # Target pinch point position (x, y, z)
            self.target_box_pos,
            link7_pos
        ])
        return {
            "observation": observation,
            "achieved_goal": achieved_goal,  
            "desired_goal": desired_goal     
        }
    
    def compute_reward( self, achieved_goal: np.ndarray, desired_goal: np.ndarray, info: dict) -> float:
        # Convert inputs to numpy arrays
        # We dont care about z axis
        achieved_goal = np.array(achieved_goal)
        desired_goal = np.array(desired_goal)

          # --- Handle single vs. batch inputs ---
        is_batch = len(achieved_goal.shape) > 1
        
        # Always work with 2D arrays for consistency
        if not is_batch:
            achieved_goal = achieved_goal.reshape(1, -1)
            desired_goal = desired_goal.reshape(1, -1)

        # current_pinch_pos = achieved_goal[:, :3]
        # current_box_xyz = achieved_goal[:, 7:10]
        # current_box_xy = achieved_goal[:, 7:9]
        # target_box_xy = desired_goal[:, 7:9]

        # distance_to_box = np.linalg.norm(current_pinch_pos - current_box_xyz, axis=1)
        # distance_to_target = np.linalg.norm(current_box_xy - target_box_xy, axis=1)
        # height_diff = np.abs(current_box_xyz[:, 2] - current_pinch_pos[:, 2])

        # reward = np.where(distance_to_target < 0.03, 3.0, 0.0)  # Reward is positive if close to target, negative otherwise
        # # reward += -distance_to_target - 0.25*distance_to_box - 0.25*height_diff
        # reward += -10*distance_to_target - 0.1*distance_to_box - 0.1*height_diff
        # # reward = np.clip(reward, -3.0, 3.0)  # Clip the reward to a reasonable range


        current_pinch_pos = achieved_goal[:, :3]
        current_box_xyz = achieved_goal[:, 7:10]
        current_box_xy = achieved_goal[:, 7:9]
        target_box_xy = desired_goal[:, 7:9]

        # distance_to_box = np.linalg.norm(current_pinch_pos - current_box_xyz, axis=1)
        # distance_to_target = np.linalg.norm(current_box_xy - target_box_xy - 0.05, axis=1)
        # height_diff = np.abs(current_box_xyz[:, 2] - current_pinch_pos[:, 2])
        # reward_for_approaching = self.last_distance_to_target - distance_to_target

        # reward = 0
        # reward += -distance_to_target - 0.05*distance_to_box - 0.05*height_diff + 10*reward_for_approaching
        # self.last_distance_to_target = distance_to_target.squeeze().copy() 
        
        distance_to_box = np.linalg.norm(current_pinch_pos - current_box_xyz, axis=1)
        action_penalty = -0.01 * np.linalg.norm(self.current_action[:-2]) 
        jerk_penalty = -0.05 * np.linalg.norm(self.current_action[:-2] - self.last_action[:-2])
         # Get the gripper's current orientation from the environment state
        current_gripper_quat = self.gripper_orient
        rot_matrix = quaternion_to_rotation_matrix(current_gripper_quat)
        # The third column of the rotation matrix is the gripper's local Z-axis in world coordinates
        current_z_axis = rot_matrix[:, 2]
        # The desired Z-axis for a "pinch down" grasp is straight down.
        target_z_axis = np.array([0., 0., -1.])
        # It ranges from +0 (perfectly aligned) to -2 (perfectly opposite).
        orientation_similarity = 1 - np.dot(current_z_axis, target_z_axis)
        orientation_scaling_factor = np.exp(-5 * distance_to_box)
        scaled_orientation_reward = -0.25 *orientation_scaling_factor * orientation_similarity

        has_reach = np.all(abs(current_pinch_pos - current_box_xyz)< 0.01) 
        success_reward = np.where(has_reach, 100.0, 0.0) 

        stuck_penalty = -0.5 if self.stuck_flag else 0.0

        reward = -distance_to_box + action_penalty + jerk_penalty + scaled_orientation_reward + success_reward
        #reward = -distance_to_box* 10 + success_reward

        # distance = np.linalg.norm(desired_goal[..., :3] - achieved_goal[..., :3])
        # # The core shaping reward: k * exp(-k * distance) 50 * np.exp(-50 * distance)
        # reward = -distance

        return reward
    
    def _is_success(self, achieved_goal: np.ndarray, desired_goal: np.ndarray) -> bool:
        if len(achieved_goal.shape) == 1:
            achieved_goal = achieved_goal.reshape(1, -1)
            desired_goal = desired_goal.reshape(1, -1)
        achieved_box_xy = achieved_goal[..., 7:9]
        desired_box_xy = desired_goal[..., 7:9]
        distance = np.linalg.norm(achieved_box_xy - desired_box_xy, axis=1)
        current_pinch_pos = achieved_goal[:, :3]
        current_box_xyz = achieved_goal[:, 7:10]
        has_reach = np.all(abs(current_pinch_pos - current_box_xyz)< 0.01) 
        return has_reach# first try 0.2, then 0.05

    def collide_gripper_ground(self):
        link_7 = self.link7_pos.copy()
        gripper_pos = self.gripper_pos.copy()
        # Value measure in reality
        return False if link_7[-1] > 0.055 and gripper_pos[-1] > 0.029 else True


    def _step(self, action):
        if DEBUG:
            log_file = f"env_{self.env_id}_step_log.txt"
            start_time = time.time()
            
            with open(log_file, "a") as f:
                f.write(f"\n=== Step {self.current_step} ===\n")
                f.write(f"Action: {action}\n")
                f.write(f"Pre-step Gripper Position: {self.gripper_pos}\n")
                f.write(f"Pre-step Box Positions: {[pose[:3] for pose in self.box_poses]}\n")
                f.write(f"Pre-step Target Box Position: {self.target_box_pos}\n")

        pre_step_gripper_pos = self.gripper_pos.copy()
        pre_step_box_pos = self.box_poses[self.target_name][:3].copy()

        # The 'action' from the agent is the target. We smooth it.
        # Note: We work with the normalized action from the agent.
        smoothed_action = self.action_smoothing_factor * action + (1 - self.action_smoothing_factor) * self.last_action

        # For HER to send to info
        box_pos_before_step = self.box_poses[self.target_name][:3].copy()
        
        # Denormalize the *smoothed* action to get the final velocities to send
        joint_vels, gripper_vel = self.denormalize_action(smoothed_action)
        
        # Update current_action for observation space
        self.current_action = smoothed_action.copy()

        start_time = time.time()

        while (self.last_sync_timestamp != self.joint_last_sync_timestamp) or (self.frame_skip > self.frame_elapse):
        #or (time.time() - start_time < self.step_duration):
            self.send_action(joint_vels, gripper_vel)
            rclpy.spin_once(self.__node, timeout_sec=self.wait_period) 
            
            if time.time() - start_time > 120:
                #self.image_pub.publish(ResetSeed(reset=True, seed=-1))
                log_file = f"env_{self.env_id}_step_log.txt"
                observation = self.get_observation()
                reward_array = self.compute_reward(observation["achieved_goal"], observation["desired_goal"],{})
                reward = float(reward_array[0])
                with open(log_file, "a") as f:
                    f.write(f"\n\n=== Step {self.current_step} ===\n")
                    f.write(f"Pre-step Gripper Position: {self.gripper_pos}\n")
                    f.write(f"Pre-step Box Positions: {[pose[:3] for pose in self.box_poses]}\n")
                    f.write(f"TIMEOUT 120s\n")
                return observation, reward, False, True, {}

        current_box_pos = self.box_poses[self.target_name].copy()[:2]
        # self.box_velocity =  (current_box_pos - box_pos_before_step[:2]) / 2
        self.box_velocity = np.clip(self.box_velocity, -1., 1.)

        self.joint_state_updated = False
        self.last_sync_timestamp += 1
       
        # Get observation and compute reward
        observation = self.get_observation()

        info = {}
        # if np.linalg.norm(current_box_pos - box_pos_before_step[:2]) > 0.001: # Use a small tolerance
        #     box_moved_this_step = True
        #     self.last_actionable_step = self.current_step
        # else:
        #     box_moved_this_step = False

        # info = {"box_moved_this_step": box_moved_this_step}

        reward_array = self.compute_reward(observation["achieved_goal"], observation["desired_goal"],info)
        reward = float(reward_array[0])
        self.success_mask = self._is_success(observation["achieved_goal"], observation["desired_goal"])

        # Update the last_action for the next step
        self.last_action = smoothed_action.copy()

        terminated = bool(self.success_mask)  # Convert to boolean
        truncated = False

        # TODO: Need to investigate why in the box does not reset properly and we still obtain the last box value of the last episode
        # if terminated:
        #     #self.image_pub.publish(ResetSeed(reset=True, seed=self.current_step))
        #     if self.current_step == 0:
        #         return self.get_observation(), -3, False, True, {}
        
        # if self.collide_gripper_ground():
        #     truncated = True
        #     reward = float(-200.0)
     
        if DEBUG:
            log_file = f"env_{self.env_id}_step_log.txt"
            self.image_pub.publish(ResetSeed(reset=True, seed=self.current_step))
            with open(log_file, "a") as f:
                f.write(f"\n{'='*10} Step {self.current_step} {'='*10}\n")
                
                # --- Pre-Step State ---
                f.write("\n--- Pre-Step State ---\n")
                f.write(f"  Gripper Pos: {np.round(pre_step_gripper_pos, 4)}\n")
                f.write(f"  Box Pos:     {np.round(pre_step_box_pos, 4)}\n")
                f.write(f"  Target Pos:  {np.round(self.target_box_pos, 4)}\n")

                # --- Action & Post-Step State ---
                f.write("\n--- Action & Post-Step State ---\n")
                f.write(f"  Smoothed Action: {np.round(self.current_action, 4)}\n")
                f.write(f"  Final Gripper Pos: {np.round(self.gripper_pos, 4)}\n")
                f.write(f"  Final Box Pos:     {np.round(self.box_poses[self.target_name][:3], 4)}\n")

                # --- Push-Specific Debugging (Re-calculating values for clarity) ---
                f.write("\n--- Push-Specific Debug ---\n")
                
                # 1. Re-gather source data (from the *end* of the step)
                gripper_pos = self.gripper_pos.copy()
                gripper_orient = self.gripper_orient.copy()
                box_pose = self.box_poses[self.target_name].copy()
                
                # 2. Re-calculate intermediate values needed for push features
                gripper_rot_matrix = quaternion_to_rotation_matrix(gripper_orient)
                current_pinch_pose = self.compute_gripper_pinch_pos(gripper_pos, gripper_rot_matrix, gripper_orient)
                vec_gripper_to_box = box_pose[:3] - current_pinch_pose[:3]
                vec_box_to_target = self.target_box_pos - box_pose[:3]

                # 3. Re-calculate the final push features
                norm_g2b = np.linalg.norm(vec_gripper_to_box)
                norm_b2t = np.linalg.norm(vec_box_to_target)
                if norm_g2b > 1e-6 and norm_b2t > 1e-6:
                    push_alignment = np.dot(vec_gripper_to_box / norm_g2b, vec_box_to_target / norm_b2t)
                else:
                    push_alignment = 0.0

                # 4. Log the detailed breakdown
                f.write(f"  Calculated Features (from observation):\n")
                f.write(f"    - push_alignment:           {np.round(push_alignment, 4)} (Goal: +1.0)\n")
                f.write(f"    - box_velocity:             {np.round(self.box_velocity, 4)}\n")
                
                # --- Outcome ---
                box_dist_to_target = np.linalg.norm(self.box_poses[self.target_name][:2] - self.target_box_pos[:2])
                f.write("\n--- Outcome ---\n")
                f.write(f"  Box-Target Distance: {np.round(box_dist_to_target, 4)} (Success if < 0.05)\n")
                f.write(f"  Reward: {reward:.4f}\n")
                f.write(f"  Terminated: {terminated} (Success: {self.success_mask})\n")
                f.write(f"  Truncated:  {truncated}\n")
                f.write(f"  Info: {info}\n")
                f.write(f"{'='*30}\n")



        if self.current_step >= self.max_steps:
            truncated = True

        # if truncated or terminated:
        #      info = {
        #         "box_moved_this_step": box_moved_this_step,
        #         "last_actionable_step": self.last_actionable_step, # This will be False until the first push, then True
        #     }

        return observation, reward, terminated, truncated, info

    def gripper_pose_callback(self, msg):
        self.gripper_pos = np.array([msg.position.x, msg.position.y, msg.position.z])
        self.gripper_orient = np.array([msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w])
        #self.__node.get_logger().info("Gripper")

    def pos_link7_callback(self, msg):
        self.link7_pos = np.array([msg.x, msg.y, msg.z])
    
    def gripper_joint_states_callback(self, msg):
        desired_joint_names = ["joint1", "joint2", "joint3"]
        ordered_positions = [0.0] * len(desired_joint_names)
        ordered_velocities = [0.0] * len(desired_joint_names)

        name_to_index = {name: index for index, name in enumerate(msg.name)}

        for i, desired_name in enumerate(desired_joint_names):
            if desired_name in name_to_index:
                index_in_msg = name_to_index[desired_name]
                ordered_positions[i] = msg.position[index_in_msg]
                ordered_velocities[i] = msg.velocity[index_in_msg]

        self.gripper_joint_positions = np.array(np.round(ordered_positions,decimals=3))
        self.gripper_joint_velocities = np.array(np.round(ordered_velocities,decimals=3))
        #self.__node.get_logger().info("Gripper Joint state")
     
    def custom_joint_states_callback(self, msg):
        if msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9 > self.joint_time:
            self.joint_state_updated = True
        self.joint_time = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9

        desired_joint_names = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]
        ordered_positions = [0.0] * len(desired_joint_names)
        ordered_velocities = [0.0] * len(desired_joint_names)

        name_to_index = {name: index for index, name in enumerate(msg.name)}

        for i, desired_name in enumerate(desired_joint_names):
            if desired_name in name_to_index:
                index_in_msg = name_to_index[desired_name]
                
                if desired_name in ['joint1', 'joint4', 'joint5','joint6']:  # Joints that can rotate continuously
                    ordered_positions[i] = ((msg.position[index_in_msg] + np.pi) % (2 * np.pi)) - np.pi
                else:
                    ordered_positions[i] = msg.position[index_in_msg]
                ordered_velocities[i] = msg.velocity[index_in_msg]

        self.joint_positions = np.array(ordered_positions)
        self.joint_velocities = np.array(ordered_velocities)
        self.joint_last_sync_timestamp = msg.effort[0]
        self.frame_elapse = msg.effort[1]
        self.stuck_flag = msg.effort[2] > 0.5
        #self.__node.get_logger().info(f"Custom joint state {self.last_sync_timestamp}")

    def create_box_pose_callback(self, box_id, msg):
        pos = np.array([msg.position.x, msg.position.y, msg.position.z])
        orient = np.array([msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w])
        self.box_poses[box_id] = np.concatenate([pos, orient])
        #self.__node.get_logger().info(f"pose: {self.box_poses[box_id][:3]}")  # For debugging

    def target_pos_callback(self, msg):
        self.target_box_pos = np.array([msg.x, msg.y, msg.z])
        
    def reset_arm_done_callback(self,msg):
        self.reset_arm_done = msg.data
    
    def reset_box_ack_callback(self,msg):
        self.reset_box_done = msg.data
  
    def close(self):
        self.executor.shutdown()
        self.executor_thread.join()
        rclpy.shutdown()

if __name__ == '__main__':    
    from stable_baselines3 import TD3, SAC
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import DummyVecEnv

    code_dir = os.path.dirname(os.path.abspath(__file__))
    #best_model_path = os.path.join(code_dir, "sac_stack_best", "sac_reach_800000_steps")
    #best_model_path = os.path.join(code_dir, "sac_stack_final.zip")
    #best_model_path = os.path.join(code_dir, "sac_stack_best", "best_model (Copy 4).zip")
    #best_model_path = os.path.join(code_dir, "sac_stack_best", "best_model_1.zip")
    #best_model_path = os.path.join(code_dir, "eval", "SAC_reaching_1_trunc","SAC_reaching_3000000_steps.zip")
    #best_model_path = os.path.join(code_dir, "sac_stack_best","sac_reach_4200000_stepsgdown.zip")
    best_model_path = os.path.join(code_dir, "sac_stack_best","SAC_reaching_9540000_steps.zip")

    env = SimulationEnv(port='1244', env_id=10)
 
    env = Monitor(env)
    env = DummyVecEnv([lambda:env])

    model = SAC.load(best_model_path, env=env, device= "cpu")

    num_episodes = 100
    
    obs_vec = env.reset()
    for episode in range(num_episodes):
        done = False
        episode_reward = 0
        step_count = 0

        while not done:#and step_count < max_steps_per_episode :
            # Get action from the model
            action, _states = model.predict(obs_vec, deterministic=True) # Use observation part
            print(f"Action: {action.round(decimals=3)}")
            # Step the environment
            obs_vec, reward, terminated, info  = env.step(action)
            episode_reward += reward

            obs_single = {key: val[0] for key, val in obs_vec.items()}
            # with open(log_file, "a") as f:
            #     f.write(f"Step: {step_count}\n")            
            #     f.write(f"Action: {action[0][:-1].round(decimals=3)}\n")
            #     f.write(f"Achieved Goal: {obs_single['achieved_goal'].round(decimals=3)}\n")
            #     f.write(f"Desired Goal: {obs_single['desired_goal'].round(decimals=3)}\n")
            #     f.write(f"Box: {obs_single['desired_goal'][:3].round(decimals=3)}\n")
            #     f.write(f"reward: {reward[0].round(decimals=3)}\n")
            #     if terminated[0]:
            #         f.write(f"terminated: {terminated[0]}\n\n")

            truncated = info[0].get("TimeLimit.truncated", False)
            done = terminated[0] or truncated

            step_count+=1

    

