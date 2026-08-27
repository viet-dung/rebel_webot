import gymnasium as gym
from gymnasium import spaces
from abc import ABC, abstractmethod
from utils import *
import os
import numpy as np
from ros2_message_interfaces.msg import ResetSeed
import math
import tf_transformations

class BaseEnv(gym.Env, ABC):
    def __init__(self, seed=0, port='1234', env_id=0):
        super(BaseEnv, self).__init__()
        
        self.seed = seed
        # set_seed(self.seed)

        self.action_space = None
        self.observation_space = None
        self.max_steps = 50
        self.current_step = 0
        self.step_duration = 0.032

        self.port = port
        self.env_id = env_id
        os.environ['ROS_DOMAIN_ID'] = str(self.env_id)

        # Box information
        self.box_count = 1
        self.box_poses = [np.zeros(7) for _ in range(self.box_count)]  # [pos(3) + quat(4)]
        self.box_velocity = np.zeros(2)

        # Gripper information
        self.gripper_pos = np.zeros(3)
        self.gripper_orient = np.zeros(4)
        self.last_gripper_pos = np.zeros(3)
        self.count_last_gripper_pos = 0

        self.gripper_joint_positions = np.zeros(3)
        self.gripper_joint_velocities = np.zeros(3)

        self.current_action = np.zeros(7)

        # Joint information
        self.joint_positions = np.zeros(6)  # 6 joints
        self.joint_velocities = np.zeros(6)
        self.last_sync_timestamp = 0
        self.joint_last_sync_timestamp = -1
        self.joint_time = 0
        self.joint_state_updated = False
        self.joint_limits = { # Full joint limits for normal operation
            'joint1': (-3.124139, 3.124139),
            'joint2': (-1.919862, 1.919862),
            'joint3': (-1.919862, 1.919862),
            'joint4': (-3.124139, 3.124139),
            'joint5': (-1.65806, 1.65806),
            'joint6': (-3.124139, 3.124139),
        }
        self.arm_length = 0.912 + 0.15 #offset
        # Velocities scaling
        self.max_joint_vels = 0.785398163

        self.link7_pos = np.zeros(3)

        # FRAME SKIPPING
        self.frame_skip = np.array(3)
        self.frame_elapse = np.array(-1)
        self.stuck_flag = False

        # TODO: anaylse
        self.hard_reset = False

        self.success_mask = [False]
        self.reset_arm_done = False
        self.seed_reset = 0
        self.reset_box_done = False

        self.target_name = 0
        self.grasped_boxes = [False] * self.box_count
        self.set_target_theta = False
        self.target_box_center_rot = np.zeros(1)
        self.target_box_xy = np.zeros(2)
        self.target_box_pos = np.zeros(3)
        
        self.stack_height = 0  # Track how many boxes are stacked
        self.stack_offset = 0.05  # Offset between stacked boxes
        self.box_height = 0.055  # Height of the box
        self.phase = 'grasping'  # Phase of the episode (grasping or stacking)
        
        self.success_bonus = 10.0  # Reward bonus for successful grasp
        self.pos_error_threshold = 0.3  # Position error threshold for successful grasp
        self.angle_error_threshold = 0.1
        self.safety_distance_ground = 0.03  # Safety distance from ground

        # Add error tracking for ODE errors
        self.ode_error = False

        # Generate random box pose for reaching task
        self.current_box_pose = None
        self.set_box_pose = True

        # GYM specific
        # Action and observation spaces
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(7,),
            dtype=np.float32
        )
        self.observation_space = spaces.Dict({
            "observation": spaces.Box(low=-1, high=1, shape=(66 ,), dtype=np.float64),  #57 , 69, 66
            #4
            "achieved_goal": spaces.Box(low=-1, high=1, shape=(13,), dtype=np.float64),  # [x, y, z, qx, qy, qz, qw, stack_height]  Current pinch position
            #4
            "desired_goal": spaces.Box(low=-1, high=1, shape=(13,), dtype=np.float64),   # Target pinch position 
        })

    def random_box_pose(self):
        # This function will loop until a valid position is found that is outside the forbidden zone.
        # Get the gripper's current position as the center point
        center_pos = self.gripper_pos.copy()
        
        #Decide whether to spawn near the gripper or randomly in the workspace
        if np.random.rand() < 0.75 and np.all(np.abs(center_pos) > 0.3):
            # --- STRATEGY 1: SPAWN NEAR GRIPPER ---
            distance = np.random.uniform(0.05, 0.1)
            phi = np.random.uniform(0, 2 * math.pi)
                # 2. Generate a random direction in 3D space.
            #    This is a standard method to get a uniform point on a sphere's surface.
            
            # 3. Convert the random direction into 3D offsets.
            offset_x = distance  * math.cos(phi)
            offset_y = distance  * math.sin(phi)
            
            # Calculate the potential new box position
            x = center_pos[0] + offset_x
            y = center_pos[1] + offset_y
            z = center_pos[2] - 0.1
            
            # Clip the position to stay within the overall workspace
            x = np.clip(x, -0.5, 0.5)
            y = np.clip(y, -0.5, 0.5)
            #z = np.clip(z, 0.0275, 0.2)
        else:
            forbidden_start_rad = 160 * math.pi / 180
            forbidden_end_rad = 190 * math.pi / 180
            full_circle = 2 * math.pi

            theta = np.random.uniform(0, forbidden_end_rad) if np.random.rand() < 0.5 else np.random.uniform(forbidden_start_rad, full_circle)
            radius = np.random.uniform(0.3,0.5)
            x = radius * math.cos(theta)
            y = radius * math.sin(theta)
            z = np.random.uniform(0.1, 0.3) if np.random.rand() < 0.5 else 0.1
        # --- FINALIZATION (runs only after a valid x, y is found) ---
        # Generate a random orientation
        roll = 0.0
        pitch = 0.0
        yaw = np.random.uniform(0, 2 * math.pi)

        # Convert these Euler angles (roll, pitch, yaw) into a quaternion.
        # This gives us an orientation that is always "facing up".
        upright_quat = tf_transformations.quaternion_from_euler(roll, pitch, yaw)
        
        pos = np.array([x, y, z]).flatten()
        pose = np.concatenate([pos, upright_quat])
        # Update state and return
        self.current_box_pose = pose
        self.set_box_pose = False
        return pose
    
    def random_box_target(self):
        # This function will loop until a valid position is found that is outside the forbidden zone.
        # Get the gripper's current position as the center point
        center_pos = self.current_box_pose

        distance = np.random.uniform(0.1, 0.2)
        phi = np.random.uniform(0, 2 * math.pi)
        
        offset_x = distance  * math.cos(phi)
        offset_y = distance  * math.sin(phi)
        
        x = center_pos[0] + offset_x
        y = center_pos[1] + offset_y

        x = max(x,0.2) if x>0 else min(x,-0.2)
        y = max(y,0.2) if y>0 else min(y,-0.2)

        x = np.clip(x, -0.5, 0.5)
        y = np.clip(y, -0.5, 0.5)
        self.target_box_pos = np.array([x, y, 0.0275])
        return np.array([x, y, 0.0275])
    
    # Default for simulation enviroment
    def compute_gripper_pinch_pos(self,pos,rot_matrix,orient):
        ##This function computes roughly the pinch position of the fingers (i.e. the position the fingers would meet when closing)
        ##This is based on the position and orientation of the gripper (see webots documentation) and gripper specificatio(n
        orientation = np.reshape(rot_matrix, (3,3))
        offset = np.array([0.00,0,0.07]) #pinch position is roughly 10cm from gripper pos in z-direction in gripper coordinate system
        pos_offset = np.matmul(orientation, offset) + pos
        return np.concatenate([pos_offset, orient])
    
    def denormalize_action(self, action):
        #action = np.pad(action, (0, 1), 'constant')
        joint_vels = action[:6] * self.max_joint_vels
        gripper_vel = action[6]
        return joint_vels, gripper_vel
    
    def step(self, action):
        observation, reward, terminated, truncated, info = self._step(action)
        self.current_step += 1
        return observation, reward, terminated, truncated, info

    @abstractmethod
    def _step(self, action):
        pass

    @abstractmethod
    def get_observation(self):
        pass

    @abstractmethod
    def send_action(self, action):
        pass

    @abstractmethod
    def compute_reward(self, achieved_goal, desired_goal, _info):
        pass
