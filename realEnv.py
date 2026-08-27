import rclpy
from geometry_msgs.msg import Pose, TransformStamped, Point, Quaternion
from sensor_msgs.msg import JointState
from std_msgs.msg import Int32, String, Bool, Float64MultiArray, ColorRGBA
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
import threading
from baseEnv import BaseEnv
import numpy as np
from ros2_message_interfaces.msg import ResetSeed 
from utils import *
import os
import time
from tf2_ros import TransformListener, Buffer, TransformException
from tf2_ros.static_transform_broadcaster import StaticTransformBroadcaster 
from visualization_msgs.msg import Marker 
from collections import deque
import math


DEBUG = False

class RealEnv(BaseEnv):
    def __init__(self, seed=0, port='1234',env_id=0, use_webot=True):
        super().__init__(seed=seed, port=port, env_id=env_id)

        rclpy.init()
        self.node = rclpy.create_node('real_env', namespace=f'igus_rebel_{self.port}')
        
        # Repeatability
        self.seed = 0
        set_seed(self.seed)

        self.use_webot = use_webot
        self.use_webot_observation = False # Use webot observation or tf observation
        self.send_action_real = True
        self.action_smoothing_factor = 0.7
        self.last_action = np.zeros(7)

        if self.use_webot:
            self.custom_joint_time = 0
            self.custom_joint_positions = np.zeros(6)
            self.custom_joint_velocities = np.zeros(6)
            self.custom_joint_last_sync_timestamp = -1
        
        # For keep the joint state not corrupted by access in the get observation
        self.joint_state_lock = threading.Lock()
        self.joint_position_window = deque(maxlen=10)

        self.reset_joint_positions = np.ones(6)*(-0.5)
        self.send_once = True
        self.step_duration = 0.032   # Corresponds to a 10 Hz control frequency for the agent
        self.frame_skip = np.array(3)
        self.frame_elapse = np.array(-1)

        # Initialize a list to store trajectory points
        self.tool_trajectory_points = []
        self.pinch_trajectory_points = []

        self.create_publisher()
        self.create_subscription()

        log_file = f"env_{self.env_id}_step_log.txt"
        if os.path.exists(log_file):
            os.remove(log_file)
            #pass
            
        self.executor = rclpy.executors.MultiThreadedExecutor()
        self.executor.add_node(self.node)
        self.executor_thread = threading.Thread(target=self.executor.spin, daemon=True)
        self.executor_thread.start()

    
    def close(self):
        self.executor.shutdown()
        self.node.destroy_node()

    def get_tool_pose_from_tf(self):
        """
        Computes the tool position and orientation using the TF tree.
        
        Returns:
            (np.ndarray, np.ndarray) or (None, None): A tuple of (position, orientation_quaternion)
                                                      or (None, None) if the transform is not available.
        """
        # Define the frames we want to find the transform between
        from_frame = 'table'  # The reference frame (our global origin)
        to_frame = 'tool0'    # The frame of the tool we want to find

        try:
            # Look up the transform at the latest available time
            t = self.tf_buffer.lookup_transform(
                from_frame,
                to_frame,
                rclpy.time.Time()) # or self.node.get_clock().now()

            pos = t.transform.translation
            orient = t.transform.rotation

            position_np = np.array([pos.x, pos.y, pos.z])
            orientation_np = np.array([orient.x, orient.y, orient.z, orient.w])
            
            return position_np, orientation_np

        except TransformException as ex:
            self.node.get_logger().warn(
                f'Could not transform {from_frame} to {to_frame}: {ex}',
                throttle_duration_sec=1.0 # Avoid spamming the log
            )
            return None, None
        
    def get_link7_pose_from_tf(self):
        """
        Computes the tool position and orientation using the TF tree.
        
        Returns:
            (np.ndarray, np.ndarray) or (None, None): A tuple of (position, orientation_quaternion)
                                                      or (None, None) if the transform is not available.
        """
        # Define the frames we want to find the transform between
        from_frame = 'table'  # The reference frame (our global origin)
        to_frame = 'link_7'    # The frame of the tool we want to find

        try:
            # Look up the transform at the latest available time
            t = self.tf_buffer.lookup_transform(
                from_frame,
                to_frame,
                rclpy.time.Time()) # or self.node.get_clock().now()

            pos = t.transform.translation
            orient = t.transform.rotation

            position_np = np.array([pos.x, pos.y, pos.z])
            orientation_np = np.array([orient.x, orient.y, orient.z, orient.w])
            
            return position_np, orientation_np

        except TransformException as ex:
            self.node.get_logger().warn(
                f'Could not transform {from_frame} to {to_frame}: {ex}',
                throttle_duration_sec=1.0 # Avoid spamming the log
            )
            return None, None

    def create_publisher(self):
        if self.use_webot:
            self.reset_pub = self.node.create_publisher(ResetSeed, 'reset_arm', 10)
            self.target_box_pub = self.node.create_publisher(String, 'target_box', 10)
            self.arm_pub = self.node.create_publisher(JointTrajectory, 'arm_controller/command', 10)
            self.gripper_pub = self.node.create_publisher(JointTrajectory, 'gripper_controller/command', 10)
            self.box_pub = self.node.create_publisher(Pose, 'box_pos_viz', 10)
        self.velocity_pub = self.node.create_publisher(Float64MultiArray, 'velocity_controllers/commands', 10)

        # PUBLISH THE ROBOT'S STATIC POSE IN THE WORLD
        self.tf_static_broadcaster = StaticTransformBroadcaster(self.node)
        # Create a TransformStamped message
        t = TransformStamped()
        t.header.stamp = self.node.get_clock().now().to_msg()
        t.header.frame_id = 'table'  # The parent frame
        t.child_frame_id = 'world' # The child frame (your robot's base)
        # The robot is on a box 0.04m high
        t.transform.translation.x = 0.0
        t.transform.translation.y = 0.0
        t.transform.translation.z = 0.04
        # No rotation, so use a neutral quaternion
        t.transform.rotation.x = 0.0
        t.transform.rotation.y = 0.0
        t.transform.rotation.z = 0.0
        t.transform.rotation.w = 1.0
        # Send the transform
        self.tf_static_broadcaster.sendTransform(t)

        # Add a publisher for the pinch point marker
        self.pinch_marker_pub = self.node.create_publisher(Marker, 'pinch_point_marker', 10)


    def create_subscription(self):
        if self.use_webot:
            self.node.create_subscription(JointState, 'custom_joint_states', self.custom_joint_states_callback, 1)
            self.node.create_subscription(Point, 'pos/link7', self.pos_link7_callback, 1)
            for i in range(self.box_count):
                self.node.create_subscription(
                    Pose,
                    f'box_{i}/pose',
                    lambda msg: self.create_box_pose_callback(i,msg),
                    1
                )
            self.node.create_subscription(Bool, 'reset_arm_done', self.reset_arm_done_callback, 1)
            self.node.create_subscription(JointState, 'gripper/joint_states', self.gripper_joint_states_callback, 1)
            self.node.create_subscription(Pose, 'pose/gripper', self.gripper_pose_callback, 1)
  
        self.node.create_subscription(JointState, 'joint_states', self.joint_states_callback, 1)
        
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self.node)

    def publish_trajectory_marker(self, point_list, namespace, marker_id, color, line_width=0.005):
        """
        Publishes a LINE_STRIP marker in RViz to visualize a given trajectory.

        Args:
            point_list (list): The list of geometry_msgs/Point objects for the trajectory.
            namespace (str): The namespace for the marker (e.g., "tool_trajectory").
            marker_id (int): The unique ID for this marker within its namespace.
            color (ColorRGBA): The color for the line.
            line_width (float): The width of the line in meters.
        """
        if not point_list: # Don't publish if the list is empty
            return

        marker = Marker()
        marker.header.frame_id = "table"
        marker.header.stamp = self.node.get_clock().now().to_msg()
        marker.ns = namespace
        marker.id = marker_id
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD
        
        # Pose is identity since points are in the world frame
        marker.pose.orientation.w = 1.0
        
        # Set the line's properties
        marker.scale.x = line_width
        marker.color = color
        
        # Set lifetime (0 means it persists until overwritten)
        marker.lifetime = rclpy.time.Duration(seconds=0).to_msg()
        
        # Assign the list of points
        marker.points = [Point(x=p[0], y=p[1], z=p[2]) for p in point_list]
        
        # Publish the marker
        self.pinch_marker_pub.publish(marker) # We can reuse the same publisher topic

    def publish_single_point_marker(self, point_coords, namespace, marker_id, color, marker_size=0.03):
        """
        Publishes a single SPHERE marker in RViz to visualize a specific point.

        Args:
            point_coords (tuple or list): A tuple or list of (x, y, z) coordinates for the point.
            namespace (str): The namespace for the marker (e.g., "target_point").
            marker_id (int): The unique ID for this marker within its namespace.
            color (ColorRGBA): The color for the sphere.
            marker_size (float): The diameter of the sphere in meters.
        """
        if point_coords is None or len(point_coords) != 3:
            self.node.get_logger().warn("Invalid point received. Not publishing marker.")
            return

        marker = Marker()
        marker.header.frame_id = "table"  # Or your desired frame
        marker.header.stamp = self.node.get_clock().now().to_msg()
        marker.ns = namespace
        marker.id = marker_id
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD

        # Set the pose of the marker to the point's location
        marker.pose.position.x = float(point_coords[0])
        marker.pose.position.y = float(point_coords[1])
        marker.pose.position.z = float(point_coords[2])

        # marker.pose.orientation.x = float(point_coords[3])
        # marker.pose.orientation.y = float(point_coords[4])
        # marker.pose.orientation.z = float(point_coords[5])
        marker.pose.orientation.w = 1.0

        # Set the scale of the marker (diameter of the sphere)
        marker.scale.x = marker_size
        marker.scale.y = marker_size
        marker.scale.z = marker_size

        # Set the color
        marker.color = color

        # Set lifetime (0 means it persists until deleted or overwritten)
        marker.lifetime = rclpy.time.Duration(seconds=0).to_msg()

        # The 'points' field is not used for SPHERE type, so we leave it empty.

        # Publish the marker
        self.pinch_marker_pub.publish(marker)

    def create_box_pose_callback(self, box_id, msg):
        pos = np.array([msg.position.x, msg.position.y, msg.position.z])
        orient = np.array([msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w])
        self.box_poses[box_id] = np.concatenate([pos, orient])

    def joint_states_callback(self, msg):
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
                
                if desired_name in ['joint1','joint4','joint6']:  # Joints that can rotate continuously
                    ordered_positions[i] = ((msg.position[index_in_msg] + np.pi) % (2 * np.pi)) - np.pi
                else:
                    ordered_positions[i] = msg.position[index_in_msg]
                ordered_velocities[i] = msg.velocity[index_in_msg]

        # Calculate the mean of all measurements in the window.
        # `axis=0` calculates the mean for each joint (column) independently.
        #self.joint_position_window.append(np.array(ordered_positions))
        #filter_joint_positions = np.mean(self.joint_position_window)

        with self.joint_state_lock:
            # Update the joint positions and velocities
            self.joint_positions = np.array(ordered_positions)
            self.joint_velocities = np.array(ordered_velocities)

    def custom_joint_states_callback(self, msg):
        if msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9 > self.joint_time:
            self.joint_state_updated = True
        self.custom_joint_time = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9

        desired_joint_names = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]
        ordered_positions = [0.0] * len(desired_joint_names)
        ordered_velocities = [0.0] * len(desired_joint_names)

        name_to_index = {name: index for index, name in enumerate(msg.name)}

        for i, desired_name in enumerate(desired_joint_names):
            if desired_name in name_to_index:
                index_in_msg = name_to_index[desired_name]
                
                if desired_name in ['joint1','joint4','joint6']:  # Joints that can rotate continuously
                    ordered_positions[i] = ((msg.position[index_in_msg] + np.pi) % (2 * np.pi)) - np.pi
                else:
                    ordered_positions[i] = msg.position[index_in_msg]
                ordered_velocities[i] = msg.velocity[index_in_msg]

        self.custom_joint_positions = np.array(np.round(ordered_positions, decimals=3))
        self.custom_joint_velocities = np.array(np.round(ordered_velocities, decimals=3))
        self.custom_joint_last_sync_timestamp = msg.effort[0]
        self.frame_elapse = msg.effort[1]

    def gripper_pose_callback(self, msg):
        self.gripper_pos = np.round(np.array([msg.position.x, msg.position.y, msg.position.z]),decimals=3)
        self.gripper_orient = np.round(np.array([msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w]),decimals=3)
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

    def reset_arm_done_callback(self,msg):
        self.reset_arm_done = msg.data

    def reset_arm(self):
        # Get thread-safe copies of the state
        target = self.custom_joint_positions.copy() if self.use_webot else self.reset_joint_positions
        current = self.joint_positions.copy()

        # --- Proportional Control Calculation ---
        error = target - current

        if np.all(np.abs(error) < 0.01):
                # Send zero velocity to stop
                self.send_action(np.zeros(6), np.zeros(1))
                self.send_once = True
                return True

        # Calculate velocity: velocity = Kp * error
        velocity_command = error

        # Apply velocity limits (saturation)
        velocity_command = np.clip(velocity_command, -self.max_joint_vels, self.max_joint_vels)

        #print(velocity_command)

        # --- Publish Command ---
        cmd_msg = Float64MultiArray()
        cmd_msg.data = velocity_command.tolist()
        self.velocity_pub.publish(cmd_msg)
        return False
    
    def move_arm(self,target):
        target = target.copy()
        # Get thread-safe copies of the state
        current = self.joint_positions.copy()

        # --- Proportional Control Calculation ---
        error = target - current

        if np.all(np.abs(error) < 0.01):
                # Send zero velocity to stop
                self.send_action(np.zeros(6), np.zeros(1))
                self.send_once = True
                return True

        # Calculate velocity: velocity = Kp * error
        velocity_command = error

        # Apply velocity limits (saturation)
        velocity_command = np.clip(velocity_command, -self.max_joint_vels, self.max_joint_vels)

        #print(velocity_command)

        # --- Publish Command ---
        self._step(np.concatenate([velocity_command,np.zeros(1)], axis=-1))
        return False
    
    def reset(self, seed=None):
        print(f"Reset {self.current_step} env {self.env_id}")
        
        # Clear the trajectory points for the new episode
        self.tool_trajectory_points = []
        self.pinch_trajectory_points = []

        #self.seed = np.random.randint(0,2**31)
        #self.seed = 0 if self.seed >= 2**31 else self.seed + 1
        
        self.current_step = 0
        self.set_target_theta = False

        self.reset_joint_positions = np.ones(6)
        self.reset_joint_positions[1]=-0.5

        # Random box generation set to true
        self.set_box_pose = True

        if self.use_webot:
            # Publish reset message with integer seed
            reset_msg = ResetSeed()
            reset_msg.reset = True

            reset_msg.seed = self.seed
               
            ## wait for reset to finish
            print("start")
            self.reset_arm_done = False
            start_time = time.time()
            while not self.reset_arm_done:
                if (time.time() - start_time) >= 0.0009: 
                    self.reset_pub.publish(reset_msg)
                    start_time = time.time()
                rclpy.spin_once(self.node, timeout_sec=0)
            self.reset_arm_done = False
            print("end")

            self.last_sync_timestamp = 0
            self.joint_state_updated = False
        print("start real")
        has_real_arm_reset = False
        while not has_real_arm_reset:
            rclpy.spin_once(self.node, timeout_sec=0)
            has_real_arm_reset = self.reset_arm()
        print("end real")
        observation = self.get_observation()

        if DEBUG:
            log_file = f"env_{self.env_id}_step_log.txt"
            with open(log_file, "a") as f:
                f.write(f"Reset Duration {time.time() - start_time}")

        return observation, {}

    def send_action(self, joint_vels, gripper_vel):
        ### ENABLE GRAVITY FOR BOXES
        if self.use_webot:
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
        
        # Publish arm velocities
        if self.send_once and self.send_action_real:
            arm_msg = Float64MultiArray()
            arm_msg.data = joint_vels.tolist()
            #print((f"Step {self.current_step}: {np.round(arm_msg.data, 3)}"))
            self.velocity_pub.publish(arm_msg)
            self.send_once = False

       
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

    # # Different to simulation now in z axis, depend on the tool0 orientation, in simulation is has different orientation than in real.
    # def compute_gripper_pinch_pos(self,pos,rot_matrix,orient):
    #     ##This function computes roughly the pinch position of the fingers (i.e. the position the fingers would meet when closing)
    #     ##This is based on the position and orientation of the gripper (see webots documentation) and gripper specificatio(n
    #     orientation = np.reshape(rot_matrix, (3,3))
    #     offset = np.array([0.07,0,0.0]) #pinch position is roughly 10cm from gripper pos in z-direction in gripper coordinate system
    #     pos_offset = np.matmul(orientation, offset) + pos
    #     return np.concatenate([pos_offset, orient])
        
    def get_observation(self):
        if self.use_webot_observation:
            gripper_pos = self.gripper_pos.copy()
            gripper_orient = self.gripper_orient.copy() # Assuming w, x, y, z
            link7_pos = self.link7_pos.copy()
        else:
            gripper_pos, gripper_orient = self.get_tool_pose_from_tf()
            link7_pos,_ = self.get_link7_pose_from_tf()

        # Compute pinch position
        gripper_rot_matrix = quaternion_to_rotation_matrix(gripper_orient)
        current_pinch_pose = self.compute_gripper_pinch_pos(gripper_pos, gripper_rot_matrix, gripper_orient)

        # # Append the new points to their respective lists
        # self.tool_trajectory_points.append(gripper_pos)
        self.pinch_trajectory_points.append(current_pinch_pose[:3])
        
        # --- PUBLISH BOTH TRAJECTORIES ---
        # Publish the tool (gripper_pos) trajectory as a blue line
        # self.publish_trajectory_marker(
        #     point_list=self.tool_trajectory_points,
        #     namespace="tool_trajectory",
        #     marker_id=0,
        #     color=ColorRGBA(r=0.0, g=0.0, b=1.0, a=1.0) # Blue
        # )
        
        # Publish the pinch point trajectory as a green line
        self.publish_trajectory_marker(
            point_list=self.pinch_trajectory_points,
            namespace="pinch_trajectory",
            marker_id=1, # Use a different ID to be safe, though namespace is enough
            color=ColorRGBA(r=0.0, g=1.0, b=0.0, a=1.0) # Green
        )

        # Box state
        box_pose = self.random_box_pose() if self.set_box_pose else self.current_box_pose
        self.box_pub.publish(Pose(position=Point(x=box_pose[0], y=box_pose[1], z=box_pose[2]), orientation=Quaternion(x=box_pose[3], y=box_pose[4], z=box_pose[5], w=box_pose[6])))

        # box_pose = self.box_poses[self.target_name].copy() # Box center position
        # box_pose[2] += 0.05
        # --- PUBLISH BOTH TRAJECTORIES ---
        # Publish the tool (gripper_pos) trajectory as a blue line
        self.publish_single_point_marker(
            point_coords=box_pose[:3],
            namespace="box",
            marker_id=0,
            color=ColorRGBA(r=1.0, g=1.0, b=0.0, a=1.0), # Yellow
            marker_size=0.1
        )
        self.publish_trajectory_marker(
            point_list=[current_pinch_pose[:3], box_pose[:3]],  # The list contains the start and end of the line
            namespace="distance_line",
            marker_id=10,  # Use a new, unique ID for this marker
            color=ColorRGBA(r=1.0, g=1.0, b=1.0, a=0.8)  # White and slightly transparent
        )

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

        if len(achieved_goal.shape) == 1:
            achieved_goal = achieved_goal.reshape(1, -1)
            desired_goal = desired_goal.reshape(1, -1)

        # # if self.stuck_flag:
        # #     reward -= 2
        
        # action_penalty = -0.01 * np.linalg.norm(self.current_action[:-2]) 
        # jerk_penalty = -0.05 * np.linalg.norm(self.current_action[:-2] - self.last_action[:-2])
        #  # Get the gripper's current orientation from the environment state
        # current_gripper_quat = self.gripper_orient
        # rot_matrix = quaternion_to_rotation_matrix(current_gripper_quat)
        # # The third column of the rotation matrix is the gripper's local Z-axis in world coordinates
        # current_z_axis = rot_matrix[:, 2]
        # # The desired Z-axis for a "pinch down" grasp is straight down.
        # target_z_axis = np.array([0., 0., -1.])
        # # It ranges from +0 (perfectly aligned) to -2 (perfectly opposite).
        # orientation_similarity = 1 - np.dot(current_z_axis, target_z_axis)
        # orientation_scaling_factor = np.exp(-5 * distance)
        # scaled_orientation_reward = -0.25 *orientation_scaling_factor * orientation_similarity

        # #reward += action_penalty + jerk_penalty + scaled_orientation_reward
        # if has_reach:
        #     reward += 50*2

        distance = np.linalg.norm(desired_goal[..., :3] - achieved_goal[..., :3])
        # The core shaping reward: k * exp(-k * distance) 50 * np.exp(-50 * distance)
        reward = -distance

        reward = np.squeeze(reward)
            
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
        """
        Check if the gripper collides with the ground.
        """
        link_7_pos = self.link7_pos.copy()
        gripper_pos = self.gripper_pos.copy()
        # Value measure in reality
        return False if link_7_pos[-1] > 0.055 and gripper_pos[-1] > 0.029 else True

    def _step(self, action):
        smoothed_action = self.action_smoothing_factor * action + (1 - self.action_smoothing_factor) * self.last_action
        
        # Denormalize the *smoothed* action to get the final velocities to send
        joint_vels, gripper_vel = self.denormalize_action(smoothed_action)
        
        # Update current_action for observation space (optional but good practice)
        self.current_action = smoothed_action.copy()
    
        start_time = time.time()
        
        while self.last_sync_timestamp != self.custom_joint_last_sync_timestamp or (time.time() - start_time < self.step_duration): #(self.frame_skip > self.frame_elapse):
            self.send_action(joint_vels, gripper_vel)
            rclpy.spin_once(self.node, timeout_sec=0) 
        self.send_once = True
        
        self.joint_state_updated = False
        self.last_sync_timestamp += 1
        
        # Get observation and compute reward
        observation = self.get_observation()

        reward = self.compute_reward(observation["achieved_goal"], observation["desired_goal"],{})

        # Update the last_action for the next step
        self.last_action = smoothed_action.copy()

        terminated = bool(self._is_success(observation["achieved_goal"], observation["desired_goal"]))  # Convert to boolean
        truncated = False

        if self.collide_gripper_ground():
            truncated = True
            reward = max(-10*self.max_steps,-100)

        if self.current_step >= self.max_steps:
            truncated = True


        return observation, reward, terminated, truncated, {}

if __name__ == '__main__':
    from stable_baselines3 import TD3, SAC
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import DummyVecEnv

    code_dir = os.path.dirname(os.path.abspath(__file__))
    #best_model_path = os.path.join(code_dir, "sac_stack_best", "stuck", "t3d_reach_2400000_steps.zip")
    #best_model_path = os.path.join(code_dir, "sac_stack_best", "best_model (Copy 4).zip")
    #best_model_path = os.path.join(code_dir, "sac_reach_best", "best_model.zip")
    #best_model_path = os.path.join(code_dir, "sac_stack_best", "stuck", "best_model (Copy).zip")
    # best_model_path = os.path.join(code_dir, "sac_stack_best", "sac_reach_5280000_steps.zip")
    #best_model_path = os.path.join(code_dir, "sac_stack_best", "sac_reach_4200000_stepsgdown.zip")
    best_model_path = os.path.join(code_dir, "sac_stack_best","best_model_sac_push.zip")
    log = "log.txt"
    log_file = "real_log_state57.txt"
    start_time = time.time()
    env = RealEnv(port='1244', env_id=10,use_webot=True)
    env.use_webot_observation = False
    env.send_action_real = True
    #env.max_joint_vels=0.5
    env.max_steps = 100
    env.action_smoothing_factor=0.7
    env.step_duration = 0.032*1  # 0.096 0.2 0.05 0.36
    
    env = Monitor(env)
    env = DummyVecEnv([lambda:env])

    model = SAC.load(best_model_path, device="cpu") 

    num_episodes = 5
    
    obs_vec = env.reset()
    for episode in range(num_episodes):
        done = False
        episode_reward = 0
        step_count = 0
        with open(log_file, "a") as f:
            f.write(f"-----Episode {episode}-----\n\n")

        while not done:#and step_count < max_steps_per_episode :
            # full_observation_vector = obs_vec["observation"][0]
            # reach_obs_for_model = {
            #         #"observation": np.array([full_observation_vector[:65]]),
            #         "observation": np.array([full_observation_vector[:-12]]),
            #         "achieved_goal": np.array([obs_vec["achieved_goal"][0]]),
            #         "desired_goal": np.array([obs_vec["desired_goal"][0]])
            # }
            # Get action from the model
            action, _states = model.predict(obs_vec, deterministic=True) # Use observation part
            print(action[0][:-1].round(decimals=3))

            # Step the environment
            obs_vec, reward, terminated, info  = env.step(action)
            episode_reward += reward

            obs_single = {key: val[0] for key, val in obs_vec.items()}
            with open(log_file, "a") as f:
                f.write(f"Step: {step_count}\n")            
                f.write(f"Action: {action[0][:-1].round(decimals=3)}\n")
                f.write(f"Achieved Goal: {obs_single['achieved_goal'].round(decimals=3)}\n")
                f.write(f"Desired Goal: {obs_single['desired_goal'].round(decimals=3)}\n")
                f.write(f"Box: {obs_single['desired_goal'][:3].round(decimals=3)}\n")
                f.write(f"reward: {reward[0].round(decimals=3)}\n")
                f.write(f"terminated: {terminated[0]}\n\n")

            truncated = info[0].get("TimeLimit.truncated", False)
            done = terminated[0] or truncated

            step_count+=1
            #time.sleep(0.01)
  

    # action = np.zeros(7)
    # while True:
    #     env.send_action(action[:6],action[6])
    #     env.send_once=True

    # Test goal
    # init = True
    # while True:
    #     for i in range(1000):
    #         rclpy.spin_once(env.node, timeout_sec=0)
    #     current = env.custom_joint_positions.copy()[0]
    #     min_limit, max_limit = env.joint_limits["joint1"]
    #     # Clip position to limits before normalization to avoid >1 or <-1
    #     position_clipped = np.clip(current, min_limit, max_limit)
    #     normalized_position = 2 * (position_clipped - min_limit) / (max_limit - min_limit) - 1
    #     theta = np.arctan2(env.box_poses[env.target_name][1], env.box_poses[env.target_name][0])
    #     n_theta = -theta/np.pi
    #     t_theta = theta +1
    #     t_theta = np.arctan2(np.sin(t_theta), np.cos(t_theta))/(-np.pi)
    #     action[0] = t_theta - normalized_position
    #     print(f"Action {action[0]}")
    #     print(f"Current {normalized_position}")
    #     print(f"Theta {n_theta}")
    #     print(f"Target {t_theta}")
    #     print(f"Box {env.box_poses[env.target_name][:2]}")
    #     joint_vels, gripper_vel = env.denormalize_action(action)
    #     while env.last_sync_timestamp != env.custom_joint_last_sync_timestamp: #or not self.joint_state_updated:
    #         env.send_action(joint_vels, gripper_vel)
    #         rclpy.spin_once(env.node, timeout_sec=0) 
        
    #     env.joint_state_updated = False
    #     env.last_sync_timestamp += 1
    #     init = False

    # env.reset()
    # # #time.sleep(60)
    # position = np.zeros(6)
    # position[0] = 3.1
    # position[1] = -1.1
    # reach_goal = False
    # while not reach_goal:
    #     reach_goal = env.move_arm(position)



    
      

    

