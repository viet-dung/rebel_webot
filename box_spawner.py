import rclpy
import numpy as np
import math
from geometry_msgs.msg import Point, Quaternion, Pose, TransformStamped
from std_msgs.msg import Bool,Int32
from rclpy.node import Node
import tf_transformations as tr
from ros2_message_interfaces.msg import ResetSeed 
import os
import shutil
import time
from tf2_ros import TransformListener, Buffer, LookupException, ConnectivityException, ExtrapolationException
from tf2_geometry_msgs import do_transform_pose
import collections

def set_seed(random_seed):
    import random
    import numpy as np
    np.random.seed(random_seed)
    random.seed(random_seed)

class BoxSpawner():
    def init(self, webots_node, properties):
        self.seed = 0
        #set_seed(self.seed)

        self.box_count = 1
        self.capture_count = 0  # Add this line to initialize the counter
        self.object_pose = None
        self.once = True

        self.__robot = webots_node.robot
        self.__port = properties['port']
        self.__env_id = properties['env_id']
        os.environ['ROS_DOMAIN_ID'] = str(self.__env_id)
        self.timestep = int(self.__robot.getBasicTimeStep())

        self.__node = rclpy.create_node('box_spawner', namespace=f'igus_rebel_{self.__port}')
        
        # Ensure this is a Supervisor
        if not self.__robot.getSupervisor():
            raise RuntimeError("This plugin requires Supervisor privileges")
            
        self.__boxes = []
        self.previous_positions = []
        self.last_theta = 160 * math.pi / 180
        self.new_pos = np.array([1,1,0])

        # New attributes for averaging
        self.num_frames_to_average = 10
        self.pose_buffer = collections.deque(maxlen=self.num_frames_to_average)
        
        self.__reset_processed = False
        self.last_time_reset = time.time()
        root = self.__robot.getRoot()
        self.__root_children = root.getField('children')

        # Get WorldInfo and contactProperties field
        self.__world_info = self._get_world_info_node()
        if self.__world_info is None:
            self.__node.get_logger().info("Warning: WorldInfo node not found. Contact properties will not be dynamically added.")
            self.__contact_properties_field = None
        else:
            self.__contact_properties_field = self.__world_info.getField('contactProperties')
            if self.__contact_properties_field is None:
                self.__node.get_logger().info("Warning: contactProperties field not found in WorldInfo. Contact properties will not be dynamically added.")

        # --- Image Capture Setup ---
        self.__node.create_subscription(ResetSeed, 'capture_image', self._capture_callback, 1)

        self.viewpoints = [
            ([0, 0, 2.84], [-0.57735, 0.57735, 0.57735, 2.0944], "top"),
            #([0.75, -1, 1.9], [-0.452, 0.187, 0.872, 2.418], "angled")
        ]
        # -----------------------------------------------------------

        
        cwd = os.getcwd()
        relative_save_path = f"webots_captures_env_{self.__env_id}"
        self.save_path = os.path.abspath(os.path.join(cwd, "imgs", relative_save_path))
        os.makedirs(self.save_path, exist_ok=True)
        # ---------------------------
        
        # ROS2 setup
        self.__pose_publishers = [self.__node.create_publisher(Pose, f'box_{i}/pose', 10) for i in range(self.box_count)]
        self.reset_publisher = self.__node.create_publisher(Bool,'reset_box_ack',10)

        self.fd_publisher = self.__node.create_publisher(Pose, f'fd/pose', 10)


        self.__node.create_subscription(ResetSeed, 'reset_boxes', self.reset_callback, 1)
        #self.__node.create_subscription(Pose,f'/azure_kinect/object', self.getObjectCamera_callback,1)
        
        self.is_calibrating = True  # Start in calibration mode
        self.num_calibration_samples = 10  # Collect 10 samples to average
        
        # This deque will be used for both initial calibration and live refinement
        self.calibration_offsets = collections.deque(maxlen=self.num_calibration_samples)
        
        self.calibrated_offset = np.array([0.249 - 0.044, -0.009 - (-0.067), 0.0])  # Initialize to zero, will be set after calibration
        self.calibration_samples_collected = 0

        # Initialize TF2 listener
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self.__node)

        # --- TRACKING FILTER & LIVE CALIBRATION VARIABLES ---
        self.STABILITY_THRESHOLD = 0.01  # Max std dev (meters) for an object to be "stable"
        self.MOVE_THRESHOLD = 0.01        # Deadband filter for tracking
        self.MAX_SKIPPED_FRAMES = 10      # Force update after N skipped frames
        self.last_sent_position = None    # Last position sent to the simulator
        self.skipped_frames_count = 0

        # Create 10 boxes
        # for i in range(self.box_count):
        #     self._create_box(i,add_physics=False)
    
    def _get_world_info_node(self):
        """Helper function to get the WorldInfo node."""
        for i in range(self.__root_children.getCount()):
            node = self.__root_children.getMFNode(i)
            if node.getTypeName() == "WorldInfo":
                self.__node.get_logger().info("WorldInfo node found!")
                return node
        return None
    
    
    #geometry Box {{ size 0.055 0.055 0.055 }}
    def _create_box(self, index, set_contact_properties=True, avoid_position=[0,0,0], add_physics=True):
        """Helper function to create a box with a random position and orientation."""
        box_def = f'box_{index}'
        box_material_name = f'box_material_{index}' 
        pos ,rot = self.random_position_and_rotation(avoid_position=avoid_position)
        box_str = f'''
        DEF {box_def} Solid {{
            translation {pos[0]} {pos[1]} {pos[2]}
            rotation {rot[0]} {rot[1]} {rot[2]} {rot[3]}
            contactMaterial "{box_material_name}" # Unique contactMaterial for each box
            children [
                Shape {{
                    geometry Box {{ size 0.055 0.055 0.055 }}
                    appearance PBRAppearance {{
                        baseColor {abs(pos[0])} {abs(pos[1])} {abs(pos[2])}
                        roughness 1
                        metalness 0
                    }}
                }}
            ]
            name "{box_def}"
            boundingObject Box {{ size 0.055 0.055 0.055 }}
        }}
        '''
        self.__root_children.importMFNodeFromString(-1, box_str)
        box_node = self.__robot.getFromDef(box_def)
        
        if box_node is None:
            self.__node.get_logger().info(f"Error: Failed to find node with DEF '{box_def}' in the scene tree.")
        else:
            self.__node.get_logger().info(f"Successfully created box with DEF '{box_def}'.")
        
        self.__boxes.append(box_node)
        self.previous_positions.append(pos)

        if self.__contact_properties_field and set_contact_properties:
            contact_property_str = f'''
            ContactProperties {{
                material1 "{box_material_name}"
                material2 "finger_material"
                coulombFriction 100
                maxContactJoints 10
            }}
            '''
            self.__contact_properties_field.importMFNodeFromString(-1, contact_property_str)
            self.__node.get_logger().info(f"Dynamically added ContactProperties for box {index} with material '{box_material_name}'.")

        physics_field = box_node.getField('physics')  # Get the 'physics' field
        physics_node = physics_field.getSFNode()
        if not physics_node and add_physics:  # If physics is NULL
            physics_str = '''
            Physics {
                mass 0.5  # Set the mass
            }
            '''
            physics_field.importSFNodeFromString(physics_str)  # Import the physics node

    def reset_callback(self, msg):
        """Reset the boxes to new random positions and orientations."""
        self.last_time_reset = time.time()
        if not self.__reset_processed and msg.reset:
            self.__reset_processed = True
            #set_seed(msg.seed)
            self.seed = msg.seed

            self.reset_publisher.publish(Bool(data=False))

            forbidden_start_rad = 160 * math.pi / 180
            forbidden_end_rad = 190 * math.pi / 180

            for i, box in enumerate(self.__boxes):
                if i >0 and i == len(self.__boxes) - 1:
                    break
                while True:
                    # Generate theta in radians [0, 2*pi)
                    theta = np.random.uniform(0, 2 * math.pi)
                    if not (forbidden_start_rad <= theta <= forbidden_end_rad) and not self.last_theta == theta:
                        break # Valid theta found
                self.last_theta = theta
                x =  np.random.uniform(0.1, 0.5)* math.cos(theta)
                y =  np.random.uniform(0.1, 0.5)* math.sin(theta)
                z = np.random.uniform(0.0275, 0.35) if y > 0.3 else np.random.uniform(0.0675, 0.35)
                new_pos = [x,y,z]
                   
                #Set new position and orientation
                box.getField('translation').setSFVec3f(new_pos)
                #box.getField('rotation').setSFRotation(new_rot)
                self.new_pos = np.array(new_pos).copy()
            
            # for i, box in enumerate(self.__boxes):
            #     box.remove()
            # self.__boxes = []
            
            # for i in range(self.box_count):
            #     self._create_box(i,set_contact_properties=False, avoid_position=self.previous_positions[i])

            #self.__robot.simulationResetPhysics()
            #self.__robot.step(self.timestep)
                

    def random_position_and_rotation(self, avoid_position=[0,0,0], max_attempts=1):
        """
        Generates a random 3D position and rotation (axis-angle representation) within the workspace of the robotic arm,
        ensuring that the box does not collide with the arm.

        Args:
            safe_distance (float): Minimum distance to maintain between the box and the arm's base or links.
            max_attempts (int): Maximum number of attempts to generate a valid position.

        Returns:
            position (list): A list of 3 floats representing [x, y, z] position.
            rotation (list): A list of 4 floats representing [axis_x, axis_y, axis_z, angle] rotation.
        """

        # Define the workspace bounds
        workspace_bounds = {
            'x': (0, 0.3),  # X-axis bounds (left/right)
            'y': (-0.65, -0.72),  # Y-axis bounds (forward/backward)
            'z': (0.0275, 0.35)  # Z-axis bounds (up/down)
        }
        #  workspace_bounds = {
        #     'x': (-0.5, 0.5),  # X-axis bounds (left/right)
        #     'y': (-0.5, 0.5),  # Y-axis bounds (forward/backward)
        #     'z': (0.0275, 0.35)  # Z-axis bounds (up/down)
        # }
   
        # Generate random position within the workspace bounds
        for _ in range(max_attempts):
            if np.random.rand() < 0.5:
                z = 0.0275 # Place on the ground
            else:
                z = np.random.uniform(workspace_bounds['z'][0], workspace_bounds['z'][1])  # Random height
            #z = 0.04

            position = [
                np.random.uniform(workspace_bounds['x'][0], workspace_bounds['x'][1]),  # x
                np.random.uniform(workspace_bounds['y'][0], workspace_bounds['y'][1]),  # y
                z   # z
            ]

            forbidden_start_rad = 160 * math.pi / 180
            forbidden_end_rad = 190 * math.pi / 180
            while True:
                # Generate theta in radians [0, 2*pi)
                theta = np.random.uniform(0, 2 * math.pi)
                if not (forbidden_start_rad <= theta <= forbidden_end_rad):
                    break # Valid theta found
            self.last_theta = theta
            x =  np.random.uniform(0.1, 0.5)* math.cos(theta)
            y =  np.random.uniform(0.1, 0.5)* math.sin(theta)
            position = [x, y, z]
            self.new_pos = np.array(position).copy()
            
            if not np.array_equal(position, avoid_position):
                # Position is valid, generate rotation and return
                axis = np.random.uniform(-1, 1, 3)  # Random vector in 3D space
                axis /= np.linalg.norm(axis)  # Normalize the vector
                angle = np.random.uniform(0, 2 * math.pi)
                rotation = [axis[0], axis[1], axis[2], angle]
                return position, rotation
        
        # If no valid position is found after max_attempts, return None
        self.__node.get_logger().info("Warning: Could not generate a valid position after max attempts.")
        return [1,1,0.0275], [0,0,0,1]
    
    def _capture_callback(self, msg):
        """Callback function triggered by the 'capture_image' topic."""
        if msg.reset and msg.seed > 1 and msg.seed < 150: 
            os.makedirs(self.save_path, exist_ok=True)
            base_filename = f"{self.capture_count}_{msg.seed}"
            self._capture_multiple_views(base_filename)
            self.capture_count += 1
        

    def _capture_multiple_views(self, base_filename):
        supervisor = self.__robot

        viewpoint_def_name = "MAIN_VIEWPOINT"
      
        viewpoint_node = supervisor.getFromDef(viewpoint_def_name)
    
        position_field = viewpoint_node.getField("position")
        orientation_field = viewpoint_node.getField("orientation")
      
        for i, (pos, ori, view_name) in enumerate(self.viewpoints):
            view_filename = f"{base_filename}_{view_name}.png"
            filepath = os.path.join(self.save_path, view_filename)

            # Set the viewpoint using field access
            position_field.setSFVec3f(list(pos))
            orientation_field.setSFRotation(list(ori))

            # Export the image
            supervisor.exportImage(filepath, 100)

    def transform_pose_to_robot_frame(self, pose_in_camera):
        try:
            # Get transform from camera to robot base
            transform = self.tf_buffer.lookup_transform(
                'base_link',        # Target frame
                'camera_link',       # Source frame
                rclpy.time.Time(),   # Time (0 = latest)
                timeout=rclpy.duration.Duration(seconds=0.1)
            )
            
            # Convert pose to TransformStamped
            pose_transform = TransformStamped()
            pose_transform.transform.translation.x = pose_in_camera.position.x
            pose_transform.transform.translation.y = pose_in_camera.position.y
            pose_transform.transform.translation.z = pose_in_camera.position.z
            pose_transform.transform.rotation = pose_in_camera.orientation
            
            # Apply transformation
            transformed_pose = do_transform_pose(
                pose_transform, 
                transform
            )
            
            return transformed_pose
            
        except (LookupException, 
                ConnectivityException, 
                ExtrapolationException) as e:
            self.get_logger().error(f'Transform failed: {str(e)}')
            return None

    def getObjectCamera_callback(self, msg: Pose):
        """
        This is the main entry point for processing object pose data from the camera.
        It directs the data to the appropriate handler based on the system's state.
        """
        if msg is None:
            return

        # State-driven logic: either calibrate or track.
        if self.is_calibrating:
            # We are in the initial setup phase.
            self.collect_calibration_sample(msg)
        else:
            # Calibration is complete, now we track the object.
            self.run_tracking_step(msg)

    # ==============================================================================
    # Initial Calibration Logic
    # ==============================================================================

    def collect_calibration_sample(self, camera_msg: Pose):
        """
        Collects a single sample for the initial calibration. Once enough samples
        are collected, it finalizes the calibration and switches to tracking mode.
        
        This function assumes the robot's gripper is holding the object during calibration.
        """
        try:
            # 1. Get the TRUE position of the gripper ('tool0'). This is our ground truth.
            now = rclpy.time.Time()
            trans = self.tf_buffer.lookup_transform('base_link', 'tool0', now, timeout=rclpy.duration.Duration(seconds=0.1))
            true_gripper_pos = np.array([trans.transform.translation.x, trans.transform.translation.y, trans.transform.translation.z])

            # 2. Get the MEASURED position of the object from the camera.
            measured_box_pos = np.array([camera_msg.position.x, camera_msg.position.y, camera_msg.position.z])

            # 3. Calculate the offset for this single frame: Offset = True - Measured
            current_offset = true_gripper_pos - measured_box_pos
            self.calibration_offsets.append(current_offset)
            self.calibration_samples_collected += 1

            self.__node.get_logger().info(
                f"[CALIBRATING] Sample {self.calibration_samples_collected}/{self.num_calibration_samples} collected."
            )

            # 4. If we have collected enough samples, finalize the calibration.
            if self.calibration_samples_collected >= self.num_calibration_samples:
                self._finalize_initial_calibration()

        except (LookupException, ConnectivityException, ExtrapolationException) as e:
            self.__node.get_logger().warn(f'Could not get transform for calibration sample: {e}. Retrying...')

    def _finalize_initial_calibration(self):
        """
        Averages the collected offsets, sets the final `calibrated_offset`,
        and transitions the system to tracking mode.
        """
        if not self.calibration_offsets:
            self.__node.get_logger().error("Cannot finalize: No calibration samples were collected.")
            return
            
        # Calculate the final, robust offset by averaging the samples.
        self.calibrated_offset = np.mean(list(self.calibration_offsets), axis=0)
        
        # --- TRANSITION TO TRACKING MODE ---
        self.is_calibrating = False
        self.calibration_offsets.clear()  # IMPORTANT: Clear the buffer for its next purpose (live refinement).
        
        self.__node.get_logger().info("\n--- Initial Calibration Complete ---")
        self.__node.get_logger().info(f"Final Averaged Offset: {self.calibrated_offset}")
        self.__node.get_logger().info("System is now in TRACKING mode.\n")

    # ==============================================================================
    # Tracking and Live Refinement Logic
    # ==============================================================================

    def run_tracking_step(self, msg: Pose):
        """
        Tracks the object in the simulation. Applies a deadband filter to reduce jitter
        and performs live drift correction when the object is stationary.
        """
        # 1. Get measured position and apply the master offset.
        measured_pos = np.array([msg.position.x, msg.position.y, msg.position.z])
        corrected_pos = measured_pos + self.calibrated_offset
        corrected_pos[2] = 0.0275  # Keep it on the ground plane.

        # 2. Add data to the buffer for potential live re-calibration.
        #    We now store (raw_camera_pos, corrected_sim_pos) tuples.
        self.calibration_offsets.append((measured_pos, corrected_pos))
        #self._check_and_perform_live_recalibration()

        # 3. Apply the deadband/jitter filter to decide if we update Webots.
        if self.last_sent_position is None:
            self.last_sent_position = corrected_pos

        distance_change = np.linalg.norm(corrected_pos - self.last_sent_position)
        should_update = (distance_change > self.MOVE_THRESHOLD) or (self.skipped_frames_count >= self.MAX_SKIPPED_FRAMES)

        if self.__boxes and should_update:
            object_node = self.__boxes[-1]
            if object_node:
                # Update Webots with the corrected pose.
                object_node.getField('translation').setSFVec3f(corrected_pos.tolist())
                
                cam_orient = [msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w]
                angle = 2 * np.arccos(cam_orient[3])
                axis = [1, 0, 0] # Default axis
                if angle > 1e-5:
                    s = np.sqrt(1 - cam_orient[3]**2)
                    if s > 1e-5:
                        axis = [cam_orient[0]/s, cam_orient[1]/s, cam_orient[2]/s]
                object_node.getField('rotation').setSFRotation(axis + [angle])
                
                # Reset filter state.
                self.last_sent_position = corrected_pos
                self.skipped_frames_count = 0
        else:
            # Skip the update and increment the counter.
            self.skipped_frames_count += 1
            
    def _check_and_perform_live_recalibration(self):
        """
        Helper function to check for object stability and refine the calibration offset
        to counter long-term drift.
        """
        # Don't do anything until the buffer is full.
        if len(self.calibration_offsets) < self.num_calibration_samples:
            return

        # Check for stability using the standard deviation of corrected positions.
        corrected_positions_in_buffer = [pose[1] for pose in self.calibration_offsets]
        std_dev = np.std(corrected_positions_in_buffer, axis=0)
        
        # If the object is stable and we have a reference point...
        if np.all(std_dev < self.STABILITY_THRESHOLD) and self.last_sent_position is not None:
            self.__node.get_logger().info(f"[INFO] Object stable. Performing live drift correction...")

            # The "ground truth" is the last position we commanded the simulation to use.
            ground_truth_pos = self.last_sent_position
            
            # The "measured value" is the average of the raw measurements from the camera.
            raw_positions_in_buffer = [pose[0] for pose in self.calibration_offsets]
            average_measured_pos = np.mean(raw_positions_in_buffer, axis=0)
            
            # Calculate the new, more accurate offset.
            new_offset = ground_truth_pos - average_measured_pos
            
            # Smoothly update the master offset to prevent a sudden jump.
            smoothing_alpha = 0.5
            self.calibrated_offset = (smoothing_alpha * new_offset) + ((1 - smoothing_alpha) * self.calibrated_offset)
            self.__node.get_logger().info(f"  -> New Master Offset: {self.calibrated_offset}")

            # CRITICAL: Clear the buffer to reset the stability check.
            self.calibration_offsets.clear()

    def publish_box_pose(self):
        for i, box in enumerate(self.__boxes):
            # Retrieve position and orientation from the box
            position = self.__boxes[i].getPosition()
            orientation = self.__boxes[i].getOrientation()  # 9-element rotation matrix (row-major)

            # Convert the 9-element list to a 3x3 rotation matrix
            rot_matrix = np.array(orientation).reshape(3, 3)

            # Create a 4x4 identity matrix and insert the 3x3 rotation matrix
            rot_4x4 = np.eye(4)
            rot_4x4[:3, :3] = rot_matrix

            # Convert the 4x4 matrix to a quaternion
            quat = tr.quaternion_from_matrix(rot_4x4)

            # Create Position and Orientation messages
            pos_msg = Point()
            pos_msg.x = position[0]
            pos_msg.y = position[1]
            pos_msg.z = position[2]

            orient_msg = Quaternion()
            orient_msg.x = quat[0]
            orient_msg.y = quat[1]
            orient_msg.z = quat[2]
            orient_msg.w = quat[3]

            # Publish the Pose message
            self.__pose_publishers[i].publish(Pose(position=pos_msg, orientation=orient_msg))
      
    def publish_fd_pose(self):
        # Retrieve position and orientation from the box
        position = self.__boxes[-1].getPosition()
        orientation = self.__boxes[-1].getOrientation()  # 9-element rotation matrix (row-major)
        
        # Convert the 9-element list to a 3x3 rotation matrix
        rot_matrix = np.array(orientation).reshape(3, 3)

        # Create a 4x4 identity matrix and insert the 3x3 rotation matrix
        rot_4x4 = np.eye(4)
        rot_4x4[:3, :3] = rot_matrix

        # Convert the 4x4 matrix to a quaternion
        quat = tr.quaternion_from_matrix(rot_4x4)

        # Create Position and Orientation messages
        pos_msg = Point()
        pos_msg.x = position[0]
        pos_msg.y = position[1]
        pos_msg.z = position[2]

        orient_msg = Quaternion()
        orient_msg.x = quat[0]
        orient_msg.y = quat[1]
        orient_msg.z = quat[2]
        orient_msg.w = quat[3]

        # Publish the Pose message
        self.fd_publisher.publish(Pose(position=pos_msg, orientation=orient_msg))
      
    def step(self):
        # self.publish_box_pose()
        # self.publish_fd_pose()
        # rclpy.spin_once(self.__node, timeout_sec=0)
        pass
        # request_reset = False
        # if time.time() - self.last_time_reset > 30:
        #     request_reset = True
        # if self.__reset_processed or request_reset:
        #     done = False
        #     current_pos = self.__boxes[0].getPosition()
        #     tolerance = 0.01
        #     if np.sum(np.abs(current_pos - self.new_pos)) <= tolerance:
        #         done = True
        #     else:
        #         self.__boxes[0].getField('translation').setSFVec3f(self.new_pos.tolist())
        #     if done:
        #         reset_msg = Bool(data=True)
        #         self.reset_publisher.publish(reset_msg)
        #         self.__reset_processed = False
        #         self.last_time_reset = time.time() 
        
        



