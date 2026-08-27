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

def quaternion_to_rotation_matrix(quaternion):
    import numpy as np
    import tf_transformations
    quaternion_np = np.array(quaternion, dtype=np.float64)
    rotation_matrix_4x4 = tf_transformations.quaternion_matrix(quaternion_np)
    rotation_matrix_3x3 = rotation_matrix_4x4[:3, :3]
    return rotation_matrix_3x3

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
        self.__node.create_subscription(Pose,f'/azure_kinect/object', self.getObjectCamera_callback,1)
        
         # --- ADAPTIVE CALIBRATION & TRACKING VARIABLES ---
        # The master offset, continuously refined. Can be initialized with a good guess or zeros.
        self.calibrated_offset = np.array([0.205, 0.058, 0.0]) # Your previous example [0.249-0.044, -0.009-(-0.067), 0.0]

        # The max distance (meters) between gripper and box for calibration to trigger.
        self.CALIBRATION_PROXIMITY_THRESHOLD = 0.2 # 1 cm

        # The smoothing factor for the exponential moving average (EMA).
        # A smaller alpha means the offset changes more slowly.
        # new_offset = alpha * current_sample + (1 - alpha) * old_offset
        self.CALIBRATION_SMOOTHING_ALPHA = 0.1 # A value of 0.1 gives 10% weight to the new sample.

        # Initialize TF2 listener
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self.__node)

        # --- JITTER FILTER VARIABLES ---
        self.MOVE_THRESHOLD = 0.01
        self.MAX_SKIPPED_FRAMES = 10
        self.last_sent_position = None
        self.skipped_frames_count = 0

        self.last_update_time = time.time()

        # Create 10 boxes
        for i in range(self.box_count):
            self._create_box(i,add_physics=True)
    
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

    def compute_gripper_pinch_pos(self,pos,rot_matrix, offset=[0,0,0.07]):
        ##This function computes roughly the pinch position of the fingers (i.e. the position the fingers would meet when closing)
        ##This is based on the position and orientation of the gripper (see webots documentation) and gripper specification
        orientation = np.reshape(rot_matrix, (3,3))
        offset = np.array(offset) #pinch position is roughly 15cm from gripper pos in z-direction in gripper coordinate system
        pos_offset = np.matmul(orientation, offset) + pos
        return pos_offset
    
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

    
    def getObjectCamera_callback(self, camera_msg: Pose):
        """
        This is the unified handler for all incoming camera data. It performs two main tasks:
        1. Opportunistically refines the calibration offset if the gripper is near the box.
        2. Tracks the object in the simulation using the latest calibrated offset.
        """
        if camera_msg is None:
            return
        current_time = time.time()
        # throttling to 10 Hz
        # if (current_time - self.last_update_time) < 0.1:
        #     return 
        self.last_update_time = current_time
        self._refine_calibration_if_possible(camera_msg)
        self._track_object_in_simulation(camera_msg)


    def _refine_calibration_if_possible(self, camera_msg: Pose):
        """
        Checks if the gripper is close to the box. If so, it calculates a new
        offset sample and smoothly updates the master `calibrated_offset`.
        """
        try:
            # Get the current positions of the gripper and the camera's measurement
            now = rclpy.time.Time()
            trans = self.tf_buffer.lookup_transform('base_link', 'rebel_link8', now, timeout=rclpy.duration.Duration(seconds=0.0))
            true_gripper_pos = np.array([trans.transform.translation.x, trans.transform.translation.y, trans.transform.translation.z])
            gripper_orientation = quaternion_to_rotation_matrix([trans.transform.rotation.x, trans.transform.rotation.y, trans.transform.rotation.z, trans.transform.rotation.w])
            pinch_pos = self.compute_gripper_pinch_pos(true_gripper_pos, gripper_orientation)
            measured_box_pos = np.array([camera_msg.position.x, camera_msg.position.y, camera_msg.position.z])

            # Check the distance between gripper and detected box
            distance = np.linalg.norm(pinch_pos - measured_box_pos)
            self.__node.get_logger().info(
                f"[CALIBRATING] Gripper is close to box (Dist: {distance:.3f}m). Refining offset."
            )
            # If they are not close, do nothing.
            if distance > self.CALIBRATION_PROXIMITY_THRESHOLD:
                return

            # --- CONDITIONS MET: REFINE THE CALIBRATION ---
            self.__node.get_logger().info(
                f"[CALIBRATING] Gripper is close to box (Dist: {distance:.3f}m). Refining offset."
            )

            # Calculate a new offset sample: Offset = True Position - Measured Position
            offset_sample = pinch_pos - measured_box_pos

            # Smoothly update the master offset using an Exponential Moving Average (EMA)
            # This blends the new sample with the existing offset, preventing jumps.
            self.calibrated_offset = (self.CALIBRATION_SMOOTHING_ALPHA * offset_sample) + \
                                    ((1.0 - self.CALIBRATION_SMOOTHING_ALPHA) * self.calibrated_offset)

            self.__node.get_logger().info(f"  -> New refined offset: {self.calibrated_offset}")

        except (LookupException, ConnectivityException, ExtrapolationException) as e:
            # This is not a critical error, just means we can't calibrate on this frame.
            self.__node.get_logger().debug(f'Could not get gripper transform for calibration check: {e}')


    def _track_object_in_simulation(self, msg: Pose):
        """
        Applies the current `calibrated_offset` and a deadband filter to update
        the object's pose in the Webots simulation.
        (This is your previous run_tracking_step, slightly renamed for clarity)
        """
        # 1. Get the measured position and apply the master offset.
        measured_pos = np.array([msg.position.x, msg.position.y, msg.position.z])
        corrected_pos = measured_pos + self.calibrated_offset
        corrected_pos[2] = 0.0275  # Hardcode Z-position.

        # 2. Apply the deadband/jitter filter.
        if self.last_sent_position is None:
            self.last_sent_position = corrected_pos

        distance_change = np.linalg.norm(corrected_pos - self.last_sent_position)
        should_update = (distance_change > self.MOVE_THRESHOLD) or \
                        (self.skipped_frames_count >= self.MAX_SKIPPED_FRAMES)

        # 3. Perform the update in Webots if the condition is met.
        if self.__boxes and should_update:
            object_node = self.__boxes[-1]
            if object_node:
                # Update translation
                object_node.getField('translation').setSFVec3f(corrected_pos.tolist())

                # Update rotation
                cam_orient = [msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w]
                angle = 2 * np.arccos(cam_orient[3])
                axis = [1, 0, 0] # Default axis
                if angle > 1e-5:
                    s = np.sqrt(1 - cam_orient[3]**2)
                    if s > 1e-5:
                        axis = [cam_orient[0]/s, cam_orient[1]/s, cam_orient[2]/s]
                object_node.getField('rotation').setSFRotation(axis + [angle])
                
                # Reset filter state
                self.last_sent_position = corrected_pos
                self.skipped_frames_count = 0
        else:
            # Skip the update and increment counter
            self.skipped_frames_count += 1


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
        self.publish_box_pose()
        self.publish_fd_pose()
        rclpy.spin_once(self.__node, timeout_sec=0)
        
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
        
        



