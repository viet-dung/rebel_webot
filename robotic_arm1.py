import rclpy
from rclpy.node import Node
from trajectory_msgs.msg import JointTrajectory
from std_msgs.msg import String, Bool
import numpy as np
from geometry_msgs.msg import Point, Quaternion, Pose
import tf_transformations as tr
from sensor_msgs.msg import JointState
from ros2_message_interfaces.msg import ResetSeed 
import os
import time
import math
import tf_transformations


# Real Robot Configuration
BRAKING_FACTOR_ = 0.875
MIN_STATIC_BUFFER_ = 0.02
LINK_7_ON_GROUND_ = 0.055  # Height threshold for link 7 to be considered on ground
TOOL0_ON_GROUND_ = 0.029  # Height threshold for tool0 to be considered on ground

def set_seed(random_seed):
    import random
    import numpy as np
    np.random.seed(random_seed)
    random.seed(random_seed)


def quaternion_to_axis_angle(quaternion):
    """Converts a quaternion [x, y, z, w] to an axis-angle [ax, ay, az, angle] list."""
    w = max(-1.0, min(1.0, quaternion[3])) # Clamp w
    angle = 2 * np.arccos(w)
    
    if np.isclose(angle, 0.0):
        return [0, 0, 1, 0] # Default axis for zero rotation
        
    s = np.sqrt(1 - w*w)
    axis = [
        quaternion[0] / s,
        quaternion[1] / s,
        quaternion[2] / s
    ]
    return axis + [angle]

class RoboticArm:
    def init(self, webots_node, properties):
        self.__robot = webots_node.robot
        self.__port = properties['port']
        self.__env_id = properties['env_id']
        os.environ['ROS_DOMAIN_ID'] = str(self.__env_id)

        rclpy.init(args=None) ### IMPORTANT NEED TO BE CALL ONLY ONCE TIME FOR THE WHOLE LAUNCH FILE
        self.__node = rclpy.create_node('robotic_arm_controller', namespace=f'igus_rebel_{self.__port}')

        self.__timestep = int(self.__robot.getBasicTimeStep())
        
        self.__node.get_logger().info(f"Webots port: {self.__port}") # DEBUG - using logger

        # Initialize joint motors and sensors
        self.__joint_names = ['joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6']
        self.__joint_limits = { # Full joint limits for normal operation
            'joint1': (-3.141593, 3.141593),
            'joint2': (-1.48353, 1.9),
            'joint3': (-1.39626, 1.7),
            'joint4': (-3.141593, 3.141593),
            'joint5': (-1.65806, 1.65806),
            'joint6': (-3.141593, 3.141593),
        }
        self.__reset_joint_limits = { # Narrower limits for reset - ADJUST THESE VALUES
                'joint1': (-2.0, 2.0),    # (-3.0, 3.0)
                'joint2': (-0.5, -0.1),    # (-1, 0)
                'joint3': (-1.2, 1.2),    # (-1.2, 1.7)
                'joint4': (-3.0, 3.0),    
                'joint5': (-0.5, 0.5),  
                'joint6': (-3.0, 3.0),    
            }
        # self.__reset_joint_limits_push = { # Narrower limits for reset - ADJUST THESE VALUES
        #         'joint1': (-3.0, 3.0),    # (-3.0, 3.0)
        #         'joint2': (1, 1.2),    # (-1, 0)
        #         'joint3': (-0.56, -0.4),    # (-1.2, 1.7)
        #         'joint4': (0.4, 0.5),    
        #         'joint5': (1.5, 1.6),  
        #         'joint6': (-3.0, 3.0),    
        #     }
        self.__reset_joint_limits_push = {
            'joint1': (-2.0, 2.0),      # Limit rotation to be mostly forward -0.2, 0.2
            'joint2': (1.1, 1.1),       # Keep arm low 1.1
            'joint3': (-0.6, -0.4),     # Adjust for forward reach
            'joint4': (0.3, 0.6),
            'joint5': (1.4, 1.6),       # Keep wrist pointing down
            'joint6': (-0.2, 0.2),
        }
        
        self.__joint_motors = []
        self.__joint_sensors = []
        self.__max_velocities = {} # Store max velocities for each joint
        self.current_joint_vel = np.zeros(6)

        # Get the handle to the link node using its DEF name
        robot_supervisor_node = self.__robot.getSelf()
        self.__link7_node = robot_supervisor_node.getFromProtoDef('rebel_link7_node')
      
        # Initialize gripper
        self.__gripper = self.__robot.getFromDef('gripper') # For determining pinch pose of the gripper
        self.__gripper_motors = []
        self.__gripper_sensors = []
        self.__gripper_max_velocities = {}
        self.__gripper_joint_limits = [(0,0.3625)]

        # Variables for the pinch position visualizer
        self.__gripper_pinch_position = np.zeros(3)  # Initialize to origin
        self.__pinch_visualizer_node = None
        self.__pinch_visualizer_translation_field = None
        self.__box_pose_viz = None  # Initialize to origin for box position visualizer
        self.box_node = None  # Initialize box node to None
        self.set_box_pose = False  
        self.target_random_box = np.zeros(3) 
        self.target_random_box_previous = np.zeros(3)
        self.current_box_pose = np.zeros(7)   

        # Collision detection
        self.robot_node = self.__robot.getFromDef('igus_rebel')
        # Enable contact point tracking for the robot and its descendants
        self.robot_node.enableContactPointsTracking(self.__timestep)
  
        # Reset state machine
        self.__reset_in_progress = False
        self.__reset_in_progress_world = False
        self.reset_ack = False
        
        self.__target_positions = np.ones(6) * (-1)
        self.__current_reset_joint = 0
        self.__target_reset_attemp = 0

        # Initialize variables for Enable Gravity
        self.__isgrasp = False
        self.__target_box = None

        self.__prev_joint_positions = None
        self.__prev_time = None

        self.__prev_gripper_joint_positions = None
        self.__prev_gripper_time = None

        self.last_time_arm_callback = 0
        self.last_time_reset_callback = 0
        self.last_sync_timestamp = -1
        self.waiting_after_reset = False
        self.world_reload = False
        self.reset_callback_msg = ResetSeed(reset=False)
        self.reset_box = False
        self.reset_seed = -1
        self.set_seed_once = True  # Flag to set the seed only once

        # === NEW VARIABLES FOR FRAME SKIPPING ===
        self.__frames_to_run = 0         # How many frames the current action should last
        self.__frames_elapsed = 0        # How many frames have passed for the current action
        self.stuck_flag = False  # Flag to indicate if the robot is stuck
       
        # ROS2 publishers
        self.__gripper_publisher = self.__node.create_publisher(Pose, 'pose/gripper', 10)
        self.__link7_publisher = self.__node.create_publisher(Point, 'pos/link7', 10)
        self.__joint_state_publisher = self.__node.create_publisher(JointState, 'custom_joint_states', 10)
        self.gripper_joint_state_publisher = self.__node.create_publisher(JointState, 'gripper/joint_states', 10)
        self.reset_publisher = self.__node.create_publisher(Bool, 'reset_arm_done', 10)
        self.reset_box_pub = self.__node.create_publisher(ResetSeed, 'reset_boxes', 10)
        self.target_random_box_pub = self.__node.create_publisher(Point, 'target_box_pos', 10)

        # ROS2 subscribers for joint and gripper control
        self.__node.create_subscription(JointTrajectory, 'arm_controller/command', self.__arm_trajectory_callback, 1)
        self.__node.create_subscription(JointTrajectory, 'gripper_controller/command', self.__gripper_trajectory_callback, 1)
        self.__node.create_subscription(ResetSeed, 'reset_arm', self.__reset_callback, 1)
        self.__node.create_subscription(Bool, 'reset_world', self.__reset_world, 1)
        #self.__node.create_subscription(Bool, 'reset_ack', self.__reset_ack_callback, 1)
        self.__node.create_subscription(Pose, 'box_pos_viz', self.box_pos_viz_callback, 1)

        # ROS2 subscribers for next target box 
        self.__node.create_subscription(String, 'target_box', self.__target_box_callback, 1)

        self.on_init()
        root = self.__robot.getRoot()
        self.__root_children = root.getField('children')
        # for i in range(1):
        #     self._create_box(i,add_physics=True)
        #     self.box_pub = self.__node.create_publisher(Pose, f'box_{i}/pose', 10)

    def _create_box(self, index, set_contact_properties=True, avoid_position=[0,0,0], add_physics=True):
        """Helper function to create a box with a random position and orientation."""
        box_def = f'box_{index}'
        box_material_name = f'box_material_{index}' 
        box_pose  = self.random_box_pose()
        pos = box_pose[:3]
        rot = quaternion_to_axis_angle(box_pose[3:])
        box_str = f'''
        DEF {box_def} Solid {{
            translation {pos[0]} {pos[1]} {pos[2]}
            rotation {rot[0]} {rot[1]} {rot[2]} {rot[3]}
            contactMaterial "{box_material_name}" # Unique contactMaterial for each box
            children [
                Shape {{
                    geometry Box {{ size 0.1 0.1 0.1 }}
                    appearance PBRAppearance {{
                        baseColor {abs(pos[0])} {abs(pos[1])} {abs(pos[2])}
                        roughness 1
                        metalness 0
                    }}
                }}
            ]
            name "{box_def}"
            boundingObject Box {{ size 0.1 0.1 0.1 }}
        }}
        '''
        self.__root_children.importMFNodeFromString(-1, box_str)
        box_node = self.__robot.getFromDef(box_def)
        self.box_node = box_node
        
        if box_node is None:
            self.__node.get_logger().info(f"Error: Failed to find node with DEF '{box_def}' in the scene tree.")
        else:
            self.__node.get_logger().info(f"Successfully created box with DEF '{box_def}'.")

        physics_field = box_node.getField('physics')  # Get the 'physics' field
        physics_node = physics_field.getSFNode()
        if not physics_node and add_physics:  # If physics is NULL
            physics_str = '''
            Physics {
                mass 0.5  # Set the mass
            }
            '''
            physics_field.importSFNodeFromString(physics_str)  # Import the physics node


    def random_box_pose(self):
        # This function will loop until a valid position is found that is outside the forbidden zone.
        # Get the gripper's current position as the center point
        center_pos = self.__gripper.getPosition()  # Get the gripper's current position
        # x = center_pos[0] + 0.05 if center_pos[0] > 0 else center_pos[0] - 0.05
        # y = center_pos[1] + 0.05 if center_pos[1] > 0 else center_pos[1] - 0.05
        #distance = np.random.uniform(0.1, 0.2)*np.random.choice([-1,1]) # 10-15cm in front
        distance = np.random.uniform(0.07, 0.1)
        angle_offset = np.random.uniform(-0.2, 0.2)
        x = center_pos[0] + distance  * math.cos(angle_offset)
        y = center_pos[1] + distance  * math.sin(angle_offset)

        x = np.clip(x, -0.6, 0.6)
        y = np.clip(y, -0.6, 0.6)
        z = 0.0
     
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

          #self.__node.get_logger().info(f"Random box target position: {np.linalg.norm([x-center_pos[0], y-center_pos[1]])}") # DEBUG - using logger
        if abs(x) >= 0.3 or abs(y) >= 0.3:
            return pose
        
        if x <= 0.1 and x > 0:
            x += 0.2
        if x >= -0.1 and x < 0:
            x -= 0.2
        if y <= 0.25 and y > 0:
            y += 0.3
        if y >= -0.25 and y < 0:
            y -= 0.3
        x = np.clip(x, -0.6, 0.6)
        y = np.clip(y, -0.6, 0.6)
        pose[:2] = np.array([x, y])
        self.current_box_pose = pose
        return pose
    
    def random_box_target(self):
        # This function will loop until a valid position is found that is outside the forbidden zone.
        # Get the gripper's current position as the center point
        center_pos = self.current_box_pose
        
        #distance = np.random.uniform(0.12, 0.2) # 0.65, 0.2
        distance = np.random.uniform(0.22, 0.25)
        phi = np.random.uniform(-math.pi/2, math.pi/2)
        
        offset_x = distance  * math.cos(phi)
        offset_y = distance  * math.sin(phi)
        
        x = center_pos[0] + offset_x
        y = center_pos[1] + offset_y

        # x = np.clip(x, -0.7, 0.7)
        # y = np.clip(y, -0.7, 0.7)

        # x = np.clip(x, -0.9, 0.9)
        # y = np.clip(y, -0.9, 0.9)

        #self.__node.get_logger().info(f"Random box target position: {np.linalg.norm([x-center_pos[0], y-center_pos[1]])}") # DEBUG - using logger
        if abs(x) >= 0.1 or abs(y) >= 0.25:
            return np.array([x, y, 0.0275])
        if x < 0.2 and x > 0:
            x += 0.2
        if y < 0.2 and y > 0:
            y += 0.2
        if x > -0.2 and x < 0:
            x -= 0.2
        if y > -0.2 and y < 0:
            y -= 0.2
        return np.array([x, y, 0.0275])

    def on_init(self):
        for i in range(1, 7):  # Joints 1 to 6
            motor = self.__robot.getDevice(f'joint{i}')
            sensor = self.__robot.getDevice(f'joint{i}_sensor')
            motor.setPosition(float('inf'))  # Enable velocity control
            motor.setVelocity(0.0)  # Initial velocity
            sensor.enable(self.__timestep)
            self.__joint_motors.append(motor)
            self.__joint_sensors.append(sensor)
            self.__max_velocities[f'joint{i}'] = motor.getMaxVelocity() # Get max velocity from Webots
            self.__joint_limits[f'joint{i}'] = (motor.getMinPosition(), motor.getMaxPosition())

        for i in range(3):
            motor = self.__robot.getDevice(f'finger_{i}_joint')
            sensor = self.__robot.getDevice(f'finger_{i}_joint_sensor')
            sensor.enable(self.__timestep)
            self.__gripper_motors.append(motor)
            self.__gripper_sensors.append(sensor)
            self.__gripper_max_velocities[f'finger_{i}_joint'] = motor.getMaxVelocity()
            self.__gripper_joint_limits[0] = (motor.getMinPosition(), motor.getMaxPosition())
        self.__create_pinch_visualizer()

    def check_self_collision(self):
        """Check for self-collisions based on distance thresholds and floor contact."""
        
        contact_points = self.robot_node.getContactPoints(includeDescendants=True)
        
        return any(
            x < 0.05 and y < 0.05  and z < 0.3 #or z < 0.01
            for cp in contact_points
            for x, y, z in [cp.getPoint()]
        )
    
    def check_ode_error(self):
        for i in range(len(self.__joint_sensors)):
            current_pos = self.__joint_sensors[i].getValue()
            joint_name = self.__joint_names[i]
            min_limit, max_limit = self.__joint_limits[joint_name]
            if abs(current_pos) > max_limit*3:
                return True
        return False

    def compute_gripper_pinch_pos(self,pos,rot_matrix, offset=[0,0,0.07]):
        ##This function computes roughly the pinch position of the fingers (i.e. the position the fingers would meet when closing)
        ##This is based on the position and orientation of the gripper (see webots documentation) and gripper specification
        orientation = np.reshape(rot_matrix, (3,3))
        offset = np.array(offset) #pinch position is roughly 15cm from gripper pos in z-direction in gripper coordinate system
        pos_offset = np.matmul(orientation, offset) + pos
        return pos_offset
    
    def __create_pinch_visualizer(self):
        """Creates a visual sphere in the world to mark the pinch position."""
        root_node = self.__robot.getRoot()
        children_field = root_node.getField('children')
        
        # Define the sphere node as a string
        # We give it a DEF name 'PINCH_VIZ' to find it easily later
        node_string = """
            DEF PINCH_VIZ Transform {
              translation 0 0 0 # Initial position
              children [
                Shape {
                  appearance Appearance {
                    material Material {
                      diffuseColor 1 1 0  # Yellow
                      emissiveColor 1 0 0 # Bright red
                    }
                  }
                  geometry Sphere {
                    radius 0.01
                    subdivision 2
                  }
                }
              ]
            }
        """
        # Add the node to the world
        children_field.importMFNodeFromString(-1, node_string)

        # Get the handle to the node and its translation field for later updates
        self.__pinch_visualizer_node = self.__robot.getFromDef('PINCH_VIZ')
        if self.__pinch_visualizer_node:
            self.__pinch_visualizer_translation_field = self.__pinch_visualizer_node.getField('translation')
            self.__node.get_logger().info("Successfully created pinch position visualizer.")
        else:
            self.__node.get_logger().error("Failed to create pinch position visualizer.")

    def __arm_trajectory_callback(self, trajectory):
        """
        Callback for controlling the robotic arm joints (VELOCITY control).
        Now explicitly reads velocities from trajectory.points[-1].velocities.
        """
        last_sync_timestamp = trajectory.points[-1].effort[0]

        if not self.__reset_in_progress and not self.last_sync_timestamp == last_sync_timestamp:
            self.last_sync_timestamp = last_sync_timestamp
            self.last_time_arm_callback = self.__node.get_clock().now().nanoseconds

            # Reset complete, switch back to velocity control
            for motor in self.__joint_motors:
                motor.setPosition(float('inf'))  # Velocity control mode
            
            if not trajectory.points:
                self.__node.get_logger().info("Received trajectory message with no points. Ignoring.") # DEBUG - using logger
                return

            velocities = trajectory.points[-1].velocities # <--- Read from 'velocities' field NOW
            #self.__node.get_logger().info(f"Received joint VELOCITIES: {velocities}") # DEBUG - using logger
            
            # if self.__isgrasp:
            #     box = self.__robot.getFromDef(self.__target_box)
            #     box_pos = np.array(box.getPosition())
            #     pos_gripper = self.__gripper.getPosition()
            #     orientation_gripper = self.__gripper.getOrientation()

            #     gripper_pinch_pos = np.array(self.compute_gripper_pinch_pos(pos_gripper, orientation_gripper))
            #     distance = np.linalg.norm(gripper_pinch_pos - box_pos)
                
            #     if distance < 0.1:
            #         self.enable_gravity()
            
            link7_pos = np.array(self.__link7_node.getPosition())
            pos_gripper = self.__gripper.getPosition()
            orientation_gripper = self.__gripper.getOrientation()
            self.__gripper_pinch_position = np.array(self.compute_gripper_pinch_pos(pos_gripper, orientation_gripper))
            gripper_pinch_pos = self.__gripper_pinch_position
            on_ground = False
            is_near_ground = (link7_pos[2] <= LINK_7_ON_GROUND_) or (gripper_pinch_pos[2] <= TOOL0_ON_GROUND_)

            for i, velocity in enumerate(velocities):
                if velocity == 0:
                    self.__joint_motors[i].setVelocity(0)
                else:
                    # Check self-collision
                    # if self.check_self_collision():
                    #     clamped_velocity = 0.0

                    joint_name = self.__joint_names[i]
                    max_vel = self.__max_velocities[joint_name]
                    clamped_velocity = np.clip(velocity, -max_vel, max_vel)

                    # NEW: Check joint position limits
                    current_pos = self.__joint_sensors[i].getValue()
                    min_limit, max_limit = self.__joint_limits[joint_name]

                    is_moving_down = clamped_velocity * current_pos > 0 
                    if (i in {1,2} and is_near_ground and is_moving_down):
                        clamped_velocity = 0 
                        on_ground = True
                        self.__joint_motors[i].setVelocity(clamped_velocity)
                        continue

                    if on_ground and i in {3,4,5}:
                        self.__joint_motors[i].setVelocity(0)
                        continue

                    # Prevent velocity from moving joint beyond limits
                    braking_distance = abs(clamped_velocity) * BRAKING_FACTOR_
                    buffer = max(MIN_STATIC_BUFFER_, braking_distance)
                    if (current_pos <= min_limit+buffer and clamped_velocity < 0) or \
                        (current_pos >= max_limit-buffer and clamped_velocity > 0):
                            clamped_velocity = 0 
                    
                    self.current_joint_vel[i] = clamped_velocity
                    self.__joint_motors[i].setVelocity(clamped_velocity)


            # --- STORE THE COMMAND AND FRAME COUNT ---
            self.__frames_to_run = int(trajectory.points[-1].effort[1])  # Get frame_skip from the message
            self.__frames_elapsed = 0                 # Reset the counter for the new action
            
                
    def __gripper_trajectory_callback(self, trajectory):
        """
        Callback for controlling the gripper.
        """
        last_sync_timestamp = trajectory.points[-1].effort[0]

        if not self.__reset_in_progress and not self.last_sync_timestamp == last_sync_timestamp:
            velocity = trajectory.points[-1].velocities[0]

            if velocity > 0:
                self.__isgrasp = True
            else:
                self.__isgrasp = False

            min_pos, max_pos = self.__gripper_joint_limits[0]
            set_position = min_pos if velocity <= 0 else max_pos

            for i in range(3):
                self.__gripper_motors[i].setPosition(max_pos)
                self.__gripper_motors[i].setVelocity(self.__gripper_max_velocities[f'finger_{i}_joint'])
                
                
    def __target_box_callback(self, msg):
        self.__target_box_pos = msg.data

    # For real robot
    # def __execute_reset(self):
    #     if not self.__reset_in_progress:
    #         return

    #     position_tolerance = 0.1

    #     # Process one joint per simulation step
    #     joint_idx = self.__current_reset_joint
    #     target_pos = self.__target_positions[joint_idx]
    #     current_pos = self.__joint_sensors[joint_idx].getValue()
    #     error = target_pos - current_pos

    #     # Calculate velocity
    #     velocity = error
    #     joint_name = self.__joint_names[joint_idx]
    #     max_vel = self.__max_velocities[joint_name]
    #     clamped_velocity = np.clip(velocity, -max_vel, max_vel) # Clamp velocity for reset
    #     self.__joint_motors[joint_idx].setVelocity(clamped_velocity)

    #     if self.__current_reset_joint < 3:
    #         self.__gripper_motors[self.__current_reset_joint].setVelocity(-self.__gripper_max_velocities[f'finger_{self.__current_reset_joint}_joint'])  # Open gripper

    #     # Check if joint reached target
    #     if abs(error) <= position_tolerance:
    #         self.__current_reset_joint += 1

    #         # Check if all joints done
    #         if self.__current_reset_joint >= 6:
    #             # Finalize reset
    #             self.__current_reset_joint = 0
                
    #             self.__node.get_logger().info("Reset complete")
    #             self.__reset_in_progress = False

    def __reset_world(self,msg):
        if msg.data:
            needs_hard_reset = False
            for i in range(len(self.__joint_sensors)):
                current_pos = self.__joint_sensors[i].getValue()
                joint_name = self.__joint_names[i]
                min_limit, max_limit = self.__joint_limits[joint_name]
                if current_pos < min_limit or current_pos > max_limit:
                    needs_hard_reset = True
                    break

                #Check gripper joints against their normal limits
                # if not needs_hard_reset:
                #     for sensor in self.__gripper_sensors:
                #         current_pos = sensor.getValue()
                #         min_limit, max_limit = self.__gripper_joint_limits[0]
                #         if current_pos < min_limit*3 or current_pos > max_limit*3:
                #             needs_hard_reset = True
                #             break

            if needs_hard_reset:
                    # Perform hard reset by reloading the world
                    self.__reset_in_progress_world = True
                    


    def __reset_callback(self, msg):
        """
        Callback for the reset command.
        """
        if msg.reset and not self.__reset_in_progress:
            if self.set_seed_once:
                self.set_seed_once = False
                self.reset_seed = msg.seed
                set_seed(msg.seed)  # Set the random seed for reproducibility

            self.__reset_in_progress = True
            self.last_time_reset_callback = self.__node.get_clock().now().nanoseconds
            
            # Publish reset started flag
            reset_start_msg = Bool()
            reset_start_msg.data = False
            self.reset_publisher.publish(reset_start_msg)

            # Check if any joint or gripper is outside normal operational limits
            needs_hard_reset = False

            # Check arm joints against normal joint limits
            for i in range(len(self.__joint_sensors)):
                current_pos = self.__joint_sensors[i].getValue()
                joint_name = self.__joint_names[i]
                min_limit, max_limit = self.__joint_limits[joint_name]
                if current_pos < min_limit*3 or current_pos > max_limit*3:
                    needs_hard_reset = True
                    break

            # #Check gripper joints against their normal limits
            # if not needs_hard_reset:
            #     for sensor in self.__gripper_sensors:
            #         current_pos = sensor.getValue()
            #         min_limit, max_limit = self.__gripper_joint_limits[0]
            #         if current_pos < min_limit*3 or current_pos > max_limit*3:
            #             needs_hard_reset = True
            #             break

            if needs_hard_reset:
                self.__reset_in_progress_world = True
            else:
                # Generate random target positions - **USE __reset_joint_limits here**
                self.__target_positions = [
                    np.random.uniform(low, high)
                    for (low, high) in self.__reset_joint_limits.values() # Changed to __reset_joint_limits
                ]
                self.__target_positions = np.zeros(6)
                #self.__target_positions = np.array([sensor.getValue() for sensor in self.__joint_sensors])
                self.reset_callback_msg = msg

                # Reset box
                self.set_box_pose = True
                # self.box_node.getField("translation").setSFVec3f(list([1,1,0]))
                #mass = np.random.uniform(0.1, 1.0)
                #size = np.random.uniform(0.06, 0.1)
                #self.box_node.getField('physics').getSFNode().getField('mass').setSFFloat(mass)
                #self.box_node.getField('boundingObject').getSFNode().getField('size').setSFVec3f([size, size, size])
                
            
    def __reset_ack_callback(self, msg):
        self.reset_ack = msg.data
               
    def enable_gravity(self):
        box = self.__robot.getFromDef(self.__target_box)
        physics_field = box.getField('physics')  # Get the 'physics' field
        physics_node = physics_field.getSFNode()
        if not physics_node:  # If physics is NULL
            physics_str = '''
            Physics {
                mass 0.5  # Set the mass
            }
            '''
            physics_field.importSFNodeFromString(physics_str)  # Import the physics node


    def publish_joint_states(self):
        current_time = self.__robot.getTime()
        current_joint_positions = [sensor.getValue() for sensor in self.__joint_sensors]
        
        # Compute joint velocities
        joint_velocities = []
        if self.__prev_time is not None and current_time != self.__prev_time:
            dt = current_time - self.__prev_time
            joint_velocities = [(current_joint_positions[i] - self.__prev_joint_positions[i]) / dt 
                                for i in range(len(current_joint_positions))]
        else:
            joint_velocities = [0.0] * len(current_joint_positions)
        
        # Update previous values
        self.__prev_joint_positions = current_joint_positions
        self.__prev_time = current_time
        
        # Create and publish JointState message
        joint_msg = JointState()
        joint_msg.header.stamp = self.__node.get_clock().now().to_msg()
        joint_msg.name = self.__joint_names
        joint_msg.position = current_joint_positions
        joint_msg.velocity = joint_velocities
        # joint_msg.effort = self.last_sync_timestamp * np.ones(len(current_joint_positions))
        joint_msg.effort = [self.last_sync_timestamp, self.__frames_elapsed, 1.0 if self.stuck_flag else 0.0] 
        self.__joint_state_publisher.publish(joint_msg)
        

    def publish_gripper_joint_states(self):
        current_time = self.__robot.getTime()
        current_joint_positions = [sensor.getValue() for sensor in self.__gripper_sensors]
        
        # Compute joint velocities
        joint_velocities = []
        if self.__prev_gripper_time is not None and current_time != self.__prev_gripper_time:
            dt = current_time - self.__prev_gripper_time
            joint_velocities = [(current_joint_positions[i] - self.__prev_gripper_joint_positions[i]) / dt 
                                for i in range(len(current_joint_positions))]
        else:
            joint_velocities = [0.0] * len(current_joint_positions)
        
        # Update previous values
        self.__prev_gripper_joint_positions = current_joint_positions
        self.__prev_gripper_time = current_time
        
        # Create and publish JointState message
        joint_msg = JointState()
        joint_msg.header.stamp = self.__node.get_clock().now().to_msg()
        joint_msg.name = self.__joint_names[:3]
        joint_msg.position = current_joint_positions
        joint_msg.velocity = joint_velocities
        self.gripper_joint_state_publisher.publish(joint_msg)

    def publish_gripper_pose(self):
        # Retrieve position and orientation from the box
        pos_gripper = self.__gripper.getPosition()
        orientation_gripper = self.__gripper.getOrientation()

        # Convert the 9-element list to a 3x3 rotation matrix
        rot_matrix = np.array(orientation_gripper).reshape(3, 3)

        # Create a 4x4 identity matrix and insert the 3x3 rotation matrix
        rot_4x4 = np.eye(4)
        rot_4x4[:3, :3] = rot_matrix

        # Convert the 4x4 matrix to a quaternion
        quat = tr.quaternion_from_matrix(rot_4x4)

        # Create Position and Orientation messages
        pos_msg = Point()
        pos_msg.x = pos_gripper[0]
        pos_msg.y = pos_gripper[1]
        pos_msg.z = pos_gripper[2]

        orient_msg = Quaternion()
        orient_msg.x = quat[0]
        orient_msg.y = quat[1]
        orient_msg.z = quat[2]
        orient_msg.w = quat[3]

        # Publish the Pose message
        self.__gripper_publisher.publish(Pose(position=pos_msg, orientation=orient_msg))

    def publish_link7_point(self):
        pos = self.__link7_node.getPosition()
        pos_msg = Point()
        pos_msg.x = pos[0]
        pos_msg.y = pos[1]
        pos_msg.z = pos[2]

        self.__link7_publisher.publish(pos_msg)

    def publish_box_pose(self):
        # Retrieve position and orientation from the box
        position = self.box_node.getPosition()
        orientation = self.box_node.getOrientation()  # 9-element rotation matrix (row-major)

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
        self.box_pub.publish(Pose(position=pos_msg, orientation=orient_msg))

    def publish_target_random_pos(self):
        position = self.target_random_box
        pos_msg = Point()
        pos_msg.x = position[0]
        pos_msg.y = position[1]
        pos_msg.z = position[2]

        # Publish the Pose message
        self.target_random_box_pub.publish(pos_msg)

    def box_pos_viz_callback(self, msg):
        """
        Callback to update the pinch position visualizer based on the box position.
        """
        self.__box_pose_viz = np.array([msg.position.x, msg.position.y, msg.position.z, msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w])
        self.__pinch_visualizer_translation_field.setSFVec3f(list(self.__box_pose_viz[:3]))

    def step(self):
        """
        Perform a simulation step.
        """
        # Due to Webot execute the step simulation, all actual happends after that step. Thus we publish directly in the next iteration t+1 for the observation at time t.
        self.publish_gripper_pose()
        self.publish_gripper_joint_states()
        self.publish_joint_states()
        self.publish_link7_point()
        # self.publish_box_pose()
        # self.publish_target_random_pos()
        
        rclpy.spin_once(self.__node, timeout_sec=0.0)

        # Webots API expects a list of floats, not a numpy array
        # if self.target_box_pos is not None:
        #     self.__pinch_visualizer_translation_field.setSFVec3f(list(self.target_box_pos))
        #self.__pinch_visualizer_translation_field.setSFVec3f(list(self.target_random_box))
    
        # while self.waiting_after_reset and not self.last_sync_timestamp == 0:
        #     rclpy.spin_once(self.__node, timeout_sec=0)
        # self.waiting_after_reset = False
        
        if self.__reset_in_progress_world:
            if  not self.world_reload:
                self.__robot.worldReload()
                self.world_reload = True
            if (self.__node.get_clock().now().nanoseconds - self.last_time_reset_callback) < 10:
                self.__node.get_logger().info("Reload World")
            else:
                box_pose = self.random_box_pose()  # Generate a new box pose
                #self.target_random_box_previous = self.target_random_box.copy()  # Store the previous target box position
                self.target_random_box = self.random_box_target()  # Generate a new target box position

                # self.box_node.getField("translation").setSFVec3f(list(self.__box_pose_viz[:3]))
                # self.box_node.getField("rotation").setSFRotation(quaternion_to_axis_angle(self.__box_pose_viz[3:]))  # Update the box position in Webots
                self.box_node.getField("translation").setSFVec3f(list(box_pose[:3]))
                self.box_node.getField("rotation").setSFRotation(quaternion_to_axis_angle(box_pose[3:]))  # Update the box position in Webots
                self.set_box_pose = False

                # Publish reset completion flag
                reset_done_msg = Bool()
                reset_done_msg.data = True
                self.reset_publisher.publish(reset_done_msg)

                self.last_sync_timestamp = -1
                self.waiting_after_reset = True
                self.reset_box = False
                self.__target_reset_attemp = 0
                self.__reset_in_progress = False
                self.world_reload = False
                self.__reset_in_progress_world = False

        else:
            # Reset velocities to zero only if not in reset
            if not self.__reset_in_progress:
                # if (self.__node.get_clock().now().nanoseconds - self.last_time_arm_callback)/1e9  > 0.032:
                #     for i in range(len(self.__joint_motors)):
                #         self.__joint_motors[i].setPosition(float('inf'))
                #         self.__joint_motors[i].setVelocity(0.0)
                #         if i < 3:                                     # For 3 finger gripper
                #             self.__gripper_motors[i].setVelocity(0.0)
                # if self.__frames_to_run == self.__frames_elapsed:           # let execute at least one frame , then check if the action is done
                #     for i in range(len(self.__joint_sensors)):
                #         self.__joint_motors[i].setVelocity(0)
                self.__frames_elapsed += 1                                  # Increment the elapsed frame count

                self.stuck_flag = False
                for i in range(len(self.__joint_sensors)):
                    current_pos = self.__joint_sensors[i].getValue()
                    joint_name = self.__joint_names[i]
                    min_limit, max_limit = self.__joint_limits[joint_name]
                    buffer = MIN_STATIC_BUFFER_
                    if (current_pos <= min_limit+buffer and self.current_joint_vel[i] < 0) or \
                        (current_pos >= max_limit-buffer and self.current_joint_vel[i] > 0):
                            self.stuck_flag = True
                            break
                if self.stuck_flag:# or np.random.rand() <= 0.05:
                    for i in range(len(self.__joint_sensors)):
                        self.__joint_motors[i].setVelocity(0)
                    self.__frames_elapsed = self.__frames_to_run

            else:
                tolerance = 0.05  # Position tolerance in radians
                all_reached = True 
                self.__target_reset_attemp +=1
                for i in range(len(self.__joint_motors)):
                    current_pos = self.__joint_sensors[i].getValue()
                    target_pos = self.__target_positions[i]
                    error = target_pos - current_pos
                    if abs(error) > tolerance:
                        all_reached = False
                        max_vel = self.__max_velocities[self.__joint_names[i]]
                        self.__joint_motors[i].setPosition(target_pos)
                        self.__joint_motors[i].setVelocity(max_vel)
                    else:
                        self.__joint_motors[i].setPosition(float('inf')) 
                        self.__joint_motors[i].setVelocity(0.0)
                    
                    pos_gripper = self.__gripper.getPosition()
                if all_reached or (self.__target_reset_attemp > 30 and pos_gripper[2]>0.04):  # If all joints reached their target or too many attempts
                    for i in range(len(self.__joint_motors)):
                        self.__joint_motors[i].setPosition(float('inf')) 
                        self.__joint_motors[i].setVelocity(0.0)
                    # Box reset
                    if self.set_box_pose and False:
                        box_pose = self.random_box_pose()  # Generate a new box pose
                        #self.target_random_box_previous = self.target_random_box.copy()  # Store the previous target box position
                        self.target_random_box = self.random_box_target()  # Generate a new target box position

                        # self.box_node.getField("translation").setSFVec3f(list(self.__box_pose_viz[:3]))
                        # self.box_node.getField("rotation").setSFRotation(quaternion_to_axis_angle(self.__box_pose_viz[3:]))  # Update the box position in Webots
                        self.box_node.getField("translation").setSFVec3f(list(box_pose[:3]))
                        self.box_node.getField("rotation").setSFRotation(quaternion_to_axis_angle(box_pose[3:]))  # Update the box position in Webots
                        self.set_box_pose = False
                    
                    # Publish reset completion flag
                    reset_done_msg = Bool()
                    reset_done_msg.data = True
                    self.reset_publisher.publish(reset_done_msg)

                    self.last_sync_timestamp = -1
                    self.waiting_after_reset = True
                    self.reset_box = False
                    self.__target_reset_attemp = 0
                    self.__reset_in_progress = False
                    # while not self.reset_ack:
                    #     rclpy.spin_once(self.__node, timeout_sec=0.0)
                    #     self.reset_publisher.publish(reset_done_msg)
                    # self.reset_ack = False

        # Enable roboter to move without stop
        #self.__robot.step(self.__timestep)  # Step the Webots simulation, if this not execute the webot will not updated. However, this will call the webot step and webot step calls all plugins function

       

       
        