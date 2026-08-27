#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray
import numpy as np
from sensor_msgs.msg import JointState
import os
import threading
import math

class RealRoboticArm(Node):
    def __init__(self, node_name='real_robotic_arm_controller'):
        super().__init__(node_name)

        self.declare_parameter('env_id', 0)
        
        self.__namespace = self.get_namespace()
        self.get_logger().info(f"Namespace: {self.__namespace}")

        self.__env_id = self.get_parameter('env_id').get_parameter_value().integer_value
        os.environ['ROS_DOMAIN_ID'] = str(self.__env_id)

        self.custom_joint_state_updated = False
        self.custom_joint_time = 0

        self.custom_joint_positions = np.zeros(6)
        self.custom_joint_velocities = np.zeros(6)
        self.custom_last_sync_timestamp = -1

        self.real_joint_positions = np.zeros(6)

        self._state_lock = threading.Lock()
        self._max_velocity = math.pi/4

        self._control_loop_hz = 50
        
        self._velocity_publisher = self.create_publisher(Float64MultiArray, 'velocity_controllers/commands', 10)

        self.create_subscription(JointState, 'custom_joint_states', self.webot_joint_states_callback, 1)
        self.create_subscription(JointState, 'joint_states', self.real_joint_states_callback, 1)
        self._control_timer = self.create_timer(1.0 / self._control_loop_hz, self._control_loop)

        self.get_logger().info(f"Control loop started at {self._control_loop_hz} Hz.")

    def _process_joint_states(self, msg: JointState, is_webot:bool):
        desired_joint_names = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]
        ordered_positions = [0.0] * len(desired_joint_names)
        ordered_velocities = [0.0] * len(desired_joint_names)

        name_to_index = {name: index for index, name in enumerate(msg.name)}

        for i, desired_name in enumerate(desired_joint_names):
            if desired_name in name_to_index:
                index_in_msg = name_to_index[desired_name]
                
                if desired_name in ['joint1', 'joint4','joint6']:  # Joints that can rotate continuously
                    ordered_positions[i] = ((msg.position[index_in_msg] + np.pi) % (2 * np.pi)) - np.pi
                else:
                    ordered_positions[i] = msg.position[index_in_msg]
                ordered_velocities[i] = msg.velocity[index_in_msg]
        if is_webot:
            self.custom_joint_positions = np.array(np.round(ordered_positions, decimals=3))
            self.custom_joint_velocities = np.array(np.round(ordered_velocities, decimals=3))
            self.custom_last_sync_timestamp = msg.effort[0]
        else:
            self.real_joint_positions = np.array(np.round(ordered_positions, decimals=3))

    def webot_joint_states_callback(self, msg):
        self._process_joint_states(msg=msg, is_webot=True)

    def real_joint_states_callback(self, msg):
        self._process_joint_states(msg=msg, is_webot=False)

    def _control_loop(self):
        with self._state_lock:
            # Get thread-safe copies of the state
            target = self.custom_joint_positions.copy()
            current = self.real_joint_positions.copy()

        # --- Proportional Control Calculation ---
        error = target - current

        # Handle angle wrap-around for error (shortest path)
        for i in range(6):
            error[i] = math.atan2(math.sin(error[i]), math.cos(error[i]))

        # Calculate velocity: velocity = Kp * error
        velocity_command = error

        # Apply velocity limits (saturation)
        velocity_command = np.clip(velocity_command, -self._max_velocity, self._max_velocity)

        # --- Publish Command ---
        cmd_msg = Float64MultiArray()
        cmd_msg.data = velocity_command.tolist()
        self._velocity_publisher.publish(cmd_msg)


def main(args=None):
    # Initialize rclpy ONCE here
    rclpy.init(args=args)

    # Create an instance of the Node class
    real_robotic_arm_node = RealRoboticArm(node_name='real_robotic_arm_controller')

    # Spin the node
    try:
        rclpy.spin(real_robotic_arm_node)
    except KeyboardInterrupt:
        pass # Allow clean shutdown
    except Exception as e:
         real_robotic_arm_node.get_logger().fatal(f"Unhandled exception in spin: {e}")
    finally:
        # --- Clean Shutdown ---
        # Destroy the node explicitly
        # (optional - Done automatically on shutdown)
        real_robotic_arm_node.destroy_node()
        # Shutdown rclpy
        if rclpy.ok():
           rclpy.shutdown()

# This standard boilerplate allows the script to be run directly
if __name__ == '__main__':
    main()


    
    

