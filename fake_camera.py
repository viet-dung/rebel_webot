import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Pose, Point, Quaternion
import numpy as np

# This is the topic your BoxSpawner is already listening to.
CAMERA_TOPIC = '/azure_kinect/object' 

# This is a new topic we create to control this fake camera node.
CONTROL_TOPIC = '/fake_camera/set_pose'

class FakeCameraPublisher(Node):
    """
    A node that simulates a camera detecting an object.

    It publishes a Pose message to a specified topic. The published pose
    can be controlled by sending a new position to the CONTROL_TOPIC.
    This node intentionally adds a fixed offset to the ground truth position
    to simulate a real, uncalibrated camera, which allows for testing
    the calibration logic of a subscriber node.
    """

    def __init__(self):
        super().__init__('fake_camera_publisher')

        # 1. Publisher that mimics the real camera
        self.publisher_ = self.create_publisher(Pose, CAMERA_TOPIC, 10)

        # 2. Subscriber to receive commands to change the object's position
        self.subscription = self.create_subscription(
            Point, # We only need to set the position for this test
            CONTROL_TOPIC,
            self.control_callback,
            10)

        # 3. The "ground truth" position of the object we are simulating.
        #    Initialize it to a default starting location.
        self.true_position = np.array([0.3, 0.0, 0.0275]) # A reasonable starting point

        # 4. The FAKE ERROR OFFSET of our simulated camera.
        #    This is the error that your BoxSpawner's calibration should correct.
        #    Let's make it a noticeable, non-trivial offset.
        self.camera_error_offset = np.array([-0.04, 0.06, 0.01]) # 4cm error in X, 6cm in Y

        # 5. Timer to continuously publish the camera data
        timer_period = 0.03  # Publish at ~33 Hz
        self.timer = self.create_timer(timer_period, self.publish_pose)

        self.get_logger().info('================================================')
        self.get_logger().info('Fake Camera Publisher Node is RUNNING.')
        self.get_logger().info(f'-> Publishing to: {CAMERA_TOPIC}')
        self.get_logger().info(f'-> Listening for commands on: {CONTROL_TOPIC}')
        self.get_logger().info(f'-> Simulated Camera Error: {self.camera_error_offset}')
        self.get_logger().info('================================================')

    def control_callback(self, msg: Point):
        """
        Callback to update the true position of the simulated object.
        """
        self.true_position = np.array([msg.x, msg.y, msg.z])
        self.get_logger().info(f'Received new command. Setting true object position to: {self.true_position}')

    def publish_pose(self):
        """
        Calculates the "measured" pose by adding the error offset
        and publishes it.
        """
        # Calculate the position the camera *thinks* it sees
        measured_position = self.true_position + self.camera_error_offset

        # Create the Pose message
        pose_msg = Pose()
        
        # Set the position part of the message
        pose_msg.position.x = measured_position[0]
        pose_msg.position.y = measured_position[1]
        pose_msg.position.z = measured_position[2]
        
        # Set a default orientation (identity quaternion)
        pose_msg.orientation.x = 0.0
        pose_msg.orientation.y = 0.0
        pose_msg.orientation.z = 0.0
        pose_msg.orientation.w = 1.0

        # Publish the message
        self.publisher_.publish(pose_msg)


def main(args=None):
    rclpy.init(args=args)
    node = FakeCameraPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()