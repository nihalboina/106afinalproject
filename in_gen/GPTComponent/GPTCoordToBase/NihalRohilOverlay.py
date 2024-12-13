import cv2
import numpy as np
import json
import matplotlib.pyplot as plt

import rospy
import rospkg
import roslaunch

from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import cv2
import numpy as np
import tf2_ros
# Importing the CameraTransform class
# from claude_attempt import CameraTransform
import tf
from tf.transformations import quaternion_matrix


# Load the black-and-white LEGO base plate image
image_path = "/Users/rohilkhare/106afinalproject/in_gen/sawyer/CV/src/debug/1733993787.9014525.png"
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)




def quaternion_to_rotation_matrix(quaternion):
    """
    Convert a quaternion into a rotation matrix.

    Args:
        quaternion (geometry_msgs.msg.Quaternion): Quaternion representing rotation.

    Returns:
        np.ndarray: 4x4 rotation matrix.
    """
    q = [
        quaternion.x,
        quaternion.y,
        quaternion.z,
        quaternion.w
    ]
    rot_matrix = quaternion_matrix(q)  # Returns a 4x4 matrix
    return rot_matrix[:3, :3]  # Extract the 3x3 rotation matrix


def tuck():
    """
    Tuck the robot arm to the start position. Use with caution
    """
    if input('Would you like to tuck the arm? (y/n): ') == 'y':
        rospack = rospkg.RosPack()
        path = rospack.get_path('sawyer_full_stack')
        launch_path = path + '/launch/custom_sawyer_tuck.launch'
        uuid = roslaunch.rlutil.get_or_generate_uuid(None, False)
        roslaunch.configure_logging(uuid)
        launch = roslaunch.parent.ROSLaunchParent(uuid, [launch_path])
        launch.start()
    else:
        print('Canceled. Not tucking the arm.')


def subscribe_once(topic_name, message_type):
    """
    Subscribe to a topic and receive a message only once.

    Args:
        topic_name (str): The name of the topic to subscribe to.
        message_type (type): The ROS message type of the topic.

    Returns:
        message: The message received from the topic.
    """
    try:
        rospy.loginfo(f"Waiting for a single message on topic: {topic_name}")
        message = rospy.wait_for_message(topic_name, message_type)
        return message
    except rospy.ROSException as e:
        rospy.logerr(f"Error while waiting for message: {e}")
        return None


def run_cv(image):
    # Camera matrix (updated with provided intrinsics)
    K = np.array([
        [627.794983, 0.0, 360.174988],
        [0.0, 626.838013, 231.660996],
        [0.0, 0.0, 1.0]
    ])

    # Distortion coefficients
    D = np.array([-0.438799, 0.257299, 0.001038, 0.000384, -0.105028])

    # Get the optimal new camera matrix
    h, w = image.shape[:2]
    new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(K, D, (w, h), 1, (w, h))

    # Undistort the image
    undistorted_image = cv2.undistort(image, K, D, None, new_camera_matrix)

    # Crop the image if needed
    x, y, w, h = roi
    undistorted_image = undistorted_image[y:y + h, x:x + w]

    # Apply GaussianBlur to reduce noise
    blurred = cv2.GaussianBlur(undistorted_image, (5, 5), 0)

    # Apply Sobel Edge Detection (X and Y)
    sobel_x = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
    sobel_combined = cv2.convertScaleAbs(sobel_x) + cv2.convertScaleAbs(sobel_y)

    # Use binary thresholding on Sobel edges
    _, thresh = cv2.threshold(sobel_combined, 100, 255, cv2.THRESH_BINARY)

    # Find contours from the Sobel edge-detected image
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    output_image = cv2.cvtColor(undistorted_image, cv2.COLOR_GRAY2BGR)
    midpoint_x, midpoint_y, angle = None, None, None
    box = None
    largest_square_area = 0

    generated_json = {"u_v_points": []}

    if contours:
        for contour in contours:
            # Get the rotated bounding box
            rotated_rect = cv2.minAreaRect(contour)
            width, height = rotated_rect[1]

            # Ensure it is approximately a square
            if abs(width - height) <= 10 and width > 0 and height > 0:  # Allow some tolerance
                area = width * height
                if area > largest_square_area:
                    largest_square_area = area
                    box = cv2.boxPoints(rotated_rect)
                    box = np.int0(box)
                    angle = rotated_rect[-1]  # Get the angle of rotation

                    # Calculate the midpoint of the rotated bounding box
                    midpoint_x = int((box[0][0] + box[2][0]) / 2)
                    midpoint_y = int((box[0][1] + box[2][1]) / 2)

        if box is not None:
            # Draw the rotated bounding box and midpoint on the undistorted image
            cv2.drawContours(output_image, [box], 0, (0, 255, 0), 2)
            cv2.circle(output_image, (midpoint_x, midpoint_y), 5, (0, 0, 255), -1)

            # Adjust the angle
            if angle < -45:
                angle += 90
            print(f"Corrected Computed Rotation Angle: {angle:.2f} degrees")

    # Save generated JSON
    generated_json_path = "/home/cc/ee106a/fa24/class/ee106a-aah/final_project/106afinalproject/in_gen/GPTComponent/GPTCoordToBase/Generated_UV_Points.json"
    with open(generated_json_path, 'w') as f:
        json.dump(generated_json, f, indent=4)

    # Display the final result
    cv2.imshow("Undistorted Image with Overlay", output_image)
    cv2.waitKey(1)


def main():
    right_hand_camera_topic = "/io/internal_camera/right_hand_camera/image_raw"

    rospy.init_node('blocks_publisher', anonymous=True)
    # pub = rospy.Publisher('/blocks', Blocks, queue_size=10)

    bridge = CvBridge()
    publish_blocks = []

    # Initialize TF buffer and listener
    tf_buffer = tf2_ros.Buffer()

    rospy.sleep(1.0)  # Give TF some time to fill

    # Instantiate CameraTransform
    # camera_transform = CameraTransform()

    # Get initial camera pose
    # camera_pose = get_camera_pose(tf_buffer)
    # if not camera_pose.header.frame_id:
    #     rospy.logerr("Initial camera pose is invalid. Exiting.")
    #     return

    rate = rospy.Rate(1)  # 1 Hz

    while not rospy.is_shutdown():
        # Subscribe to new image
        image_msg = subscribe_once(right_hand_camera_topic, Image)
        if image_msg is None:
            rospy.logerr("Failed to get new image.")
            continue

        # Update camera pose
        # camera_pose = get_camera_pose(tf_buffer)

        # if not camera_pose.header.frame_id:
        #     rospy.logerr("Camera pose is invalid.")
        #     continue

        # Detect objects and get real-base coordinates

        blocks_image = bridge.imgmsg_to_cv2(
            image_msg, desired_encoding='mono8')
        run_cv(
            blocks_image)

        # Draw detected blocks on the image
        try:
            image = bridge.imgmsg_to_cv2(image_msg, desired_encoding='bgr8')
            cv2.imshow("Raw", image)
            image = draw_blocks(image, publish_blocks)
            cv2.imshow("Detected Blocks", image)
            cv2.waitKey(1)  # Non-blocking
        except CvBridgeError as e:
            rospy.logerr(f"CV Bridge Error: {e}")

        # Create the Blocks message
        blocks_msg = {}
        blocks_msg['blocks'] = publish_blocks
        blocks_msg['time_updated'] = rospy.Time.now().to_nsec()

        # Log the message
        rospy.loginfo(f"Publishing blocks: {blocks_msg}")

        # Publish the message (uncomment and modify as per your message type)
        # pub.publish(blocks_msg)
        print(f"To publish: {blocks_msg}")

        # Sleep for the remainder of the loop
        rate.sleep()


if __name__ == '__main__':
    try:
        tuck()
        rospy.sleep(5.0)
        main()
    except rospy.ROSInterruptException:
        pass
