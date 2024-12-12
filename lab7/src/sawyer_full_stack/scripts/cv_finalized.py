#!/usr/bin/env python

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
from claude_attempt import CameraTransform
import tf
from tf.transformations import quaternion_matrix


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


def detect_objects(image, n=1):
    """
    Detect up to n objects in the image using edge detection and contour analysis.

    Args:
        image (numpy.ndarray): Grayscale image.
        n (int): Number of objects to detect.

    Returns:
        list of tuples: List containing (cX, cY) for each detected object.
    """
    # Apply Gaussian Blur to reduce noise
    blurred = cv2.GaussianBlur(image, (5, 5), 0)

    # Edge Detection
    edges = cv2.Canny(blurred, 50, 150)

    # Find contours
    contours, _ = cv2.findContours(
        edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter contours by area and sort by area descending
    filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 500]
    filtered_contours = sorted(
        filtered_contours, key=cv2.contourArea, reverse=True)

    detected_objects = []

    for cnt in filtered_contours[:n]:
        M = cv2.moments(cnt)
        if M["m00"] == 0:
            continue
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        detected_objects.append((cX, cY))

    return detected_objects


def run_cv(image_msg, camera_transform, camera_pose, max_objects=2):
    """
    Detect objects in the image and compute their real-base coordinates.

    Args:
        image_msg (Image): ROS Image message with the current frame.
        camera_transform (CameraTransform): Instance of CameraTransform class.
        camera_pose (PoseStamped): Current pose of the camera.
        max_objects (int): Maximum number of objects to detect.

    Returns:
        list of dict: Detected objects with their positions and classifications.
    """
    bridge = CvBridge()
    try:
        blocks_image = bridge.imgmsg_to_cv2(
            image_msg, desired_encoding='mono8')
    except CvBridgeError as e:
        rospy.logerr(f"CV Bridge Error: {e}")
        return []

    # Detect objects using edge detection
    detected_centroids = detect_objects(blocks_image, n=max_objects)

    detected_blocks = []

    # Extract camera position from camera_pose
    camera_position = np.array([
        camera_pose.pose.position.x,
        camera_pose.pose.position.y,
        camera_pose.pose.position.z
    ])

    camera_rotation = camera_pose.pose.orientation
    print(f"camera rotation: {camera_rotation}")

    for (cX, cY) in detected_centroids:
        try:
            # Convert pixel to base coordinates using CameraTransform
            real_x, real_y, real_z = camera_transform.pixel_to_base(
                cX, cY)
            print(f"Real base coordinates: {real_x}, {real_y}, {real_z}")

            # Create Block message
            block = {
                "camera_coordinates": {
                    "x": cX,
                    "y": cY
                },
                "pose": {
                    "position": {
                        "x": real_x,
                        "y": real_y,
                        "z": real_z
                    },
                    "orientation": {
                        "x": 0,
                        "y": 0,
                        "z": 0,
                        "w": 1  # Neutral orientation
                    }
                },
                "classification": "block.stl",  # Placeholder classification
                "confidence": 100.0  # Initial confidence
            }

            detected_blocks.append(block)
        except ValueError as ve:
            rospy.logerr(f"Coordinate Transformation Error: {ve}")
            continue
        except Exception as e:
            rospy.logerr(
                f"Unexpected error during coordinate transformation: {e}")
            continue

    rospy.loginfo(f"Detected {len(detected_blocks)} blocks in current frame.")

    # Save picture of frame with detected blocks
    detected_blocks_image = blocks_image.copy()
    for (cX, cY) in detected_centroids:
        cv2.circle(detected_blocks_image, (cX, cY), 20, (255, 0, 0), 3)
        cv2.putText(detected_blocks_image, f"({cX}, {cY})", (cX + 10, cY),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    cv2.imwrite("detected_blocks_image.jpg", detected_blocks_image)

    return detected_blocks


def get_camera_pose(tf_buffer):
    """
    Get the current pose of the camera in the base frame.

    Args:
        tf_buffer (tf2_ros.Buffer): TF buffer to lookup transforms.

    Returns:
        PoseStamped: Pose of the camera in the base frame.
    """
    try:
        # Lookup the latest available transform
        transform = tf_buffer.lookup_transform(
            'base', 'right_hand_camera', rospy.Time(0), rospy.Duration(1.0))
        pose = PoseStamped()
        pose.header = transform.header
        pose.pose.position.x = transform.transform.translation.x
        pose.pose.position.y = transform.transform.translation.y
        pose.pose.position.z = transform.transform.translation.z
        pose.pose.orientation = transform.transform.rotation
        print(f"Camera pose: {pose}")
        return pose
    except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
        rospy.logerr(f"Failed to get camera pose: {e}")
        return PoseStamped()  # Return an empty PoseStamped

# Given a list of blocks, draw circular markers on the image


def draw_blocks(image, blocks):
    for block in blocks:
        cur_idx = blocks.index(block)
        x = block['camera_coordinates']['x']
        y = block['camera_coordinates']['y']
        cv2.circle(image, (x, y), 20, (0, 0, 255), 2)

        font = cv2.FONT_HERSHEY_SIMPLEX
        text = f"Camera Position: ({block['camera_coordinates']['x']}, {block['camera_coordinates']['y']})"
        cv2.putText(image, text, (10, 30 + 30 * cur_idx),
                    font, 0.7, (0, 0, 255), 2)
        text = f"Predicted Position: ({block['pose']['position']['x']:.2f}, {block['pose']['position']['y']:.2f}, {block['pose']['position']['z']:.2f})"
        cv2.putText(image, text, (10, 60 + 30 * cur_idx),
                    font, 0.7, (0, 0, 255), 2)

    return image


def main():
    right_hand_camera_topic = "/io/internal_camera/right_hand_camera/image_raw"

    rospy.init_node('blocks_publisher', anonymous=True)
    # pub = rospy.Publisher('/blocks', Blocks, queue_size=10)

    bridge = CvBridge()
    publish_blocks = []

    # Initialize TF buffer and listener
    tf_buffer = tf2_ros.Buffer()
    tf_listener = tf2_ros.TransformListener(tf_buffer)

    rospy.sleep(1.0)  # Give TF some time to fill

    # Instantiate CameraTransform
    camera_transform = CameraTransform()

    # Get initial camera pose
    camera_pose = get_camera_pose(tf_buffer)
    if not camera_pose.header.frame_id:
        rospy.logerr("Initial camera pose is invalid. Exiting.")
        return

    rate = rospy.Rate(1)  # 1 Hz

    while not rospy.is_shutdown():
        # Subscribe to new image
        image_msg = subscribe_once(right_hand_camera_topic, Image)
        if image_msg is None:
            rospy.logerr("Failed to get new image.")
            continue

        # Update camera pose
        camera_pose = get_camera_pose(tf_buffer)

        if not camera_pose.header.frame_id:
            rospy.logerr("Camera pose is invalid.")
            continue

        # Detect objects and get real-base coordinates
        publish_blocks = run_cv(
            image_msg, camera_transform, camera_pose, max_objects=2)

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
