#!/usr/bin/env python

import rospy
from CV.msg import Blocks, Block  # Replace 'CV' with your package name
from geometry_msgs.msg import Pose, PoseStamped
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import cv2
import numpy as np
import tf
import tf2_ros
import tf2_geometry_msgs
from collections import defaultdict


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
        rospy.loginfo(f"Received message: {message}")
        return message
    except rospy.ROSException as e:
        rospy.logerr(f"Error while waiting for message: {e}")
        return None


def get_real_world_coordinates(image_x, image_y, camera_calibration, camera_pose, tf_buffer):
    """
    Convert image pixel coordinates to real-world coordinates considering the robot's pose.

    Args:
        image_x (int): X coordinate in the image.
        image_y (int): Y coordinate in the image.
        camera_calibration (dict): Camera calibration parameters.
        camera_pose (PoseStamped): Current pose of the camera.
        tf_buffer (tf2_ros.Buffer): TF buffer to lookup transforms.

    Returns:
        (float, float, float): Real-world coordinates (x, y, z) in the world frame.
    """
    # Extract camera intrinsic parameters
    fx = camera_calibration['fx']
    fy = camera_calibration['fy']
    cx = camera_calibration['cx']
    cy = camera_calibration['cy']

    # Assume a constant depth if not provided
    depth = camera_calibration.get('depth', 1.0)

    # Convert pixel to normalized camera coordinates
    x_norm = (image_x - cx) / fx
    y_norm = (image_y - cy) / fy

    # Point in camera frame (assuming Z = depth)
    point_camera = np.array([x_norm * depth, y_norm * depth, depth, 1.0])

    # Convert camera_pose to a transformation matrix
    quat = [
        camera_pose.pose.orientation.x,
        camera_pose.pose.orientation.y,
        camera_pose.pose.orientation.z,
        camera_pose.pose.orientation.w
    ]
    translation = [
        camera_pose.pose.position.x,
        camera_pose.pose.position.y,
        camera_pose.pose.position.z
    ]
    tf_listener = tf.TransformListener()

    try:
        # Wait for the transform to be available
        tf_listener.waitForTransform(
            "world", camera_pose.header.frame_id, rospy.Time(0), rospy.Duration(1.0))
        # Get the transformation matrix from camera frame to world frame
        (trans, rot) = tf_listener.lookupTransform(
            "world", camera_pose.header.frame_id, rospy.Time(0))
        # Create transformation matrix
        rotation_matrix = tf.transformations.quaternion_matrix(rot)
        translation_matrix = tf.transformations.translation_matrix(trans)
        transform_matrix = np.dot(translation_matrix, rotation_matrix)

        # Transform the point to world frame
        point_world = np.dot(transform_matrix, point_camera)
        real_x, real_y, real_z = point_world[:3]
        return real_x, real_y, real_z
    except (tf.Exception, tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException) as e:
        rospy.logerr(f"TF transformation error: {e}")
        # Fallback to camera frame if transformation fails
        return x_norm * depth, y_norm * depth, depth


def run_cv_first(background_image_msg, blocks_image_msg, camera_calibration, camera_pose, tf_buffer):
    """
    Compare the background and blocks images to detect initial blocks.

    Args:
        background_image_msg (Image): ROS Image message of the background.
        blocks_image_msg (Image): ROS Image message with blocks.
        camera_calibration (dict): Camera calibration parameters.
        camera_pose (PoseStamped): Current pose of the camera.
        tf_buffer (tf2_ros.Buffer): TF buffer to lookup transforms.

    Returns:
        list of Block: Detected blocks with their positions and classifications.
    """
    bridge = CvBridge()
    try:
        background_image = bridge.imgmsg_to_cv2(
            background_image_msg, desired_encoding='mono8')
        blocks_image = bridge.imgmsg_to_cv2(
            blocks_image_msg, desired_encoding='mono8')
    except CvBridgeError as e:
        rospy.logerr(f"CV Bridge Error: {e}")
        return []

    # Background subtraction
    diff = cv2.absdiff(blocks_image, background_image)
    _, thresh = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)

    # Morphological operations to remove noise
    kernel = np.ones((5, 5), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    # Find contours of the blocks
    contours, _ = cv2.findContours(
        thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    detected_blocks = []

    for cnt in contours:
        if cv2.contourArea(cnt) < 500:  # Filter out small contours
            continue
        M = cv2.moments(cnt)
        if M["m00"] == 0:
            continue
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])

        # Get real-world coordinates
        real_x, real_y, real_z = get_real_world_coordinates(
            cX, cY, camera_calibration, camera_pose, tf_buffer)

        # Create Block message
        block = Block()
        block.pose.position.x = real_x
        block.pose.position.y = real_y
        block.pose.position.z = real_z
        block.pose.orientation.x = 0
        block.pose.orientation.y = 0
        block.pose.orientation.z = 0
        block.pose.orientation.w = 1  # Neutral orientation
        block.classification = "block.stl"  # Placeholder classification
        block.confidence = 100.0  # Initial confidence

        detected_blocks.append(block)

    rospy.loginfo(f"Detected {len(detected_blocks)} blocks in initial run.")
    return detected_blocks


def run_cv(blocks_image_msg, background_image_msg, previous_blocks, camera_calibration, camera_pose, tf_buffer):
    """
    Update the blocks seen in the image with the new calculated real-world coordinates
    as well as confidence levels.

    Args:
        blocks_image_msg (Image): ROS Image message with blocks.
        background_image_msg (Image): ROS Image message of the background.
        previous_blocks (list of Block): Previously detected blocks.
        camera_calibration (dict): Camera calibration parameters.
        camera_pose (PoseStamped): Current pose of the camera.
        tf_buffer (tf2_ros.Buffer): TF buffer to lookup transforms.

    Returns:
        list of Block: Updated list of blocks.
    """
    bridge = CvBridge()
    try:
        background_image = bridge.imgmsg_to_cv2(
            background_image_msg, desired_encoding='mono8')
        blocks_image = bridge.imgmsg_to_cv2(
            blocks_image_msg, desired_encoding='mono8')
    except CvBridgeError as e:
        rospy.logerr(f"CV Bridge Error: {e}")
        return previous_blocks

    # Background subtraction
    diff = cv2.absdiff(blocks_image, background_image)
    _, thresh = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)

    # Morphological operations to remove noise
    kernel = np.ones((5, 5), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    # Find contours of the blocks
    contours, _ = cv2.findContours(
        thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    current_detected = []
    for cnt in contours:
        if cv2.contourArea(cnt) < 500:
            continue
        M = cv2.moments(cnt)
        if M["m00"] == 0:
            continue
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])

        # Get real-world coordinates
        real_x, real_y, real_z = get_real_world_coordinates(
            cX, cY, camera_calibration, camera_pose, tf_buffer)

        # Attempt to match with previous blocks
        matched = False
        for prev_block in previous_blocks:
            distance = np.sqrt((real_x - prev_block.pose.position.x)**2 +
                               (real_y - prev_block.pose.position.y)**2 +
                               (real_z - prev_block.pose.position.z)**2)
            if distance < 0.05:  # Threshold for matching (meters)
                matched = True
                # Update confidence
                new_confidence = min(prev_block.confidence + 5, 100.0)
                prev_block.confidence = new_confidence
                # Update position
                prev_block.pose.position.x = real_x
                prev_block.pose.position.y = real_y
                prev_block.pose.position.z = real_z
                current_detected.append(prev_block)
                break

        if not matched:
            # New block detected
            block = Block()
            block.pose.position.x = real_x
            block.pose.position.y = real_y
            block.pose.position.z = real_z
            block.pose.orientation.x = 0
            block.pose.orientation.y = 0
            block.pose.orientation.z = 0
            block.pose.orientation.w = 1
            block.classification = "block.stl"  # Placeholder classification
            block.confidence = 50.0  # Initial confidence for new block
            current_detected.append(block)

    # Optionally, decrease confidence for blocks not detected in this frame
    for prev_block in previous_blocks:
        if prev_block not in current_detected:
            prev_block.confidence -= 10
            if prev_block.confidence > 0:
                current_detected.append(prev_block)

    rospy.loginfo(f"Updated {len(current_detected)} blocks in real-time run.")
    return current_detected


def get_camera_pose(tf_buffer, camera_frame):
    """
    Get the current pose of the camera in the world frame.

    Args:
        tf_buffer (tf2_ros.Buffer): TF buffer to lookup transforms.
        camera_frame (str): Frame ID of the camera.

    Returns:
        PoseStamped: Pose of the camera in the world frame.
    """
    try:
        # Lookup the latest available transform
        transform = tf_buffer.lookup_transform(
            'world', camera_frame, rospy.Time(0), rospy.Duration(1.0))
        pose = PoseStamped()
        pose.header = transform.header
        pose.pose.position.x = transform.transform.translation.x
        pose.pose.position.y = transform.transform.translation.y
        pose.pose.position.z = transform.transform.translation.z
        pose.pose.orientation = transform.transform.rotation
        return pose
    except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
        rospy.logerr(f"Failed to get camera pose: {e}")
        return PoseStamped()  # Return an empty PoseStamped


def main():
    right_hand_camera_topic = "/io/internal_camera/right_hand_camera/image_raw"
    camera_frame = "right_hand_camera_frame"  # Replace with actual camera frame ID
    camera_calibration = {
        'fx': 600,  # Example focal length in pixels
        'fy': 600,
        'cx': 320,  # Example principal point
        'cy': 240,
        # Assume blocks are 0.5 meters away (can be adjusted or estimated)
        'depth': 0.5
    }

    rospy.init_node('blocks_publisher', anonymous=True)
    pub = rospy.Publisher('/blocks', Blocks, queue_size=10)

    bridge = CvBridge()
    publish_blocks = []

    # Initialize TF buffer and listener
    tf_buffer = tf2_ros.Buffer()
    tf_listener = tf2_ros.TransformListener(tf_buffer)

    rospy.sleep(1.0)  # Give TF some time to fill

    input("Press Enter once background is clear...")
    background_image = subscribe_once(right_hand_camera_topic, Image)
    if background_image is None:
        rospy.logerr("Failed to get background image. Exiting.")
        return

    input("Press Enter once blocks are placed...")
    blocks_image = subscribe_once(right_hand_camera_topic, Image)
    if blocks_image is None:
        rospy.logerr("Failed to get blocks image. Exiting.")
        return

    # Get initial camera pose
    camera_pose = get_camera_pose(tf_buffer, camera_frame)
    if not camera_pose.header.frame_id:
        rospy.logerr("Camera pose is invalid. Exiting.")
        return

    # Initial block detection
    publish_blocks = run_cv_first(
        background_image, blocks_image, camera_calibration, camera_pose, tf_buffer)

    rate = rospy.Rate(1)  # 1 Hz

    while not rospy.is_shutdown():
        # Create the Blocks message
        blocks_msg = Blocks()
        blocks_msg.blocks = publish_blocks
        blocks_msg.time_updated = rospy.Time.now().to_nsec()

        # Log the message
        rospy.loginfo(f"Publishing blocks: {blocks_msg}")

        # Publish the message
        pub.publish(blocks_msg)

        # Sleep for the remainder of the loop
        rate.sleep()

        # Subscribe to new image
        blocks_image = subscribe_once(right_hand_camera_topic, Image)
        if blocks_image is None:
            rospy.logerr("Failed to get new blocks image.")
            continue

        # Update camera pose
        camera_pose = get_camera_pose(tf_buffer, camera_frame)
        if not camera_pose.header.frame_id:
            rospy.logerr("Camera pose is invalid.")
            continue

        # Update blocks with real-time CV
        publish_blocks = run_cv(
            blocks_image, background_image, publish_blocks, camera_calibration, camera_pose, tf_buffer)


if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass
