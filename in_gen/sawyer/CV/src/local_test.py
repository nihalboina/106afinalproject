#!/usr/bin/env python

import rospy
from geometry_msgs.msg import Pose, PoseStamped
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import cv2
import numpy as np
import tf
import tf2_ros
import tf2_geometry_msgs
from collections import defaultdict
from tf.transformations import quaternion_matrix, translation_matrix

# Removed ROS image subscription and added OpenCV VideoCapture


def get_local_camera_frames():
    """
    Capture frames from the local camera.

    Returns:
        tuple: (background_image, blocks_image)
    """
    cap = cv2.VideoCapture(0)  # 0 is typically the built-in camera

    if not cap.isOpened():
        rospy.logerr("Cannot open camera")
        return None, None

    bridge = CvBridge()

    rospy.loginfo("Press 'b' to capture the background image.")
    background_image = None
    blocks_image = None

    while True:
        ret, frame = cap.read()
        if not ret:
            rospy.logerr("Failed to grab frame")
            break

        # Display the live feed
        cv2.imshow(
            'Live Camera Feed - Press "b" for background, "s" for blocks, "q" to quit', frame)

        key = cv2.waitKey(1)
        if key & 0xFF == ord('b'):
            # Capture background image
            background_image = bridge.cv2_to_imgmsg(
                cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), "mono8")
            rospy.loginfo("Background image captured.")
        elif key & 0xFF == ord('s'):
            # Capture blocks image
            blocks_image = bridge.cv2_to_imgmsg(
                cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), "mono8")
            rospy.loginfo("Blocks image captured.")
        elif key & 0xFF == ord('q'):
            break

        if background_image is not None and blocks_image is not None:
            break

    cap.release()
    cv2.destroyAllWindows()

    return background_image, blocks_image


def get_real_world_coordinates(image_x, image_y, camera_calibration, camera_pose, tf_buffer):
    """
    Convert image pixel coordinates to real-world coordinates considering the camera's pose.

    Args:
        image_x (int): X coordinate in the image.
        image_y (int): Y coordinate in the image.
        camera_calibration (dict): Camera calibration parameters.
        camera_pose (PoseStamped): Current pose of the camera.
        tf_buffer (tf2_ros.Buffer): TF2 buffer to lookup transforms.

    Returns:
        (float, float, float): Real-world coordinates (x, y, z) in the world frame.
    """
    try:
        # Extract camera intrinsic parameters
        fx = camera_calibration['fx']
        fy = camera_calibration['fy']
        cx = camera_calibration['cx']
        cy = camera_calibration['cy']

        # Obtain depth from camera_calibration or another reliable source
        # Here, we assume that depth is part of camera_calibration
        # Default to 0.5 meters if not specified
        depth = camera_calibration.get('depth', 0.5)
        if depth <= 0:
            rospy.logerr(
                f"Invalid depth value: {depth}. Depth must be positive.")
            return None

        # Convert pixel to normalized camera coordinates
        x_norm = (image_x - cx) / fx
        y_norm = (image_y - cy) / fy

        # Point in camera frame (assuming Z = depth)
        point_camera = np.array(
            [x_norm * depth, y_norm * depth, depth, 1.0]).reshape(4, 1)

        # Convert PoseStamped to transformation matrix
        # Assuming camera_pose is already in the world frame
        rot_matrix = quaternion_matrix([
            camera_pose.pose.orientation.x,
            camera_pose.pose.orientation.y,
            camera_pose.pose.orientation.z,
            camera_pose.pose.orientation.w
        ])  # 4x4 matrix
        trans_matrix = translation_matrix([
            camera_pose.pose.position.x,
            camera_pose.pose.position.y,
            camera_pose.pose.position.z
        ])  # 4x4 matrix

        # Combine rotation and translation to form the full transformation matrix
        transform_matrix = np.dot(trans_matrix, rot_matrix)  # 4x4 matrix

        # Transform the point from camera frame to world frame
        point_world_homogeneous = np.dot(transform_matrix, point_camera)
        real_x, real_y, real_z, _ = point_world_homogeneous.flatten()

        return real_x, real_y, real_z

    except tf2_ros.LookupException as e:
        rospy.logerr(f"TF2 LookupException: {e}")
    except tf2_ros.ExtrapolationException as e:
        rospy.logerr(f"TF2 ExtrapolationException: {e}")
    except tf2_ros.ConnectivityException as e:
        rospy.logerr(f"TF2 ConnectivityException: {e}")
    except KeyError as e:
        rospy.logerr(f"Camera calibration parameter missing: {e}")
    except Exception as e:
        rospy.logerr(f"Unexpected error in get_real_world_coordinates: {e}")

    # Return None if any exception occurs
    return None


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
        list of dict: Detected blocks with their positions and classifications.
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
        rospy.loginfo(f"Identified camera coordinates: {cX}, {cY}")
        real_coords = get_real_world_coordinates(
            cX, cY, camera_calibration, camera_pose, tf_buffer)
        if real_coords is None:
            continue
        real_x, real_y, real_z = real_coords
        rospy.loginfo(f"Real world coordinates: {real_x}, {real_y}, {real_z}")

        # Create Block dictionary
        block = {
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

    rospy.loginfo(f"Detected {len(detected_blocks)} blocks in initial run.")
    return detected_blocks


def run_cv(blocks_image_msg, background_image_msg, previous_blocks, camera_calibration, camera_pose, tf_buffer):
    """
    Update the blocks seen in the image with the new calculated real-world coordinates
    as well as confidence levels.

    Args:
        blocks_image_msg (Image): ROS Image message with blocks.
        background_image_msg (Image): ROS Image message of the background.
        previous_blocks (list of dict): Previously detected blocks.
        camera_calibration (dict): Camera calibration parameters.
        camera_pose (PoseStamped): Current pose of the camera.
        tf_buffer (tf2_ros.Buffer): TF buffer to lookup transforms.

    Returns:
        list of dict: Updated list of blocks.
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
        rospy.loginfo(f"Identified camera coordinates: {cX}, {cY}")
        real_coords = get_real_world_coordinates(
            cX, cY, camera_calibration, camera_pose, tf_buffer)
        if real_coords is None:
            continue
        real_x, real_y, real_z = real_coords
        rospy.loginfo(f"Real world coordinates: {real_x}, {real_y}, {real_z}")

        # Attempt to match with previous blocks
        matched = False
        for prev_block in previous_blocks:
            distance = np.sqrt(
                (real_x - prev_block['pose']['position']['x'])**2 +
                (real_y - prev_block['pose']['position']['y'])**2 +
                (real_z - prev_block['pose']['position']['z'])**2
            )
            if distance < 0.05:  # Threshold for matching (meters)
                matched = True
                # Update confidence
                new_confidence = min(prev_block['confidence'] + 5, 100.0)
                prev_block['confidence'] = new_confidence
                # Update position
                prev_block['pose']['position']['x'] = real_x
                prev_block['pose']['position']['y'] = real_y
                prev_block['pose']['position']['z'] = real_z
                current_detected.append(prev_block)
                break

        if not matched:
            # New block detected
            block = {
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
            current_detected.append(block)

    # Optionally, decrease confidence for blocks not detected in this frame
    for prev_block in previous_blocks:
        if prev_block not in current_detected:
            prev_block['confidence'] -= 10
            if prev_block['confidence'] > 0:
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
            'world',
            camera_frame,
            rospy.Time(0),
            rospy.Duration(1.0)
        )
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
    # pub = rospy.Publisher('/blocks', Blocks, queue_size=10)  # Uncomment and define Blocks message as needed

    bridge = CvBridge()
    publish_blocks = []

    # Initialize TF buffer and listener
    tf_buffer = tf2_ros.Buffer()
    tf_listener = tf2_ros.TransformListener(tf_buffer)

    rospy.sleep(1.0)  # Give TF some time to fill

    # Capture background and blocks images from local camera
    background_image, blocks_image = get_local_camera_frames()
    if background_image is None or blocks_image is None:
        rospy.logerr("Failed to capture background or blocks image. Exiting.")
        return

    # Get initial camera pose
    # If you don't have a TF tree running on your Mac, you can mock the camera pose
    # For debugging purposes, we'll assume the camera is at the origin with no rotation
    camera_pose = PoseStamped()
    camera_pose.header.frame_id = "world"
    camera_pose.header.stamp = rospy.Time.now()
    camera_pose.pose.position.x = 0.0
    camera_pose.pose.position.y = 0.0
    camera_pose.pose.position.z = 0.0
    camera_pose.pose.orientation.x = 0.0
    camera_pose.pose.orientation.y = 0.0
    camera_pose.pose.orientation.z = 0.0
    camera_pose.pose.orientation.w = 1.0

    # Initial block detection
    publish_blocks = run_cv_first(
        background_image, blocks_image, camera_calibration, camera_pose, tf_buffer)

    rate = rospy.Rate(1)  # 1 Hz

    # Reinitialize VideoCapture for real-time processing
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        rospy.logerr("Cannot open camera for real-time processing")
        return

    rospy.loginfo("Starting real-time block detection. Press 'q' to quit.")

    while not rospy.is_shutdown():
        ret, frame = cap.read()
        if not ret:
            rospy.logerr("Failed to grab frame")
            break

        # Display the live feed
        cv2.imshow('Real-time Block Detection - Press "q" to quit', frame)

        # Convert frame to ROS Image message
        try:
            blocks_image_msg = bridge.cv2_to_imgmsg(
                cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), "mono8")
        except CvBridgeError as e:
            rospy.logerr(f"CV Bridge Error: {e}")
            continue

        # Update camera pose if using TF (here we use the mocked pose)
        # camera_pose = get_camera_pose(tf_buffer, camera_frame)
        # For debugging, we use the mocked camera_pose

        # Update blocks with real-time CV
        publish_blocks = run_cv(
            blocks_image_msg, background_image, publish_blocks, camera_calibration, camera_pose, tf_buffer)

        # Create the Blocks message
        blocks_msg = {}
        blocks_msg['blocks'] = publish_blocks
        blocks_msg['time_updated'] = rospy.Time.now().to_nsec()

        # Log the message
        rospy.loginfo(f"Publishing blocks: {blocks_msg}")

        # Publish the message
        # pub.publish(blocks_msg)  # Uncomment when the Blocks message is defined

        print(f"To publish: {blocks_msg}")

        key = cv2.waitKey(1)
        if key & 0xFF == ord('q'):
            break

        rate.sleep()

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass
