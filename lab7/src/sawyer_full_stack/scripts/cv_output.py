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
from claude_attempt import CameraTransform
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


def run_output(image, camera_transform):
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
    new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(
        K, D, (w, h), 1, (w, h))

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
    sobel_combined = cv2.convertScaleAbs(
        sobel_x) + cv2.convertScaleAbs(sobel_y)

    # Use binary thresholding on Sobel edges
    _, thresh = cv2.threshold(sobel_combined, 100, 255, cv2.THRESH_BINARY)

    # Find contours from the Sobel edge-detected image
    contours, _ = cv2.findContours(
        thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    output_image = cv2.cvtColor(undistorted_image, cv2.COLOR_GRAY2BGR)
    midpoint_x, midpoint_y, angle = None, None, None
    box = None

    generated_json = {"u_v_points": []}
    output_data = {}
    output_data["world_points"] = []

    if contours:
        largest_contour = max(contours, key=cv2.contourArea)

        # Get the rotated bounding box
        rotated_rect = cv2.minAreaRect(largest_contour)
        box = cv2.boxPoints(rotated_rect)
        box = np.int0(box)
        angle = rotated_rect[-1]  # Get the angle of rotation

        # Calculate the midpoint of the rotated bounding box
        midpoint_x = int((box[0][0] + box[2][0]) / 2)
        midpoint_y = int((box[0][1] + box[2][1]) / 2)

        # Draw the rotated bounding box and midpoint on the undistorted image
        cv2.drawContours(output_image, [box], 0, (0, 255, 0), 2)
        cv2.circle(output_image, (midpoint_x, midpoint_y), 5, (0, 0, 255), -1)

        # Adjust the angle
        if angle < -45:
            angle += 90
        print(f"Corrected Computed Rotation Angle: {angle:.2f} degrees")

    # Load JSON data from the GPT file
    json_file_path = "/home/cc/ee106a/fa24/class/ee106a-aei/final_project/106afinalproject/in_gen/GPTComponent/Chatgpt_input.json"
    with open(json_file_path, 'r') as f:
        json_data_file = json.load(f)

    # Rotate and map the GPT blocks if valid midpoint is found
    if midpoint_x is not None and midpoint_y is not None and angle is not None:
        # Convert angle to radians
        angle_rad = np.deg2rad(90 - angle)
        rotation_matrix = np.array([
            [np.cos(angle_rad), -np.sin(angle_rad)],
            [np.sin(angle_rad),  np.cos(angle_rad)]
        ])

        # Plot only the GPT-based JSON data points

        for block in json_data_file["blocks"]:
            x_original, y_original = block["position"][:2]
            # Apply rotation and scaling
            rotated_position = rotation_matrix @ np.array(
                [x_original * 0.7, y_original * 0.75])
            x_final = int(midpoint_x + rotated_position[0])
            y_final = int(midpoint_y - rotated_position[1])

            # Add to generated JSON
            generated_json["u_v_points"].append(
                {"block_id": block["block_id"], "u": x_final, "v": y_final})

            # Set color based on the block's color attribute
            if block["color"] == "gray":
                # GPT gray blocks -> Green
                color = (0, 255, 0)
            else:
                # Other colors -> Cyan
                color = (0, 255, 255)

            # Draw the block position as a circle
            cv2.circle(output_image, (x_final, y_final), 5, color, -1)
            # Normalize by dividing by 40 since the layers are in increments of 40
            layer_number = int(block["position"][2] / 40.0)
            real_x, real_y, real_z = camera_transform.distorted_pixel_to_base(
                x_final, y_final, layer_num=layer_number)
            print(
                f"Block ID {block['block_id']} -> Translated (u, v): ({x_final}, {y_final}) -> Translated real-world (x,y,z): ({real_x, real_y, real_z})")

            output_data["world_points"].append({
                "block_id": block['block_id'],
                "x": round(real_x, 4),
                "y": round(real_y, 4),
                "z": round(real_z, 4)
            })

    with open("/home/cc/ee106a/fa24/class/ee106a-aei/final_project/106afinalproject/main/src/output.json", 'w') as f:
        json.dump(output_data, f, indent=4)
    # Overlay the stud pattern if we have the box
    if box is not None:
        sorted_box_x = sorted(box, key=lambda x: x[0])
        left_edge = sorted_box_x[0:2]

        sorted_box_y = sorted(box, key=lambda x: -x[1])
        top_edge = sorted_box_y[0:2]

        # Stud points -> Red
        stud_color = (255, 0, 0)

        for left_edge_i in range(8):
            left_edge_x = left_edge[0][0] + \
                (left_edge[1][0] - left_edge[0][0]) * (left_edge_i + 0.5)/8
            left_edge_y = left_edge[0][1] + \
                (left_edge[1][1] - left_edge[0][1]) * (left_edge_i + 0.5)/8

            left_edge_x += (top_edge[1][0] - top_edge[0][0]) / 16
            left_edge_y += (top_edge[1][1] - top_edge[0][1]) / 16
            cv2.circle(output_image, (int(left_edge_x),
                       int(left_edge_y)), 5, stud_color, -1)

            for top_edge_i in range(7):
                left_edge_x += (top_edge[1][0] - top_edge[0][0]) / 8
                left_edge_y += (top_edge[1][1] - top_edge[0][1]) / 8
                cv2.circle(output_image, (int(left_edge_x),
                           int(left_edge_y)), 5, stud_color, -1)

    # Save generated JSON
    generated_json_path = "/home/cc/ee106a/fa24/class/ee106a-aei/final_project/106afinalproject/in_gen/GPTComponent/GPTCoordToBase/Generated_UV_Points.json"
    with open(generated_json_path, 'w') as f:
        json.dump(generated_json, f, indent=4)

    # Display the final result
    # fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    # ax.imshow(cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB))
    # ax.set_title("Undistorted Image with Overlaid GPT Coordinates (Green/Cyan) and Stud Pattern (Red)")
    # ax.axis("off")
    # plt.show()

    return output_image

    # cv2.imshow("Undistorted Image with Overlaid GPT Coordinates", output_image)
    # cv2.waitKey(1)


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
    camera_transform = CameraTransform()

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
        output_image = run_cv(blocks_image, camera_transform)

        cv2.imshow(
            "Undistorted Image with Overlaid GPT Coordinates", output_image)
        cv2.waitKey(1)

        # Draw detected blocks on the image
        # try:
        #     image = bridge.imgmsg_to_cv2(image_msg, desired_encoding='bgr8')
        #     cv2.imshow("Raw", image)
        #     image = draw_blocks(image, publish_blocks)
        #     cv2.imshow("Detected Blocks", image)
        #     cv2.waitKey(1)  # Non-blocking
        # except CvBridgeError as e:
        #     rospy.logerr(f"CV Bridge Error: {e}")

        # # Create the Blocks message
        # blocks_msg = {}
        # blocks_msg['blocks'] = publish_blocks
        # blocks_msg['time_updated'] = rospy.Time.now().to_nsec()

        # # Log the message
        # rospy.loginfo(f"Publishing blocks: {blocks_msg}")

        # # Publish the message (uncomment and modify as per your message type)
        # # pub.publish(blocks_msg)
        # print(f"To publish: {blocks_msg}")

        # Sleep for the remainder of the loop
        rate.sleep()


if __name__ == '__main__':
    try:
        tuck()
        rospy.sleep(5.0)
        main()
    except rospy.ROSInterruptException:
        pass
