#!/usr/bin/env python
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import cv2
import time
import rospy
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Header
from sensor_msgs.msg import JointState
from intera_core_msgs.srv import SolvePositionFK, SolvePositionFKRequest
from moveit_commander import MoveGroupCommander


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


def is_topic_running(topic_name):
    """
    Check if a given topic is currently running.

    Args:
        topic_name (str): The name of the topic to check.

    Returns:
        bool: True if the topic is running, False otherwise.
    """
    try:
        # Get the list of currently published topics
        published_topics = rospy.get_published_topics()

        # Check if the topic is in the list
        for topic, _ in published_topics:
            if topic == topic_name:
                return True
        return False
    except rospy.ROSInterruptException:
        rospy.logerr("ROS Interrupt Exception occurred.")
        return False


def image_callback(msg, save=False):
    rospy.loginfo("Image callback triggered")
    try:
        # Convert the ROS Image message to a CV image
        cv_image = bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

        # Save the image to verify if display fails
        # cv2.imwrite("debug_output.jpg", cv_image)
        # rospy.loginfo("Saved debug image as debug_output.jpg")

        # run_cv on it

        # cv_image = run_cv(cv_image)

        # Display the image in a window
        cv2.imshow("Right Hand Camera", cv_image)
        if save:
            cv2.imwrite(
                f"/home/cc/ee106a/fa24/class/ee106a-aei/final_project/106afinalproject/in_gen/sawyer/CV/src/debug/{time.time()}.png", cv_image)
        cv2.waitKey(1)  # Update the window
    except CvBridgeError as e:
        rospy.logerr(f"Could not convert image: {e}")


def fk_to_cartesian(joint_angles):
    service_name = "ExternalTools/right/PositionKinematicsNode/FKService"
    fk_service_proxy = rospy.ServiceProxy(service_name, SolvePositionFK)
    fk_request = SolvePositionFKRequest()

    joints = JointState()
    joints.name = ['right_j0', 'right_j1', 'right_j2',
                   'right_j3', 'right_j4', 'right_j5', 'right_j6']
    joints.position = joint_angles
    fk_request.configuration.append(joints)

    fk_request.tip_names.append('right_hand')

    try:
        rospy.wait_for_service(service_name, 5.0)
        response = fk_service_proxy(fk_request)
    except (rospy.ServiceException, rospy.ROSException) as e:
        rospy.logerr("Service call failed: %s" % (e,))
        return None

    if response.isValid[0]:
        rospy.loginfo("SUCCESS - Valid Cartesian Solution Found")
        return response.pose_stamp[0]  # Return the pose of the end effector
    else:
        rospy.logerr("INVALID JOINTS - No Cartesian Solution Found.")
        return None


def move_to_position(pose):
    group = MoveGroupCommander("right_arm")

    # Max speed (use a value < 1.0 to scale down)
    group.set_max_velocity_scaling_factor(1.0)
    group.set_max_acceleration_scaling_factor(1.0)  # Max acceleration

    group.set_pose_target(pose)

    # Plan the motion
    plan = group.plan()

    # Check if planning was successful
    if isinstance(plan, tuple):
        trajectory = plan[1]  # Extract the trajectory from the tuple
    else:
        trajectory = plan

    group.execute(trajectory, wait=True)


def setup():
    rospy.init_node("main", anonymous=True)

    # move to start
    end_effector_pose = fk_to_cartesian(
        [0.034791015625, -1.318587890625, -0.103244140625, 1.436109375, 0.0623955078125, -0.099697265625, 1.5763955078125])
    move_to_position(end_effector_pose)
    # check if topic /blocks is available, if not, then wait, if so then subscribe
    # while (not is_topic_running("/blocks")):
    #     rospy.loginfo("Waiting for /blocks to be running...")
    #     time.sleep(1)

    # rospy.Subscriber("/blocks", PoseStamped, block_callback)

    # do a full scan of workspace, identify areas for load + build
    rospy.Subscriber(
        "/io/internal_camera/right_hand_camera/image_raw", Image, image_callback)
    rospy.loginfo("Right Hand Camera viewer node started.")
    # Keep the program running and processing callbacks
    rospy.spin()


if __name__ == '__main__':
    bridge = CvBridge()
    setup()
