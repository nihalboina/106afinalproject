#!/usr/bin/env python
import rospy
import rospkg
import roslaunch
from moveit_msgs.srv import GetPositionIK, GetPositionIKRequest, GetPositionIKResponse
from geometry_msgs.msg import PoseStamped
from moveit_commander import MoveGroupCommander

from paths.trajectories import LinearTrajectory, CircularTrajectory
from paths.paths import MotionPath
from paths.path_planner import PathPlanner
from controllers.controllers import ( 
    PIDJointVelocityController, 
    FeedforwardJointVelocityController
)
import numpy as np
from numpy import linalg
import sys
from intera_interface import gripper as robot_gripper

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

def blocks_callback(msg):
    """
    Callback function to process the received blocks message.

    Args:
        msg (std_msgs.msg.String): Message received from the /blocks topic.
    """
    try:
        # Assuming the published message is in JSON format
        blocks_data = json.loads(msg.data)
        rospy.loginfo("Received blocks data:")
        rospy.loginfo(json.dumps(blocks_data, indent=4))

        # Example: Process each block
        for block in blocks_data.get('blocks', []):
            rospy.loginfo(f"Block Position: {block['pose']['position']}")
            rospy.loginfo(f"Block Area: {block['area']}")

    except json.JSONDecodeError as e:
        rospy.logerr(f"Failed to decode JSON: {e}")
    except Exception as e:
        rospy.logerr(f"Unexpected error: {e}")

def pick_or_place(request, compute_ik, x, y, z, gripper_command=-1):
    request.ik_request.pose_stamped.pose.position.x = x
    request.ik_request.pose_stamped.pose.position.y = y
    request.ik_request.pose_stamped.pose.position.z = z
    request.ik_request.pose_stamped.pose.orientation.x = 0
    request.ik_request.pose_stamped.pose.orientation.y = 1
    request.ik_request.pose_stamped.pose.orientation.z = 0
    request.ik_request.pose_stamped.pose.orientation.w = 0

    try:
        # Send the request to the service
        response = compute_ik(request)
        
        # Print the response HERE
        print(response)
        group = MoveGroupCommander("right_arm")

        # Setting position and orientation target
        group.set_pose_target(request.ik_request.pose_stamped)

        # TRY THIS
        # Setting just the position without specifying the orientation
        ###group.set_position_target([0.5, 0.5, 0.0])

        # Plan IK
        plan = group.plan()
        user_input = input("Enter 'y' if the trajectory looks safe on RVIZ")
        
        # Execute IK if safe
        if user_input == 'y':
            group.execute(plan[1])
            right_gripper = robot_gripper.Gripper('right_gripper')
            print('Closing...')
            right_gripper.close()
            rospy.sleep(1.0)

    except rospy.ServiceException as e:
        print("Service call failed: %s"%e)

    if gripper_command == 0:
        print('Closing...')
        right_gripper.close()
        rospy.sleep(1.0)
    elif gripper_command == 1:
        print('Opening...')
        right_gripper.open()
        rospy.sleep(1.0)


def main():
    # Wait for the IK service to become available
    rospy.wait_for_service('compute_ik')
    rospy.init_node('service_query')
    # Create the function used to call the service
    compute_ik = rospy.ServiceProxy('compute_ik', GetPositionIK)
    
    # Set up the right gripper
    right_gripper = robot_gripper.Gripper('right_gripper')

    """
    Initializes the ROS node and subscribes to the /blocks topic.
    """
    rospy.init_node('blocks_subscriber', anonymous=True)

    # Subscribe to the /blocks topic
    rospy.Subscriber('/blocks', String, blocks_callback)

    rospy.loginfo("Blocks subscriber started. Listening to /blocks topic...")
    rospy.spin()

    tuck()

    # Calibrate the gripper (other commands won't work unless you do this first)
    print('Calibrating...')
    right_gripper.calibrate()
    rospy.sleep(2.0)

    while not rospy.is_shutdown():
        input('Press [ Enter ]: ')
        
        # Construct the request
        request = GetPositionIKRequest()
        request.ik_request.group_name = "right_arm"

        # If a Sawyer does not have a gripper, replace '_gripper_tip' with '_wrist' instead
        # link = "stp_022312TP99620_tip_1"
        link = "right_gripper_tip"

        request.ik_request.ik_link_name = link
        # request.ik_request.attempts = 20
        request.ik_request.pose_stamped.header.frame_id = "base"
        
        # Set the desired orientation for the end effector HERE
        ###group.set_position_target([0.5, 0.5, 0.0])
        
        z_offset = 0.15

        # pick 1
        x1, y1, z1 = 
        pick_or_place(request, compute_ik, x1, y1, z1 + z_offset)
        pick_or_place(request, compute_ik, x1, y1, z1, 0)
        
        tuck()

        # place 1
        x2, y2, z2 = 
        pick_or_place(request, compute_ik, x2, y2, z2 + z_offset)
        pick_or_place(request, compute_ik, x2, y2, z2, 1)
        pick_or_place(request, compute_ik, x2, y2, z2 + z_offset)
        
        tuck()

        # pick 2
        x3, y3, z3 = 
        pick_or_place(request, compute_ik, x3, y3, z3 + z_offset)
        pick_or_place(request, compute_ik, x3, y3, z3, 0)
        
        tuck()

        # place 2
        x4, y4, z4 = 
        pick_or_place(request, compute_ik, x4, y4, z4 + z_offset)
        pick_or_place(request, compute_ik, x4, y4, z4, 1)
        pick_or_place(request, compute_ik, x4, y4, z4 + z_offset)
        
        tuck()

        # pick 3
        x5, y5, z5 = 
        pick_or_place(request, compute_ik, x5, y5, z5 + z_offset)
        pick_or_place(request, compute_ik, x5, y5, z5, 0)
        
        tuck()

        # place 3
        x6, y6, z6 = 
        pick_or_place(request, compute_ik, x6, y6, z6 + z_offset)
        pick_or_place(request, compute_ik, x6, y6, z6, 1)
        pick_or_place(request, compute_ik, x6, y6, z6 + z_offset)
        
        tuck()


# Python's syntax for a main() method
if __name__ == '__main__':
    main()
