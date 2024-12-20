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
<<<<<<< HEAD
        # PICK
        # should be 0.670, 0.181, -0.081
        # 0.6496786873208795, -0.039610976832769845, -0.10166218522505455
        # -0.018, 0.710, -0.019, 0.704
        x, y, z = 0.722, -0.143, -0.139 
        xw, yw, zw, ww = 0, 1, 0, 0
        # xw, yw, zw, ww = -0.018, 0.710, -0.019, 0.704
        request.ik_request.pose_stamped.pose.position.x = x
        request.ik_request.pose_stamped.pose.position.y = y
        request.ik_request.pose_stamped.pose.position.z = z + 0.12
        request.ik_request.pose_stamped.pose.orientation.x = xw
        request.ik_request.pose_stamped.pose.orientation.y = yw
        request.ik_request.pose_stamped.pose.orientation.z = zw
        request.ik_request.pose_stamped.pose.orientation.w = ww
=======
>>>>>>> 860d06753ef9ca76460596bba87e3eaf8a52afa6
        
        z_offset = 0.15

<<<<<<< HEAD
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

        # Set the desired orientation for the end effector HERE
        s = input("stop")
        rospy.sleep(1.0)

        # NEUTRAL
        x, y, z = 0.722, -0.143, -0.139 
        xw, yw, zw, ww = 0, 1, 0, 0
        # xw, yw, zw, ww = -0.018, 0.710, -0.019, 0.704
        request.ik_request.pose_stamped.pose.position.x = x
        request.ik_request.pose_stamped.pose.position.y = y
        request.ik_request.pose_stamped.pose.position.z = z
        request.ik_request.pose_stamped.pose.orientation.x = xw
        request.ik_request.pose_stamped.pose.orientation.y = yw
        request.ik_request.pose_stamped.pose.orientation.z = zw
        request.ik_request.pose_stamped.pose.orientation.w = ww
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
            
        except rospy.ServiceException as e:
            print("Service call failed: %s"%e)

        print('Closing...')
        right_gripper.close()
        s = input("enter")

        # PLACE 0.757, 0.045, -0.149
        request.ik_request.pose_stamped.pose.position.x = 0.689
        request.ik_request.pose_stamped.pose.position.y = 0.161
        request.ik_request.pose_stamped.pose.position.z = 0.381        
        request.ik_request.pose_stamped.pose.orientation.x = 0.0
        request.ik_request.pose_stamped.pose.orientation.y = 1.0
        request.ik_request.pose_stamped.pose.orientation.z = 0.0
        request.ik_request.pose_stamped.pose.orientation.w = 0.0
=======
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
>>>>>>> 860d06753ef9ca76460596bba87e3eaf8a52afa6
        
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
