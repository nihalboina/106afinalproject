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
import json

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

def pick_or_place(right_gripper, request, compute_ik, x, y, z, gripper_command=-1):
    # right_gripper = robot_gripper.Gripper('right_gripper')

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

import json
import time

def get_block_coordinates(json_file, num, retry_interval=5):
    """
    Extracts x, y, z coordinates of item `num` from a JSON file.
    If the file doesn't contain items, waits and keeps trying.

    :param file_path: Path to the JSON file.
    :param num: Item number to extract coordinates for.
    :return: Tuple of (x, y, z) coordinates.
    """
    while True:
        try:
            # Load the JSON file
            with open(json_file, 'r') as file:
                data = json.load(file)

            # Check if the item `num` exists
            if str(num) in data:
                item = data[str(num)]
                position = item["pose"]["position"]
                x = position["x"]
                y = position["y"]
                z = position["z"]
                return x, y, z
            else:
                print(f"Item {num} not found. Retrying...")

        except json.JSONDecodeError:
            print("Invalid JSON format. Retrying...")
        except FileNotFoundError:
            print("File not found. Retrying...")
        except KeyError:
            print("Unexpected data structure. Retrying...")

        # Wait before retrying
        time.sleep(1)

# Example usage
# coordinates = extract_xyz_from_json('data.json', 3)
# print(coordinates)


def get_placement_coordinates(json_file, block_num):
    try:
        # Load the JSON file
        with open(json_file, 'r') as file:
            data = json.load(file)

        # Check if the data has a 'world_points' key and ensure it's a list
        if 'world_points' in data and isinstance(data['world_points'], list):
            # Find the block with the specified block_id
            for block in data['world_points']:
                if block.get('block_id') == block_num:
                    # Extract the x, y, z coordinates
                    x = block.get('x', None)
                    y = block.get('y', None)
                    z = block.get('z', None)

                    if x is not None and y is not None and z is not None:
                        return x, y, z
                    else:
                        raise ValueError(f"Block {block_num} does not contain complete x, y, z coordinates.")
            raise ValueError(f"Block with block_id {block_num} not found.")
        else:
            raise ValueError("The JSON data does not contain a valid 'world_points' list.")

    except json.JSONDecodeError:
        raise ValueError("Failed to decode JSON. Please check the file format.")
    except FileNotFoundError:
        raise ValueError(f"File {json_file} not found.")
    
def main():
    json_file_cv =  "detected_blocks.json"  # Replace with your JSON file name
    json_file_gpt = "output.json"  # Replace with your JSON file name

    rospy.sleep(1)

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
        
        z_offset = 0.15

        x_pick_offset = -0.03
        y_pick_offset = 0
        z_pick_offset = -0.03

        x_place_offset = -0.1
        y_place_offset = 0.02
        z_place_offset = 0

        #get in cv coordinates block 0
        block_pick_num = 0
        try:
            x1, y1, z1 = get_block_coordinates(json_file_cv, block_pick_num)
            print(f"Coordinates of the first pick block: x={x1}, y={y1}, z={z1}")
        except ValueError as e:
            print(e)

        # pick 1
        pick_or_place(right_gripper, request, compute_ik, x1 + x_pick_offset -0.015, y1 + y_pick_offset, z1 + z_offset)
        pick_or_place(right_gripper, request, compute_ik, x1 + x_pick_offset -0.015, y1 + y_pick_offset, z1 + z_pick_offset, 0)
        
        pick_or_place(right_gripper, request, compute_ik, x1 + 0.01, y1, z1 + z_offset)

        #get gpt coordinates
        block_num = 1
        try:
            x2, y2, z2 = get_placement_coordinates(json_file_gpt, block_num)
            print(f"Coordinates of place block {block_num}: x={x2}, y={y2}, z={z2}")
        except ValueError as e:
            print(e)

        # place 1
        pick_or_place(right_gripper, request, compute_ik, x2 + x_place_offset +0.01, y2 + y_place_offset, z2 + z_offset)
        pick_or_place(right_gripper, request, compute_ik, x2 + x_place_offset +0.01, y2 + y_place_offset, z2 + z_place_offset, 1)
        pick_or_place(right_gripper, request, compute_ik, x2 +0.01, y2 - 0.005, z2 + z_offset)
        

        # pick 2
        block_pick_num = 1
        try:
            x3, y3, z3 = get_block_coordinates(json_file_cv, block_pick_num)
            print(f"Coordinates of the second pick block: x={x3}, y={y3}, z={z3}")
        except ValueError as e:
            print(e)
         
        pick_or_place(right_gripper, request, compute_ik, x3 + x_pick_offset, y3+ y_pick_offset, z3 + z_offset)

        pick_or_place(right_gripper, request, compute_ik, x3 + x_pick_offset, y3+ y_pick_offset, z3 + z_pick_offset, 0)
        
        pick_or_place(right_gripper, request, compute_ik, x3, y3, z3 + z_offset)

        # place 2
        block_num += 1
        try:
            x4, y4, z4 = get_placement_coordinates(json_file_gpt, block_num)
            print(f"Coordinates of place block {block_num}: x={x4}, y={y4}, z={z4}")
        except ValueError as e:
            print(e)
        pick_or_place(right_gripper, request, compute_ik, x4 + x_place_offset, y4 + y_place_offset, z4 + z_offset)
        pick_or_place(right_gripper, request, compute_ik, x4 + x_place_offset + 0.03, y4 + y_place_offset, z4 + z_place_offset, 1)
        pick_or_place(right_gripper, request, compute_ik, x4 - 0.04, y4, z4 + z_offset)
        

        # pick 3
        block_pick_num = 2
        try:
            x5, y5, z5 = get_block_coordinates(json_file_cv, block_pick_num)
            print(f"Coordinates of the third pick block: x={x5}, y={y5}, z={z5}")
        except ValueError as e:
            print(e)
         
        pick_or_place(right_gripper, request, compute_ik, x5 + x_pick_offset, y5+ y_pick_offset, z5 + z_offset)

        pick_or_place(right_gripper, request, compute_ik, x5 + x_pick_offset, y5+ y_pick_offset, z5 + z_pick_offset, 0)
        
        pick_or_place(right_gripper, request, compute_ik, x5, y5, z5 + z_offset)

        # place 3
        block_num += 1
        try:
            x6, y6, z6 = get_placement_coordinates(json_file_gpt, block_num)
            print(f"Coordinates of place block {block_num}: x={x6}, y={y6}, z={z6}")
        except ValueError as e:
            print(e)

        pick_or_place(right_gripper, request, compute_ik, x6 + x_place_offset, y6 + y_place_offset, z6 + z_offset)
        pick_or_place(right_gripper, request, compute_ik, x6 + x_place_offset + 0.035, y6 + y_place_offset + z_place_offset, z6, 1)
        pick_or_place(right_gripper, request, compute_ik, x6 + 0.035, y6, z6 + z_offset)
        
        tuck()


# Python's syntax for a main() method
if __name__ == '__main__':
    main()
