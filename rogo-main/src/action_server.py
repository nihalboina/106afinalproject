#!/usr/bin/env python
import sys
import argparse
import numpy as np
import rospkg
import roslaunch
import time
import json

from paths.trajectories import LinearTrajectory, CircularTrajectory
from paths.paths import MotionPath
from paths.path_planner import PathPlanner
from controllers.controllers import (
    PIDJointVelocityController,
    FeedforwardJointVelocityController
)
from utils.utils import *

from trac_ik_python.trac_ik import IK

import rospy
import tf2_ros
import intera_interface
from moveit_msgs.msg import DisplayTrajectory, RobotState
from sawyer_pykdl import sawyer_kinematics

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


def get_trajectory(limb, kin, ik_solver, tag_pos, args):
    """
    Returns an appropriate robot trajectory for the specified task.  You should 
    be implementing the path functions in paths.py and call them here

    Parameters
    ----------
    task : string
        name of the task.  Options: line, circle, square
    tag_pos : 3x' :obj:`numpy.ndarray`

    Returns
    -------
    :obj:`moveit_msgs.msg.RobotTrajectory`
    """
    num_way = args.num_way
    task = args.task

    tfBuffer = tf2_ros.Buffer()
    listener = tf2_ros.TransformListener(tfBuffer)

    try:
        trans = tfBuffer.lookup_transform(
            'base', 'right_hand', rospy.Time(0), rospy.Duration(10.0))
    except Exception as e:
        print(e)

    current_position = np.array(
        [getattr(trans.transform.translation, dim) for dim in ('x', 'y', 'z')])
    print("Current Position:", current_position)

    if task == 'line':
        # target_pos = tag_pos[0] #idk why they did this
        target_pos = np.array(tag_pos)  # we used this in PID_skyler
        print("TARGET POSITION:", target_pos)
        print("TARGET POSITION:", target_pos)
        trajectory = LinearTrajectory(
            start_position=current_position, goal_position=target_pos, total_time=9)
    elif task == 'circle':
        target_pos = tag_pos[0]
        target_pos[2] += 0.5
        print("TARGET POSITION:", target_pos)
        trajectory = CircularTrajectory(
            center_position=target_pos, radius=0.1, total_time=15)

    else:
        raise ValueError('task {} not recognized'.format(task))

    path = MotionPath(limb, kin, ik_solver, trajectory)
    return path.to_robot_trajectory(num_way, True)


def get_controller(controller_name, limb, kin):
    """
    Gets the correct controller from controllers.py

    Parameters
    ----------
    controller_name : string

    Returns
    -------
    :obj:`Controller`
    """
    if controller_name == 'open_loop':
        controller = FeedforwardJointVelocityController(limb, kin)
    elif controller_name == 'pid':
        Kp = 0.2 * np.array([0.4, 2, 1.7, 1.5, 2, 2, 3])
        Kd = 0.01 * np.array([2, 1, 2, 0.5, 0.8, 0.8, 0.8])
        Ki = 0.01 * np.array([1.4, 1.4, 1.4, 1, 0.6, 0.6, 0.6])
        Kw = np.array([0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9])
        controller = PIDJointVelocityController(limb, kin, Kp, Ki, Kd, Kw)
    else:
        raise ValueError(
            'Controller {} not recognized'.format(controller_name))
    return controller


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
                        raise ValueError(
                            f"Block {block_num} does not contain complete x, y, z coordinates.")
            raise ValueError(f"Block with block_id {block_num} not found.")
        else:
            raise ValueError(
                "The JSON data does not contain a valid 'world_points' list.")

    except json.JSONDecodeError:
        raise ValueError(
            "Failed to decode JSON. Please check the file format.")
    except FileNotFoundError:
        raise ValueError(f"File {json_file} not found.")


def find_total_pick_blocks(file_path):
    """
    Finds the total number of blocks in a JSON file.

    :param file_path: Path to the JSON file.
    :return: The total number of blocks (integer).
    """
    try:
        # Load the JSON file
        with open(file_path, 'r') as file:
            data = json.load(file)

        # Return the total number of blocks
        total_blocks = len(data.keys())
        return total_blocks
    except Exception as e:
        print(f"Error: {e}")
        return None


def find_highest_block_id(file_path):
    """
    Finds the highest block_id in a JSON file containing world points.

    :param file_path: Path to the JSON file.
    :return: The highest block_id (integer).
    """
    try:
        # Load the JSON file
        with open(file_path, 'r') as file:
            data = json.load(file)

        # Extract the highest block_id
        highest_block_id = max(point["block_id"]
                               for point in data["world_points"])
        return highest_block_id
    except Exception as e:
        print(f"Error: {e}")
        return None


def main():
    """
    Examples of how to run me:
    python scripts/main.py --help <------This prints out all the help messages
    and describes what each parameter is
    python pid_parth.py -t line -ar_marker 3 -c pid --log

    You can also change the rate, timeout if you want
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-task', '-t', type=str, default='line', help='Options: line, circle.  Default: line'
                        )
    parser.add_argument('-ar_marker', '-ar', nargs='+', help='Which AR marker to use.  Default: 1'
                        )
    parser.add_argument('-controller_name', '-c', type=str, default='moveit',
                        help='Options: moveit, open_loop, pid.  Default: moveit'
                        )
    parser.add_argument('-rate', type=int, default=200, help="""
        This specifies how many ms between loops.  It is important to use a rate
        and not a regular while loop because you want the loop to refresh at a
        constant rate, otherwise you would have to tune your PD parameters if 
        the loop runs slower / faster.  Default: 200"""
                        )
    parser.add_argument('-timeout', type=int, default=None, help="""after how many seconds should the controller terminate if it hasn\'t already.  
        Default: None"""
                        )
    parser.add_argument('-num_way', type=int, default=50, help='How many waypoints for the :obj:`moveit_msgs.msg.RobotTrajectory`.  Default: 300'
                        )
    parser.add_argument('--log', action='store_true',
                        help='plots controller performance')
    args = parser.parse_args()

    rospy.init_node('moveit_node')

    tuck()

    # Set up the right gripper
    right_gripper = robot_gripper.Gripper('right_gripper')
    # Calibrate the gripper (other commands won't work unless you do this first)
    print('Calibrating...')
    right_gripper.calibrate()
    rospy.sleep(2.0)

    # this is used for sending commands (velocity, torque, etc) to the robot
    ik_solver = IK("base", "right_gripper_tip")
    limb = intera_interface.Limb("right")
    kin = sawyer_kinematics("right")

    # json file paths
    json_file_cv = "detected_blocks.json"  # Replace with your JSON file name
    json_file_gpt = "output.json"  # Replace with your JSON file name

    # find the min number of blocks between pick and place blocks to know how many times we can do it
    num_pick_blocks = find_total_pick_blocks(json_file_cv)
    num_place_blocks = find_highest_block_id(json_file_gpt)
    loop_length = min(num_pick_blocks, num_place_blocks)

    z_hover_offset = 0.15  # offset to be above block but not hit others
    z_pick_offset = -0.03  # offset to grip the block
    z_place_offset = 0  # offset to drop the block

    x_cam_offset = 0.1
    y_cam_offset = 0.03

    for i in range(loop_length):

        '''
        6 movement steps for each block

        find pick location, then:
        1. go slightly vertically above pick position
        2. go down to pick up and close the gripper
        3. go slightly vertically up to not hit other blocks
        find drop location, then:
        4. go vertically above drop position so as to not hit other blocks
        5. go down and open the gripper
        6. go slightly vertically up to not hit other blocks
        '''

        # Lookup the block pick up position.
        block_pick_num = i  # pick numbers start from 0
        x1, y1, z1 = get_block_coordinates(json_file_cv, block_pick_num)
        print(f"Coordinates of the first pick block: x={x1}, y={y1}, z={z1}")
        pick_hover_pos = [x1, y1, z1 + z_hover_offset, 0, 1, 0, 0]
        pick_grip_pos = [x1, y1, z1 + z_pick_offset, 0, 1, 0, 0]

        # 1. go slightly vertically above pick position
        # Get an appropriate RobotTrajectory for the task (circular, linear, or square)
        robot_trajectory = get_trajectory(
            limb, kin, ik_solver, pick_hover_pos, args)

        # This is a wrapper around MoveIt! for you to use.  We use MoveIt! to go to the start position
        planner = PathPlanner('right_arm')

        # By publishing the trajectory to the move_group/display_planned_path topic, you should
        # be able to view it in RViz.  You will have to click the "loop animation" setting in
        # the planned path section of MoveIt! in the menu on the left side of the screen.
        pub = rospy.Publisher('move_group/display_planned_path',
                              DisplayTrajectory, queue_size=10)
        disp_traj = DisplayTrajectory()
        disp_traj.trajectory.append(robot_trajectory)
        disp_traj.trajectory_start = RobotState()
        pub.publish(disp_traj)

        # Move to the trajectory start position
        plan = planner.plan_to_joint_pos(
            robot_trajectory.joint_trajectory.points[0].positions)
        if args.controller_name != "moveit":
            plan = planner.retime_trajectory(plan, 0.3)
        planner.execute_plan(plan[1])

        if args.controller_name == "moveit":
            try:
                input('Press <Enter> to execute the trajectory using MOVEIT')
            except KeyboardInterrupt:
                sys.exit()
            # Uses MoveIt! to execute the trajectory.
            planner.execute_plan(robot_trajectory)
        else:
            controller = get_controller(args.controller_name, limb, kin)
            try:
                input(
                    'Press <Enter> to execute the trajectory using YOUR OWN controller')
            except KeyboardInterrupt:
                sys.exit()
            # execute the path using your own controller.
            done = controller.execute_path(
                robot_trajectory,
                rate=args.rate,
                timeout=args.timeout,
                log=args.log
            )
            if not done:
                print('Failed to move to position')
                sys.exit(0)

        # 2. go down to pick up and close the gripper
        robot_trajectory = get_trajectory(
            limb, kin, ik_solver, pick_grip_pos, args)

        planner = PathPlanner('right_arm')

        pub = rospy.Publisher('move_group/display_planned_path',
                              DisplayTrajectory, queue_size=10)
        disp_traj = DisplayTrajectory()
        disp_traj.trajectory.append(robot_trajectory)
        disp_traj.trajectory_start = RobotState()
        pub.publish(disp_traj)

        plan = planner.plan_to_joint_pos(
            robot_trajectory.joint_trajectory.points[0].positions)
        if args.controller_name != "moveit":
            plan = planner.retime_trajectory(plan, 0.3)
        planner.execute_plan(plan[1])

        if args.controller_name == "moveit":
            try:
                input('Press <Enter> to execute the trajectory using MOVEIT')
            except KeyboardInterrupt:
                sys.exit()
            planner.execute_plan(robot_trajectory)
        else:
            controller = get_controller(args.controller_name, limb, kin)
            try:
                input(
                    'Press <Enter> to execute the trajectory using YOUR OWN controller')
            except KeyboardInterrupt:
                sys.exit()
            done = controller.execute_path(
                robot_trajectory,
                rate=args.rate,
                timeout=args.timeout,
                log=args.log
            )
            if not done:
                print('Failed to move to position')
                sys.exit(0)
        # grab block
        print('Closing...')
        right_gripper.close()
        rospy.sleep(1.0)
        # incase it didn't work, doing again
        right_gripper.close()
        rospy.sleep(1.0)

        # 3. go slightly vertically above pick position
        robot_trajectory = get_trajectory(
            limb, kin, ik_solver, pick_hover_pos, args)

        planner = PathPlanner('right_arm')

        pub = rospy.Publisher('move_group/display_planned_path',
                              DisplayTrajectory, queue_size=10)
        disp_traj = DisplayTrajectory()
        disp_traj.trajectory.append(robot_trajectory)
        disp_traj.trajectory_start = RobotState()
        pub.publish(disp_traj)

        plan = planner.plan_to_joint_pos(
            robot_trajectory.joint_trajectory.points[0].positions)
        if args.controller_name != "moveit":
            plan = planner.retime_trajectory(plan, 0.3)
        planner.execute_plan(plan[1])

        if args.controller_name == "moveit":
            try:
                input('Press <Enter> to execute the trajectory using MOVEIT')
            except KeyboardInterrupt:
                sys.exit()
            planner.execute_plan(robot_trajectory)
        else:
            controller = get_controller(args.controller_name, limb, kin)
            try:
                input(
                    'Press <Enter> to execute the trajectory using YOUR OWN controller')
            except KeyboardInterrupt:
                sys.exit()
            done = controller.execute_path(
                robot_trajectory,
                rate=args.rate,
                timeout=args.timeout,
                log=args.log
            )
            if not done:
                print('Failed to move to position')
                sys.exit(0)

        # Lookup the block placement position.
        block_num = i + 1  # place block id numbers start from 1 so need to add 1 to i
        try:
            x2, y2, z2 = get_placement_coordinates(json_file_gpt, block_num)
            print(
                f"Coordinates of place block {block_num}: x={x2}, y={y2}, z={z2}")
        except ValueError as e:
            print(e)
        place_hover_pos = [x2, y2, z2 + z_hover_offset, 0, 1, 0, 0]
        place_grip_pos = [x2, y2, z2 + z_place_offset, 0, 1, 0, 0]

        # 4. go slightly vertically above place position
        robot_trajectory = get_trajectory(
            limb, kin, ik_solver, place_hover_pos, args)

        planner = PathPlanner('right_arm')

        pub = rospy.Publisher('move_group/display_planned_path',
                              DisplayTrajectory, queue_size=10)
        disp_traj = DisplayTrajectory()
        disp_traj.trajectory.append(robot_trajectory)
        disp_traj.trajectory_start = RobotState()
        pub.publish(disp_traj)

        plan = planner.plan_to_joint_pos(
            robot_trajectory.joint_trajectory.points[0].positions)
        if args.controller_name != "moveit":
            plan = planner.retime_trajectory(plan, 0.3)
        planner.execute_plan(plan[1])

        if args.controller_name == "moveit":
            try:
                input('Press <Enter> to execute the trajectory using MOVEIT')
            except KeyboardInterrupt:
                sys.exit()
            planner.execute_plan(robot_trajectory)
        else:
            controller = get_controller(args.controller_name, limb, kin)
            try:
                input(
                    'Press <Enter> to execute the trajectory using YOUR OWN controller')
            except KeyboardInterrupt:
                sys.exit()
            done = controller.execute_path(
                robot_trajectory,
                rate=args.rate,
                timeout=args.timeout,
                log=args.log
            )
            if not done:
                print('Failed to move to position')
                sys.exit(0)

        # 5. go down to place position
        robot_trajectory = get_trajectory(
            limb, kin, ik_solver, place_grip_pos, args)

        planner = PathPlanner('right_arm')

        pub = rospy.Publisher('move_group/display_planned_path',
                              DisplayTrajectory, queue_size=10)
        disp_traj = DisplayTrajectory()
        disp_traj.trajectory.append(robot_trajectory)
        disp_traj.trajectory_start = RobotState()
        pub.publish(disp_traj)

        plan = planner.plan_to_joint_pos(
            robot_trajectory.joint_trajectory.points[0].positions)
        if args.controller_name != "moveit":
            plan = planner.retime_trajectory(plan, 0.3)
        planner.execute_plan(plan[1])

        if args.controller_name == "moveit":
            try:
                input('Press <Enter> to execute the trajectory using MOVEIT')
            except KeyboardInterrupt:
                sys.exit()
            planner.execute_plan(robot_trajectory)
        else:
            controller = get_controller(args.controller_name, limb, kin)
            try:
                input(
                    'Press <Enter> to execute the trajectory using YOUR OWN controller')
            except KeyboardInterrupt:
                sys.exit()
            done = controller.execute_path(
                robot_trajectory,
                rate=args.rate,
                timeout=args.timeout,
                log=args.log
            )
            if not done:
                print('Failed to move to position')
                sys.exit(0)
        # drop block
        print('Opening...')
        right_gripper.open()
        rospy.sleep(1.0)
        # incase didn't work
        right_gripper.open()
        rospy.sleep(1.0)

        # 6. go slightly vertically above place position
        robot_trajectory = get_trajectory(
            limb, kin, ik_solver, place_hover_pos, args)

        planner = PathPlanner('right_arm')

        pub = rospy.Publisher('move_group/display_planned_path',
                              DisplayTrajectory, queue_size=10)
        disp_traj = DisplayTrajectory()
        disp_traj.trajectory.append(robot_trajectory)
        disp_traj.trajectory_start = RobotState()
        pub.publish(disp_traj)

        plan = planner.plan_to_joint_pos(
            robot_trajectory.joint_trajectory.points[0].positions)
        if args.controller_name != "moveit":
            plan = planner.retime_trajectory(plan, 0.3)
        planner.execute_plan(plan[1])

        if args.controller_name == "moveit":
            try:
                input('Press <Enter> to execute the trajectory using MOVEIT')
            except KeyboardInterrupt:
                sys.exit()
            planner.execute_plan(robot_trajectory)
        else:
            controller = get_controller(args.controller_name, limb, kin)
            try:
                input(
                    'Press <Enter> to execute the trajectory using YOUR OWN controller')
            except KeyboardInterrupt:
                sys.exit()
            done = controller.execute_path(
                robot_trajectory,
                rate=args.rate,
                timeout=args.timeout,
                log=args.log
            )
            if not done:
                print('Failed to move to position')
                sys.exit(0)

        # increase loop count
        i += 1


if __name__ == "__main__":
    main()
