#!/usr/bin/env python
"""
Starter script for 106a lab7. 
Author: Chris Correa
"""
import sys
import argparse
import numpy as np
import rospkg
import roslaunch

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


import intera_external_devices
from intera_interface import gripper as robot_gripper


### MOD: adding a gripper object:




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
    Returns an appropriate robot trajectory for the specified task.
    """
    num_way = args.num_way
    task = args.task

    tfBuffer = tf2_ros.Buffer()
    listener = tf2_ros.TransformListener(tfBuffer)

    try:
        trans = tfBuffer.lookup_transform('base', 'right_hand', rospy.Time(0), rospy.Duration(10.0))
    except Exception as e:
        print(e)

    current_position = np.array([getattr(trans.transform.translation, dim) for dim in ('x', 'y', 'z')])
    print("Current Position:", current_position)

    if task == 'line':
        # Convert tag_pos to numpy array and only use position components
        target_pos = np.array(tag_pos)
        print("TARGET POSITION:", target_pos)
        trajectory = LinearTrajectory(start_position=current_position, goal_position=target_pos, total_time=9)
    elif task == 'circle': # WE PROBABLY DONT USE THIS
        target_pos = np.array(tag_pos[:3])
        target_pos[2] += 0.5
        print("TARGET POSITION:", target_pos)
        trajectory = CircularTrajectory(center_position=target_pos, radius=0.1, total_time=15)

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
        raise ValueError('Controller {} not recognized'.format(controller_name))
    return controller


def main():
    """
    Examples of how to run me:
    python scripts/main.py --help <------This prints out all the help messages
    and describes what each parameter is
    python scripts/main.py -t line -ar_marker 3 -c torque --log
 
    You can also change the rate, timeout if you want
    """
    rospy.init_node('moveit_node')

    right_gripper = robot_gripper.Gripper('right_gripper')


    parser = argparse.ArgumentParser()
    parser.add_argument('-task', '-t', type=str, default='line', help=
        'Options: line, circle.  Default: line'
    )
    parser.add_argument('-ar_marker', '-ar', nargs='+', help=
        'Which AR marker to use.  Default: 1'
    )
    parser.add_argument('-controller_name', '-c', type=str, default='pid', 
        help='Options: moveit, open_loop, pid.  Default: pid'
    )
    parser.add_argument('-rate', type=int, default=200, help="""
        This specifies how many ms between loops.  It is important to use a rate
        and not a regular while loop because you want the loop to refresh at a
        constant rate, otherwise you would have to tune your PD parameters if 
        the loop runs slower / faster.  Default: 200"""
    )
    parser.add_argument('-timeout', type=int, default=None, help=
        """after how many seconds should the controller terminate if it hasn\'t already.  
        Default: None"""
    )
    parser.add_argument('-num_way', type=int, default=50, help=
        'How many waypoints for the :obj:`moveit_msgs.msg.RobotTrajectory`.  Default: 300'
    )
    parser.add_argument('--log', action='store_true', help='plots controller performance')
    args = parser.parse_args()


    print('Calibrating...')
    right_gripper.calibrate() # ADDED TO CALIBRATE THE GRIPPER< NEED TO DO THIS BEFORE WE USE IT
    rospy.sleep(2.0)


    tuck()

    
    # this is used for sending commands (velocity, torque, etc) to the robot
    ik_solver = IK("base", "right_gripper_tip")
    limb = intera_interface.Limb("right")
    kin = sawyer_kinematics("right")

    # Lookup the AR tag position.
    z_offset = 0.12
    x_cam_offset = 0.1
    y_cam_offset = 0.03

    # tag_pos = [0.6460573135993091 + x_cam_offset, -0.1749450668218313 + y_cam_offset, -0.10282176833689616 + 0.2 + z_offset, -0.7, 0.7, 0, 0] # TODO: CHANGE THIS BASED ON WHAT OUR 
    x1, y1, z1 = 
    tag_pos1 = [x1, y1, z1 + z_offset, 0, 1, 0, 0]

    x2, y2, z2 = 
    tag_pos2 = [x2, y2, z2 + z_offset, 0, 1, 0, 0]

    x3, y3, z3 = 
    tag_pos3 = [x3, y3, z3 + z_offset, 0, 1, 0, 0]

    x4, y4, z4 = 
    tag_pos4 = [x4, y4, z4 + z_offset, 0, 1, 0, 0]

    x5, y5, z5 = 
    tag_pos5 = [x5, y5, z5 + z_offset, 0, 1, 0, 0]

    x6, y6, z6 = 
    tag_pos6 = [x6, y6, z6 + z_offset, 0, 1, 0, 0]

    # Get an appropriate RobotTrajectory for the task (circular, linear, or square)
    # If the controller is a workspace controller, this should return a trajectory where the
    # positions and velocities are workspace positions and velocities.  If the controller
    # is a jointspace or torque controller, it should return a trajectory where the positions
    # and velocities are the positions and velocities of each joint.
    robot_trajectory1 = get_trajectory(limb, kin, ik_solver, tag_pos1, args)
    robot_trajectory2 = get_trajectory(limb, kin, ik_solver, tag_pos2, args)
    robot_trajectory3 = get_trajectory(limb, kin, ik_solver, tag_pos3, args)
    robot_trajectory4 = get_trajectory(limb, kin, ik_solver, tag_pos4, args)
    robot_trajectory5 = get_trajectory(limb, kin, ik_solver, tag_pos5, args)
    robot_trajectory6 = get_trajectory(limb, kin, ik_solver, tag_pos6, args)

    # This is a wrapper around MoveIt! for you to use.  We use MoveIt! to go to the start position
    # of the trajectory
    planner = PathPlanner('right_arm')
    
    # By publishing the trajectory to the move_group/display_planned_path topic, you should 
    # be able to view it in RViz.  You will have to click the "loop animation" setting in 
    # the planned path section of MoveIt! in the menu on the left side of the screen.
    pub = rospy.Publisher('move_group/display_planned_path', DisplayTrajectory, queue_size=10)

    # Tuck
    tuck()
    
    disp_traj = DisplayTrajectory()
    disp_traj.trajectory.append(robot_trajectory1)
    disp_traj.trajectory_start = RobotState()
    pub.publish(disp_traj)

    # Move to the trajectory start position
    plan = planner.plan_to_joint_pos(robot_trajectory1.joint_trajectory.points[0].positions)
    planner.execute_plan(plan[1])

    try:
        input('Press <Enter> to execute the trajectory using pid')
    except KeyboardInterrupt:
        sys.exit()
    # Move to the trajectory end position
    planner.execute_plan(robot_trajectory1)
    
    # Close gripper
    right_gripper.close()

    # Tuck
    tuck()

    disp_traj = DisplayTrajectory()
    disp_traj.trajectory.append(robot_trajectory2)
    disp_traj.trajectory_start = RobotState()
    pub.publish(disp_traj)

    # Move to the trajectory start position
    plan = planner.plan_to_joint_pos(robot_trajectory2.joint_trajectory.points[0].positions)
    if args.controller_name != "pid":
        plan = planner.retime_trajectory(plan, 0.3)
    planner.execute_plan(plan[1])

    try:
        input('Press <Enter> to execute the trajectory using pid')
    except KeyboardInterrupt:
        sys.exit()
    # Move to the trajectory end position
    planner.execute_plan(robot_trajectory2)

    # Open gripper
    right_gripper.open()

    # Tuck
    tuck()

if __name__ == "__main__":
    main()
