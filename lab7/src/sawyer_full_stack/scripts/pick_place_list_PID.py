import sys
import argparse
import numpy as np
import rospy
import tf2_ros
from paths.trajectories import LinearTrajectory, CircularTrajectory
from paths.paths import MotionPath
from paths.path_planner import PathPlanner
from controllers.controllers import ( 
    PIDJointVelocityController, 
    FeedforwardJointVelocityController
)
from utils.utils import *

from trac_ik_python.trac_ik import IK

import intera_interface
from moveit_msgs.msg import DisplayTrajectory, RobotState
from sawyer_pykdl import sawyer_kinematics

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

def get_trajectory(limb, kin, ik_solver, target_pos, args):
    """
    Generates a trajectory for the robot to move to a target position.

    Parameters
    ----------
    limb : intera_interface.Limb
        The robot's limb interface.
    kin : sawyer_kinematics
        The kinematics interface for Sawyer.
    ik_solver : trac_ik_python.trac_ik.IK
        The IK solver.
    target_pos : list or np.array
        Target position [x, y, z].
    args : argparse.Namespace
        Command-line arguments for configuration.

    Returns
    -------
    MotionPath
        The generated trajectory.
    """
    tfBuffer = tf2_ros.Buffer()
    listener = tf2_ros.TransformListener(tfBuffer)

    try:
        trans = tfBuffer.lookup_transform('base', 'right_hand', rospy.Time(0), rospy.Duration(10.0))
    except Exception as e:
        print(e)
        print("Error retrieving current position. Using default position.")
        current_position = np.zeros(3)
    else:
        current_position = np.array([getattr(trans.transform.translation, dim) for dim in ('x', 'y', 'z')])

    print("Current Position:", current_position)
    trajectory = LinearTrajectory(start_position=current_position, goal_position=target_pos, total_time=9)
    path = MotionPath(limb, kin, ik_solver, trajectory)
    return path.to_robot_trajectory(args.num_way, True)

def get_controller(controller_name, limb, kin):
    """
    Initializes and returns the appropriate controller.

    Parameters
    ----------
    controller_name : str
        Name of the controller to use (e.g., 'pid', 'open_loop').
    limb : intera_interface.Limb
        The robot's limb interface.
    kin : sawyer_kinematics
        The kinematics interface for Sawyer.

    Returns
    -------
    Controller
        The initialized controller.
    """
    if controller_name == 'pid':
        Kp = 0.2 * np.array([0.4, 2, 1.7, 1.5, 2, 2, 3])
        Kd = 0.01 * np.array([2, 1, 2, 0.5, 0.8, 0.8, 0.8])
        Ki = 0.01 * np.array([1.4, 1.4, 1.4, 1, 0.6, 0.6, 0.6])
        Kw = np.array([0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9])
        return PIDJointVelocityController(limb, kin, Kp, Ki, Kd, Kw)
    elif controller_name == 'open_loop':
        return FeedforwardJointVelocityController(limb, kin)
    else:
        raise ValueError(f"Controller {controller_name} not recognized.")

def move_to_position(limb, kin, ik_solver, planner, target_pos, args, controller):
    """
    Moves the robot to a specific position.

    Parameters
    ----------
    limb : intera_interface.Limb
    kin : sawyer_kinematics
    ik_solver : trac_ik_python.trac_ik.IK
    planner : PathPlanner
    target_pos : list or np.array
        Target position [x, y, z].
    args : argparse.Namespace
    controller : Controller
        PID controller to execute the motion.
    """
    robot_trajectory = get_trajectory(limb, kin, ik_solver, target_pos, args)
    
    # Plan and execute the motion
    plan = planner.plan_to_joint_pos(robot_trajectory.joint_trajectory.points[0].positions)
    planner.execute_plan(plan[1])

    done = controller.execute_path(robot_trajectory, rate=args.rate, timeout=args.timeout, log=args.log)
    if not done:
        print('Failed to move to position')
        sys.exit(0)

def main():
    """
    Main function for block pick-and-place task.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-controller_name', '-c', type=str, default='pid', 
        help='Options: moveit, open_loop, pid. Default: pid'
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

    rospy.init_node('moveit_node')
    
    tuck()

    # Initialize robot interfaces
    ik_solver = IK("base", "right_gripper_tip")
    limb = intera_interface.Limb("right")
    kin = sawyer_kinematics("right")
    planner = PathPlanner('right_arm')

    controller = get_controller(args.controller_name, limb, kin)

    # Input lists of pick and place coordinates
    pick_positions = []
    place_positions = []

    n_blocks = int(input("Enter number of blocks to pick and place: "))
    print("Enter pick positions (x y z):")
    for _ in range(n_blocks):
        pick_positions.append([float(i) for i in input().split()])

    print("Enter place positions (x y z):")
    for _ in range(n_blocks):
        place_positions.append([float(i) for i in input().split()])

    if len(pick_positions) != len(place_positions):
        print("Number of pick and place positions must be equal.")
        sys.exit(1)

    # Main pick-and-place loop
    for pick_pos, place_pos in zip(pick_positions, place_positions):
        print(f"Moving to pick position: {pick_pos}")
        move_to_position(limb, kin, ik_solver, planner, pick_pos, args, controller)

        print("Closing gripper...")
        gripper = intera_interface.Gripper("right")
        gripper.close()
        rospy.sleep(1.0)

        # Adjust z-coordinate to raise the arm
        place_pos_above = place_pos.copy()
        place_pos_above[2] += 0.1

        print(f"Moving to place position (above): {place_pos_above}")
        move_to_position(limb, kin, ik_solver, planner, place_pos_above, args, controller)

        print(f"Moving to place position: {place_pos}")
        move_to_position(limb, kin, ik_solver, planner, place_pos, args, controller)

        print("Opening gripper...")
        gripper.open()
        rospy.sleep(1.0)

    print("Task completed!")

if __name__ == "__main__":
    main()
