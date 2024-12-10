#!/usr/bin/env python

import rospy
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Header
from sensor_msgs.msg import JointState
from intera_core_msgs.srv import SolvePositionFK, SolvePositionFKRequest
from moveit_commander import MoveGroupCommander


def fk_to_cartesian(joint_angles):
    service_name = "ExternalTools/right/PositionKinematicsNode/FKService"
    fk_service_proxy = rospy.ServiceProxy(service_name, SolvePositionFK)
    fk_request = SolvePositionFKRequest()

    joints = JointState()
    joints.name = ['right_j0', 'right_j1', 'right_j2', 'right_j3', 'right_j4', 'right_j5', 'right_j6']
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

    group.set_max_velocity_scaling_factor(1.0)  # Max speed (use a value < 1.0 to scale down)
    group.set_max_acceleration_scaling_factor(1.0)  # Max acceleration


    group.set_pose_target(pose)

    # Plan the motion
    plan = group.plan()

    # Check if planning was successful
    if isinstance(plan, tuple):
        trajectory = plan[1]  # Extract the trajectory from the tuple
    else:
        trajectory = plan

    user_input = input("Enter 'y' if the trajectory looks safe on RViz: ")

    if user_input.lower() == 'y':
        group.execute(trajectory, wait=True)  # Pass the extracted trajectory
        rospy.loginfo("Movement executed successfully.")
    else:
        rospy.loginfo("Movement aborted by the user.")

def main():
    rospy.init_node("sawyer_fk_movement")

    while not rospy.is_shutdown():
        input("Press [Enter] to provide joint angles...")
        joint_angles_input = input("Enter joint angles (comma-separated): ")
        try:
            joint_angles = [float(angle) for angle in joint_angles_input.split(",")]
        except ValueError:
            rospy.logerr("Invalid input. Please enter comma-separated numbers.")
            continue

        if len(joint_angles) != 7:
            rospy.logerr("You must provide exactly 7 joint angles.")
            continue

        end_effector_pose = fk_to_cartesian(joint_angles)
        if end_effector_pose:
            rospy.loginfo(f"Computed Cartesian Pose: {end_effector_pose}")
            move_to_position(end_effector_pose)
        else:
            rospy.logerr("Failed to compute a valid Cartesian pose.")


if __name__ == '__main__':
    main()
