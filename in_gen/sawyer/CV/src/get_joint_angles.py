#!/usr/bin/env python

import rospy
from intera_interface import Limb

def get_current_joint_angles():
    # Initialize the ROS node
    rospy.init_node('get_joint_angles', anonymous=True)

    # Initialize the Limb interface for the right arm
    limb = Limb('right')

    # Retrieve the current joint angles
    joint_angles = limb.joint_angles()

    # Print the joint angles
    print("Current Joint Angles for Sawyer's Right Arm:")
    for joint, angle in joint_angles.items():
        print(f"{joint}: {angle:.4f} radians")

    print([angle for joint,angle in joint_angles.items()])

if __name__ == '__main__':
    try:
        get_current_joint_angles()
    except rospy.ROSInterruptException:
        pass
