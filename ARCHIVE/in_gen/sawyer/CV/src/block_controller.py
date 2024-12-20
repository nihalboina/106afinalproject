#!/usr/bin/env python
import rospy
import numpy as np
from geometry_msgs.msg import Pose, PoseStamped
from std_msgs.msg import Float64MultiArray
import intera_interface
from intera_interface import CHECK_VERSION
import time

class BlockPIDController:
    def __init__(self):
        # Initialize the ROS node
        rospy.init_node('block_pid_controller')
        
        # Initialize Sawyer's limb interface
        self.limb = intera_interface.Limb('right')
        
        # PID gains for position
        self.Kp_pos = np.array([0.5, 0.5, 0.5])  # Proportional gains
        self.Ki_pos = np.array([0.01, 0.01, 0.01])  # Integral gains
        self.Kd_pos = np.array([0.1, 0.1, 0.1])  # Derivative gains
        
        # PID gains for orientation
        self.Kp_ori = np.array([0.3, 0.3, 0.3])
        self.Ki_ori = np.array([0.01, 0.01, 0.01])
        self.Kd_ori = np.array([0.05, 0.05, 0.05])
        
        # Error terms
        self.pos_error_sum = np.zeros(3)
        self.ori_error_sum = np.zeros(3)
        self.last_pos_error = np.zeros(3)
        self.last_ori_error = np.zeros(3)
        
        # Control loop rate (Hz)
        self.rate = rospy.Rate(100)
        
        # Subscribe to block pose updates
        rospy.Subscriber('/block_target_pose', Float64MultiArray, self.target_callback)
        
        # Current target
        self.current_target = None
        
    def target_callback(self, msg):
        # Convert the Float64MultiArray message to our target format
        data = np.array(msg.data).reshape(2, -1)
        self.current_target = data
        self.move_to_target(data[0], data[1])
    
    def move_to_target(self, target_pos, target_ori):
        while not rospy.is_shutdown():
            # Get current end effector pose
            current_pose = self.limb.endpoint_pose()
            current_pos = np.array([
                current_pose['position'].x,
                current_pose['position'].y,
                current_pose['position'].z
            ])
            current_ori = np.array([
                current_pose['orientation'].x,
                current_pose['orientation'].y,
                current_pose['orientation'].z
            ])
            
            # Calculate errors
            pos_error = target_pos - current_pos
            ori_error = target_ori - current_ori
            
            # Update integral terms
            self.pos_error_sum += pos_error
            self.ori_error_sum += ori_error
            
            # Calculate derivative terms
            pos_error_deriv = pos_error - self.last_pos_error
            ori_error_deriv = ori_error - self.last_ori_error
            
            # Calculate control signals
            pos_control = (self.Kp_pos * pos_error + 
                         self.Ki_pos * self.pos_error_sum + 
                         self.Kd_pos * pos_error_deriv)
            
            ori_control = (self.Kp_ori * ori_error + 
                         self.Ki_ori * self.ori_error_sum + 
                         self.Kd_ori * ori_error_deriv)
            
            # Combine into joint velocities
            joint_velocities = self.calculate_joint_velocities(pos_control, ori_control)
            
            # Apply joint velocities
            self.limb.set_joint_velocities(joint_velocities)
            
            # Update last error terms
            self.last_pos_error = pos_error
            self.last_ori_error = ori_error
            
            # Check if we've reached the target (within tolerance)
            if np.all(np.abs(pos_error) < 0.01) and np.all(np.abs(ori_error) < 0.01):
                rospy.loginfo("Target reached!")
                break
            
            self.rate.sleep()
    
    def calculate_joint_velocities(self, pos_control, ori_control):
        # Get the Jacobian
        jacobian = self.limb.jacobian()
        
        # Combine position and orientation control
        control = np.concatenate([pos_control, ori_control])
        
        # Calculate joint velocities using pseudo-inverse of Jacobian
        joint_velocities = np.linalg.pinv(jacobian).dot(control)
        
        # Convert to dictionary for Sawyer interface
        joint_names = self.limb.joint_names()
        joint_vel_dict = dict(zip(joint_names, joint_velocities))
        
        return joint_vel_dict

if __name__ == '__main__':
    try:
        controller = BlockPIDController()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass