#!/usr/bin/env python
import rospy
from std_msgs.msg import Float64MultiArray
import time

def test_block_movement():
    # Initialize the node
    rospy.init_node('block_controller_test', anonymous=True)
    
    # Create publisher
    pose_publisher = rospy.Publisher('/block_target_pose', Float64MultiArray, queue_size=10)
    
    # Wait for publisher to be ready
    time.sleep(1)
    
    # Test cases
    test_cases = [
        # Test case 1: Simple forward movement
        {
            'current': [[15355.85, 10108.16, 9.47], [-0.28, 0.10, 168.03]],
            'target': [[17455.85, 12148.16, 10.47], [-0.34, 0.24, 148.04]]
        },
        # Test case 2: Lateral movement
        {
            'current': [[17455.85, 12148.16, 10.47], [-0.34, 0.24, 148.04]],
            'target': [[17455.85, 14148.16, 10.47], [-0.34, 0.24, 148.04]]
        },
        # Test case 3: Vertical movement
        {
            'current': [[17455.85, 14148.16, 10.47], [-0.34, 0.24, 148.04]],
            'target': [[17455.85, 14148.16, 20.47], [-0.34, 0.24, 148.04]]
        }
    ]
    
    # Run through test cases
    for i, test_case in enumerate(test_cases):
        rospy.loginfo(f"Running test case {i + 1}")
        
        # Create message
        msg = Float64MultiArray()
        msg.data = (test_case['current'][0] + 
                   test_case['current'][1] + 
                   test_case['target'][0] + 
                   test_case['target'][1])
        
        # Publish message
        pose_publisher.publish(msg)
        rospy.loginfo(f"Published test case {i + 1}")
        
        # Wait between test cases
        time.sleep(5)  # Adjust this based on how long your movements take

if __name__ == '__main__':
    try:
        test_block_movement()
        rospy.loginfo("All test cases completed")
    except rospy.ROSInterruptException:
        pass 