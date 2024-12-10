#!/usr/bin/env python

import rospy
from CV.msg import Blocks, Block  # Replace 'your_package' with your package name
from geometry_msgs.msg import Pose

def generate_fake_blocks():
    # Create a list of Block messages
    blocks = []

    # Example: Generate 3 fake blocks
    for i in range(3):
        block = Block()
        
        # Set pose (position and orientation)
        block.pose.position.x = i * 0.1
        block.pose.position.y = i * 0.2
        block.pose.position.z = i * 0.3
        block.pose.orientation.x = 0.0  # Replace with actual rx
        block.pose.orientation.y = 0.0  # Replace with actual ry
        block.pose.orientation.z = 0.0  # Replace with actual rz
        block.pose.orientation.w = 1.0  # Replace with actual rw (quaternion w)
        
        # Set classification
        block.classification = f"block_{i}.stl" if i % 2 == 0 else "input/output"
        
        # Set confidence
        block.confidence = 90.0 - i * 10.0  # Example confidence
        
        blocks.append(block)

    return blocks

def blocks_publisher():
    # Initialize the ROS node
    rospy.init_node('blocks_publisher', anonymous=True)

    # Create a publisher for /blocks
    pub = rospy.Publisher('/blocks', Blocks, queue_size=10)

    # Set the publishing rate
    rate = rospy.Rate(1)  # 1 Hz

    while not rospy.is_shutdown():
        # Create the Blocks message
        blocks_msg = Blocks()
        blocks_msg.blocks = generate_fake_blocks()

        # Log the message
        rospy.loginfo(f"Publishing blocks: {blocks_msg}")

        # Publish the message
        pub.publish(blocks_msg)

        # Sleep for the remainder of the loop
        rate.sleep()

if __name__ == '__main__':
    try:
        blocks_publisher()
    except rospy.ROSInterruptException:
        pass
