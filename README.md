GPT Component contains files related to the GPT prompting and output model visualizer
rogo-main contains files related to the CV block recognition and coordinate transformation, the json outputs from the GPT and CV, as well as action_server.py which interprets those json coordinates and actually actuates the Sawyer to perform our building tasks


To run:
1) connect to sawyer
2) run intera interface, action server commands
3) run cv_server.py and action_server.py in rogo-main/src

Brief overview:

in_gen (instruction generation)

--everything that goes from taking in the user input, image input to get to how to get to placing instructions

1. output generation (using gpt to get output csv of some sorts)

2. input mapping (using cv full position + orientation of all blocks)

instruction generation (some algo stuff to put combine 1. + 2. to get from list of input poses to output poses)


in_exec (instruction execution)

PID stuff to go from all instructions to executing properly


Startup instructions:

Connect to sawyer

source ~ee106a/sawyer_setup.bash

Enable sawyer

rosrun intera_interface enable_robot.py -e

Start the intera action server

rosrun intera_interface joint_trajectory_action_server.py

Start MoveIt to enable inverse kinematics and usage of the MoveIt controller

roslaunch sawyer_moveit_config sawyer_moveit.launch electric_gripper:=true

Place sawyer in a good starting joint configuration

roslaunch intera_examples sawyer_tuck.launch

Other instructions:

Get current gripper position

rosrun tf tf_echo base right_gripper_tip

Or for 'Amir'

rosrun tf tf_echo base stp_022312TP99620
