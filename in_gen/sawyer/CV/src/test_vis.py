#!/usr/bin/env python

import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import cv2
from stl import mesh
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import time
import os

def plot_stl_views(stl_file, output_top_down, output_top_left):
    # Load STL file
    stl_mesh = mesh.Mesh.from_file(stl_file)

    # Extract vertices for rendering
    vectors = stl_mesh.vectors
    vertices = np.concatenate(vectors)
    x, y, z = vertices[:, 0], vertices[:, 1], vertices[:, 2]

    # Define the views
    views = [
        {"elev": 90, "azim": 0, "output": output_top_down},  # Top-down view
        {"elev": 45, "azim": -45, "output": output_top_left},  # Top-left corner view
    ]

    for view in views:
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection='3d')

        # Create a 3D plot
        collection = Poly3DCollection(vectors, alpha=0.7, edgecolor='k')
        ax.add_collection3d(collection)

        # Set plot limits
        ax.set_xlim(np.min(x), np.max(x))
        ax.set_ylim(np.min(y), np.max(y))
        ax.set_zlim(np.min(z), np.max(z))

        # Set the view angle
        ax.view_init(elev=view["elev"], azim=view["azim"])

        # Save the view as a JPG
        plt.axis('off')  # Turn off the axis
        plt.savefig(view["output"], bbox_inches='tight', dpi=300)
        plt.close()

def setup():
    # run through all objects in the folder, and generate views, place into the generated_views folder
    path_to_objects = "/home/cc/ee106a/fa24/class/ee106a-aei/final_project/106afinalproject/in_gen/sawyer/CV/src/object_stls"
    path_to_views = "/home/cc/ee106a/fa24/class/ee106a-aei/final_project/106afinalproject/in_gen/sawyer/CV/src/generated_views"
    files_and_objects = os.listdir(path_to_objects)

    for obj in files_and_objects:
        wout_stl = obj.replace('.stl','')
        full_obj_path = f"{path_to_objects}/{obj}"
        output_top_down = f"{path_to_views}/{wout_stl}_top_down_view.png"
        output_top_left = f"{path_to_views}/{wout_stl}_top_left_corner_view.png"    
        print(obj)
        if not (os.path.exists(output_top_down) and os.path.exists(output_top_left)):
            plot_stl_views(full_obj_path, output_top_down, output_top_left)

def run_cv(img, number_objects=1):
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian Blur
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Apply Thresholding
    _, thresholded = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY)

    # Find contours
    contours, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Define a function to check if a bounding box is square-ish
    def is_squareish(w, h):
        ratio = max(w, h) / min(w, h)
        return ratio <= 1.33  # Allow up to 33% difference

    # Initialize variables to track the largest square-ish object
    rectangles = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if is_squareish(w, h):
            # Store rectangle information: top-left, bottom-right, color, thickness, area
            rectangle = [(x, y), (x + w, y + h), (0, 255, 0), 2, w * h]
            rectangles.append(rectangle)
    
    # Sort rectangles by area in descending order
    rectangles = sorted(rectangles, key=lambda x: -x[-1])

    # Draw the top 'number_objects' rectangles
    for rect_ind in range(min(len(rectangles), number_objects)):
        rect_info = rectangles[rect_ind]
        cv2.rectangle(img, rect_info[0], rect_info[1], rect_info[2], rect_info[3])

    return img





def image_callback(msg):
    rospy.loginfo("Image callback triggered")
    try:
        # Convert the ROS Image message to a CV image
        cv_image = bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        
        # Save the image to verify if display fails
        # cv2.imwrite("debug_output.jpg", cv_image)
        # rospy.loginfo("Saved debug image as debug_output.jpg")

        # run_cv on it

        # cv_image = run_cv(cv_image)

        # Display the image in a window
        cv2.imshow("Right Hand Camera", cv_image)
        cv2.imwrite(f"/home/cc/ee106a/fa24/class/ee106a-aei/final_project/106afinalproject/in_gen/sawyer/CV/src/debug/{time.time()}.png",cv_image)
        cv2.waitKey(1)  # Update the window
    except CvBridgeError as e:
        rospy.logerr(f"Could not convert image: {e}")

if __name__ == "__main__":
    # Initialize the ROS node
    rospy.init_node("right_hand_camera_viewer", anonymous=True)

    # Setup

    setup()
    
    # Create a CvBridge instance
    bridge = CvBridge()
    
    # Subscribe to the right_hand_camera image topic
    rospy.Subscriber("/io/internal_camera/right_hand_camera/image_raw", Image, image_callback)
    
    rospy.loginfo("Right Hand Camera viewer node started.")
    
    # Keep the program running and processing callbacks
    rospy.spin()

    # Destroy all OpenCV windows when done
    cv2.destroyAllWindows()
