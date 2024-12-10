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

def run_cv(img, number_objects=3):
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian Blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Apply Otsu's Thresholding
    _, thresholded = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Save the thresholded image for debugging
    cv2.imwrite(f"/path/to/debug/thresholded_{time.time()}.png", thresholded)

    # Apply Morphological Operations to remove small noise
    kernel = np.ones((3, 3), np.uint8)
    thresholded = cv2.morphologyEx(thresholded, cv2.MORPH_OPEN, kernel, iterations=2)

    # Find contours
    contours, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Define a function to check if a bounding box is square-ish
    def is_squareish(w, h):
        ratio = max(w, h) / min(w, h)
        return ratio <= 1.5  # Adjusted to allow more variance

    # Initialize variables to track the largest square-ish objects
    rectangles = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if is_squareish(w, h):
            # Store rectangle information
            rectangle = [(x, y), (x + w, y + h), (0, 255, 0), 2, w * h]
            rectangles.append(rectangle)

    print(f"Detected rectangles: {rectangles}")
    
    # Sort rectangles by area in descending order
    rectangles = sorted(rectangles, key=lambda x: -x[-1])

    orb = cv2.ORB_create()

    stl_image_path = "/home/cc/ee106a/fa24/class/ee106a-aei/final_project/106afinalproject/in_gen/sawyer/CV/src/generated_views/block_top_left_corner_view.png"
    stl_image = cv2.imread(stl_image_path, cv2.IMREAD_GRAYSCALE)
    if stl_image is None:
        raise ValueError(f"Could not load STL reference image from {stl_image_path}")
    
    keypoints_stl, descriptors_stl = orb.detectAndCompute(stl_image, None)
    if descriptors_stl is None or len(keypoints_stl) == 0:
        raise ValueError("No keypoints found in the STL reference image.")

    # Process each rectangle
    for rect_info in rectangles[:number_objects]:
        print(f"rect_info: {rect_info}")
        # Corrected slicing
        isolate_rectangle = img[rect_info[0][1]:rect_info[1][1], rect_info[0][0]:rect_info[1][0]]
        keypoints_real, descriptors_real = orb.detectAndCompute(isolate_rectangle, None)
        
        if descriptors_real is None or len(keypoints_real) == 0:
            print("No keypoints found in the isolated rectangle. Skipping to next rectangle.")
            continue
        
        # Match descriptors
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(descriptors_real, descriptors_stl)
        if len(matches) < 4:
            print("Not enough matches to compute homography. Skipping to next rectangle.")
            continue

        # Proceed with homography computation
        matches = sorted(matches, key=lambda x: x.distance)
        src_pts = np.float32([keypoints_real[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([keypoints_stl[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

        # Compute homography matrix
        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        if H is not None:
            # Decompose the homography matrix
            num_solutions, Rs, ts, normals = cv2.decomposeHomographyMat(H, np.eye(3))
            print(f"Number of solutions from homography decomposition: {num_solutions}")

            # Use the first solution
            R = Rs[0]
            t = ts[0]
            normal = normals[0]

            # Convert rotation matrix to Euler angles
            def rotationMatrixToEulerAngles(R):
                sy = np.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
                singular = sy < 1e-6

                if not singular:
                    x = np.arctan2(R[2, 1], R[2, 2])
                    y = np.arctan2(-R[2, 0], sy)
                    z = np.arctan2(R[1, 0], R[0, 0])
                else:
                    x = np.arctan2(-R[1, 2], R[1, 1])
                    y = np.arctan2(-R[2, 0], sy)
                    z = 0

                return np.degrees(np.array([x, y, z]))

            Rx, Ry, Rz = rotationMatrixToEulerAngles(R)
            print(f"\nOrientation (Euler angles): Rx={Rx:.2f}, Ry={Ry:.2f}, Rz={Rz:.2f}")
            print(f"Position (translation vector): t={t.flatten()}")
            font = cv2.FONT_HERSHEY_SIMPLEX
            text = f"Position: t = [{t[0][0]:.2f}, {t[1][0]:.2f}, {t[2][0]:.2f}]"
            cv2.putText(img, text, (rect_info[0][0], rect_info[0][1] - 10), font, 0.7, (0, 0, 255), 2)

            text = f"Orientation: Rx={Rx:.2f}, Ry={Ry:.2f}, Rz={Rz:.2f}"
            cv2.putText(img, text, (rect_info[0][0], rect_info[0][1] + 20), font, 0.7, (0, 0, 255), 2)

            cv2.rectangle(img, rect_info[0], rect_info[1], rect_info[2], rect_info[3])
        else:
            print("Homography computation failed. Skipping to next rectangle.")

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

        cv_image = run_cv(cv_image)

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