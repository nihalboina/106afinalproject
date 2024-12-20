import cv2
import numpy as np
import json
import matplotlib.pyplot as plt

# Load the black-and-white LEGO base plate image
image_path = "/Users/rohilkhare/106afinalproject/in_gen/sawyer/CV/src/debug/1733993787.9014525.png"
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Camera matrix (updated with provided intrinsics)
K = np.array([
    [627.794983, 0.0, 360.174988],
    [0.0, 626.838013, 231.660996],
    [0.0, 0.0, 1.0]
])

# Distortion coefficients
D = np.array([-0.438799, 0.257299, 0.001038, 0.000384, -0.105028])

# Get the optimal new camera matrix
h, w = image.shape[:2]
new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(K, D, (w, h), 1, (w, h))

# Undistort the image
undistorted_image = cv2.undistort(image, K, D, None, new_camera_matrix)

# Crop the image if needed
x, y, w, h = roi
undistorted_image = undistorted_image[y:y + h, x:x + w]

# Apply GaussianBlur to reduce noise
blurred = cv2.GaussianBlur(undistorted_image, (5, 5), 0)

# Apply Sobel Edge Detection (X and Y)
sobel_x = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
sobel_y = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
sobel_combined = cv2.convertScaleAbs(sobel_x) + cv2.convertScaleAbs(sobel_y)

# Use binary thresholding on Sobel edges
_, thresh = cv2.threshold(sobel_combined, 100, 255, cv2.THRESH_BINARY)

# Find contours from the Sobel edge-detected image
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

output_image = cv2.cvtColor(undistorted_image, cv2.COLOR_GRAY2BGR)
midpoint_x, midpoint_y, angle = None, None, None
box = None

if contours:
    largest_contour = max(contours, key=cv2.contourArea)

    # Get the rotated bounding box
    rotated_rect = cv2.minAreaRect(largest_contour)
    box = cv2.boxPoints(rotated_rect)
    box = np.int0(box)
    angle = rotated_rect[-1]  # Get the angle of rotation

    # Calculate the midpoint of the rotated bounding box
    midpoint_x = int((box[0][0] + box[2][0]) / 2)
    midpoint_y = int((box[0][1] + box[2][1]) / 2)

    # Draw the rotated bounding box and midpoint on the undistorted image
    cv2.drawContours(output_image, [box], 0, (0, 255, 0), 2)
    cv2.circle(output_image, (midpoint_x, midpoint_y), 5, (0, 0, 255), -1)

    # Adjust the angle
    if angle < -45:
        angle += 90
    print(f"Corrected Computed Rotation Angle: {angle:.2f} degrees")

# Load JSON data from the GPT file
json_file_path = "/Users/rohilkhare/106afinalproject/in_gen/GPTComponent/Chatgpt_input.json"
with open(json_file_path, 'r') as f:
    json_data_file = json.load(f)

# Rotate and map the GPT blocks if valid midpoint is found
if midpoint_x is not None and midpoint_y is not None and angle is not None:
    # Convert angle to radians
    angle_rad = np.deg2rad(90 - angle)
    rotation_matrix = np.array([
        [np.cos(angle_rad), -np.sin(angle_rad)],
        [np.sin(angle_rad),  np.cos(angle_rad)]
    ])

    # Plot only the GPT-based JSON data points
    for block in json_data_file["blocks"]:
        x_original, y_original = block["position"][:2]
        # Apply rotation and scaling
        rotated_position = rotation_matrix @ np.array([x_original * 0.7, y_original * 0.75])
        x_final = int(midpoint_x + rotated_position[0])
        y_final = int(midpoint_y - rotated_position[1])

        # Set color based on the block's color attribute
        if block["color"] == "gray":
            # GPT gray blocks -> Green
            color = (0, 255, 0)
        else:
            # Other colors -> Cyan
            color = (0, 255, 255)

        # Draw the block position as a circle
        cv2.circle(output_image, (x_final, y_final), 5, color, -1)
        print(f"Block ID {block['block_id']} -> Translated (u, v): ({x_final}, {y_final})")

# Overlay the stud pattern if we have the box
if box is not None:
    sorted_box_x = sorted(box, key=lambda x: x[0])
    left_edge = sorted_box_x[0:2]

    sorted_box_y = sorted(box, key=lambda x: -x[1])
    top_edge = sorted_box_y[0:2]

    # Stud points -> Red
    stud_color = (255, 0, 0)

    for left_edge_i in range(8):
        left_edge_x = left_edge[0][0] + (left_edge[1][0] - left_edge[0][0]) * (left_edge_i + 0.5)/8
        left_edge_y = left_edge[0][1] + (left_edge[1][1] - left_edge[0][1]) * (left_edge_i + 0.5)/8

        left_edge_x += (top_edge[1][0] - top_edge[0][0]) / 16
        left_edge_y += (top_edge[1][1] - top_edge[0][1]) / 16
        cv2.circle(output_image, (int(left_edge_x), int(left_edge_y)), 5, stud_color, -1)

        for top_edge_i in range(7):
            left_edge_x += (top_edge[1][0] - top_edge[0][0]) / 8
            left_edge_y += (top_edge[1][1] - top_edge[0][1]) / 8
            cv2.circle(output_image, (int(left_edge_x), int(left_edge_y)), 5, stud_color, -1)

# Display the final result
fig, ax = plt.subplots(1, 1, figsize=(10, 10))
ax.imshow(cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB))
ax.set_title("Undistorted Image with Overlaid GPT Coordinates (Green/Cyan) and Stud Pattern (Red)")
ax.axis("off")
plt.show()
