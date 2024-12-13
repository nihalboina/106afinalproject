import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the black-and-white LEGO base plate image
# Replace with actual image path
image_path = "/Users/lilneezy/Desktop/final-project/106afinalproject/in_gen/sawyer/CV/src/debug/1733993787.9014525.png"
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

# Crop the image (if needed)
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
contours, _ = cv2.findContours(
    thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Find the largest contour (assume this is the base plate)
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
    print(f"box: {box}")
    cv2.circle(output_image, (midpoint_x, midpoint_y), 5, (0, 0, 255), -1)


# JSON data with block positions
json_data = {
    "blocks": [
        {"block_id": 1, "position": [60.0, 60.0, 0.0], "orientation": [
            0, 0, 0], "color": "gray", "placement_order": 1},
        {"block_id": 2, "position": [-60.0, 60.0, 0.0], "orientation": [
            0, 0, 0], "color": "gray", "placement_order": 2},
        {"block_id": 3, "position": [
            60.0, -60.0, 0.0], "orientation": [0, 0, 0], "color": "gray", "placement_order": 3},
        {"block_id": 4, "position": [-60.0, -60.0, 0.0], "orientation": [
            0, 0, 0], "color": "gray", "placement_order": 4},
        {"block_id": 5, "position": [0.0, 60.0, 0.0], "orientation": [
            0, 0, 0], "color": "gray", "placement_order": 5},
        {"block_id": 6, "position": [
            0.0, -60.0, 0.0], "orientation": [0, 0, 0], "color": "gray", "placement_order": 6},
        {"block_id": 7, "position": [60.0, 0.0, 0.0], "orientation": [
            0, 0, 0], "color": "gray", "placement_order": 7},
        {"block_id": 8, "position": [-60.0, 0.0, 0.0], "orientation": [
            0, 0, 0], "color": "gray", "placement_order": 8},
        {"block_id": 9, "position": [0.0, 0.0, 0.0], "orientation": [
            0, 0, 0], "color": "gray", "placement_order": 9},
        {"block_id": 10, "position": [30.0, 30.0, 40.0], "orientation": [
            0, 0, 0], "color": "yellow", "placement_order": 10},
        {"block_id": 11, "position": [-30.0, 30.0, 40.0], "orientation": [
            0, 0, 0], "color": "yellow", "placement_order": 11},
        {"block_id": 12, "position": [
            30.0, -30.0, 40.0], "orientation": [0, 0, 0], "color": "yellow", "placement_order": 12},
        {"block_id": 13, "position": [-30.0, -30.0, 40.0], "orientation": [
            0, 0, 0], "color": "yellow", "placement_order": 13},
        {"block_id": 14, "position": [0.0, 30.0, 40.0], "orientation": [
            0, 0, 0], "color": "yellow", "placement_order": 14},
        {"block_id": 15, "position": [
            0.0, -30.0, 40.0], "orientation": [0, 0, 0], "color": "yellow", "placement_order": 15},
        {"block_id": 16, "position": [30.0, 0.0, 40.0], "orientation": [
            0, 0, 0], "color": "yellow", "placement_order": 16},
        {"block_id": 17, "position": [-30.0, 0.0, 40.0], "orientation": [
            0, 0, 0], "color": "yellow", "placement_order": 17},
        {"block_id": 18, "position": [15.0, 15.0, 80.0], "orientation": [
            0, 0, 0], "color": "gray", "placement_order": 18},
        {"block_id": 19, "position": [-15.0, 15.0, 80.0], "orientation": [
            0, 0, 0], "color": "gray", "placement_order": 19},
        {"block_id": 20, "position": [
            15.0, -15.0, 80.0], "orientation": [0, 0, 0], "color": "gray", "placement_order": 20},
        {"block_id": 21, "position": [-15.0, -15.0, 80.0], "orientation": [
            0, 0, 0], "color": "gray", "placement_order": 21},
        {"block_id": 22, "position": [0.0, 0.0, 120.0], "orientation": [
            0, 0, 0], "color": "yellow", "placement_order": 22}
    ]
}


if midpoint_x is not None and midpoint_y is not None and angle is not None:
    # Adjust the angle from OpenCV's minAreaRect output
    if angle < -45:
        angle += 90
    
    print(f"Corrected Computed Rotation Angle: {angle:.2f} degrees")
    
    # Convert angle to radians
    angle_rad = np.deg2rad((90-angle))

    # Rotation matrix for 2D points
    rotation_matrix = np.array([
        [np.cos(angle_rad), -np.sin(angle_rad)],
        [np.sin(angle_rad),  np.cos(angle_rad)]
    ])

    # Rotate and plot the JSON points
    for block in json_data["blocks"]:
        # Extract original block positions
        x_original, y_original = block["position"][:2] 

        # Apply the rotation matrix
        rotated_position = rotation_matrix @ np.array([x_original * 0.7, y_original * 0.7])
        # Rotate the points around the origin
        angle_rad = np.deg2rad(angle)
        x_rotated = x_original * \
            np.cos(angle_rad) - y_original * np.sin(angle_rad)
        y_rotated = x_original * \
            np.sin(angle_rad) + y_original * np.cos(angle_rad)

        # Translate to the midpoint
        x_final = int(midpoint_x + rotated_position[0]) 
        y_final = int(midpoint_y - rotated_position[1]) # Invert y due to top-left origin
        x_final = int(midpoint_x + x_rotated)
        # Subtract because image origin is top-left
        y_final = int(midpoint_y - y_rotated)

        # Set color based on the block's color attribute
        color = (0, 255, 0) if block["color"] == "gray" else (0, 255, 255)

        # Draw the block position as a circle
        # cv2.circle(output_image, (x_final, y_final), 5, color, -1)

# start_x = min([i[0] for i in box])
# end_x = max([i[0] for i in box])
# start_y = min([i[1] for i in box])
# end_y = max([i[1] for i in box])

# for x in range(start_x, end_x, (end_x-start_x)//7):
#     for y in range(start_y, end_y, (end_y-start_y)//7):
#         cv2.circle(output_image, (int(x), int(y)), 5, color, -1)

# for box_in in box:
#     for i in range(8):
#         cv2.circle(output_image, box_in, 5, color, -1)

sorted_box = sorted(box, key=lambda x: x[0])
left_edge = sorted_box[0:2].copy()
sorted_box = sorted(box, key=lambda x: -x[1])
top_edge = sorted_box[0:2].copy()

for left_edge_i in range(8):
    left_edge_x = left_edge[0][0] + \
        (left_edge[1][0] - left_edge[0][0]) * (left_edge_i + 0.5)/8
    left_edge_y = left_edge[0][1] + \
        (left_edge[1][1] - left_edge[0][1]) * (left_edge_i + 0.5)/8

    left_edge_x += (top_edge[1][0] - top_edge[0][0])/16
    left_edge_y += (top_edge[1][1] - top_edge[0][1])/16
    cv2.circle(output_image, (int(left_edge_x),
                              int(left_edge_y)), 5, color, -1)
    for top_edge_i in range(7):
        left_edge_x += (top_edge[1][0] - top_edge[0][0])/8
        left_edge_y += (top_edge[1][1] - top_edge[0][1])/8
        cv2.circle(output_image, (int(left_edge_x),
                                  int(left_edge_y)), 5, color, -1)
print(f"sorted box: {sorted_box}")

# Display the result
fig, ax = plt.subplots(1, 1, figsize=(10, 10))
ax.imshow(cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB))
ax.set_title("Undistorted Image with Corrected Rotated JSON Points")
ax.axis("off")
plt.show()



