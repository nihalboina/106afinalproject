import cv2
import numpy as np
from stl import mesh
import os
from datetime import datetime

# Load the STL file
your_mesh = mesh.Mesh.from_file('block.stl')

# Camera intrinsic parameters (assuming known or calibrated)
camera_matrix = np.array([[800, 0, 320],
                          [0, 800, 240],
                          [0,   0,   1]], dtype=np.float64)
dist_coeffs = np.zeros((4, 1))  # Assuming no lens distortion

# Define 3D model points of the block's corners
# Adjust these points based on the actual dimensions of your block.stl
model_points = np.array([
    [-0.5, -0.5, 0],  # Top-left-front
    [0.5, -0.5, 0],   # Top-right-front
    [0.5, 0.5, 0],    # Bottom-right-front
    [-0.5, 0.5, 0],   # Bottom-left-front
], dtype="double")

# Initialize the camera
cap = cv2.VideoCapture(0)

# Load YOLO network
net = cv2.dnn.readNet('yolov3.weights', 'yolov3.cfg')

# Load the classes (Ensure 'book' is included in the coco.names file)
classes = []
with open('coco.names', 'r') as f:
    classes = [line.strip() for line in f.readlines()]

layer_names = net.getLayerNames()
output_layers = [layer_names[int(i) - 1]
                 for i in net.getUnconnectedOutLayers()]

# Create a directory to save images if it doesn't exist
output_dir = "detected_frames"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    height, width, channels = frame.shape

    # Prepare the frame for YOLO detection
    blob = cv2.dnn.blobFromImage(
        frame, scalefactor=1/255.0, size=(416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    # Initialize lists for detected bounding boxes, confidences, and class IDs
    class_ids = []
    confidences = []
    boxes = []

    # Process YOLO detections
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            # Adjust the confidence threshold and class name as needed
            # Changed 'block' to 'book'
            if confidence > 0.5 and classes[class_id] == 'book':
                # Object detected
                center_x = int(detection[0]*width)
                center_y = int(detection[1]*height)
                w = int(detection[2]*width)
                h = int(detection[3]*height)
                x = int(center_x - w/2)
                y = int(center_y - h/2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Apply Non-Max Suppression
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    # Flag to indicate if an object was detected in this frame
    object_detected = False

    # If the object is detected, estimate its pose
    if len(indexes) > 0:
        object_detected = True  # Set the flag
        for i in indexes.flatten():
            x, y, w, h = boxes[i]
            # Draw red bounding box
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)

            # Calculate center point of the bounding box
            center_x_box = x + w / 2
            center_y_box = y + h / 2

            # Display the 2D coordinates on the image
            cv2.putText(frame, f"2D Coordinates: x={int(center_x_box)}, y={int(center_y_box)}",
                        (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            # Define 2D image points from the bounding box corners
            image_points = np.array([
                [x, y],          # Top-left corner
                [x + w, y],      # Top-right corner
                [x + w, y + h],  # Bottom-right corner
                [x, y + h],      # Bottom-left corner
            ], dtype="double")

            # Estimate pose using solvePnP
            success, rotation_vector, translation_vector = cv2.solvePnP(
                model_points, image_points, camera_matrix, dist_coeffs)

            if success:
                # Project model points to image plane for visualization
                (proj_points, _) = cv2.projectPoints(
                    model_points, rotation_vector, translation_vector, camera_matrix, dist_coeffs)
                for p in proj_points:
                    cv2.circle(frame, (int(p[0][0]), int(
                        p[0][1])), 5, (0, 255, 0), -1)  # Green dots for projected points

                # Convert rotation vector to rotation matrix
                rotation_matrix, _ = cv2.Rodrigues(rotation_vector)

                # Calculate Euler angles from rotation matrix
                sy = np.sqrt(rotation_matrix[0, 0]
                             ** 2 + rotation_matrix[1, 0] ** 2)
                singular = sy < 1e-6

                if not singular:
                    x_angle = np.arctan2(
                        rotation_matrix[2, 1], rotation_matrix[2, 2])
                    y_angle = np.arctan2(-rotation_matrix[2, 0], sy)
                    z_angle = np.arctan2(
                        rotation_matrix[1, 0], rotation_matrix[0, 0])
                else:
                    x_angle = np.arctan2(-rotation_matrix[1, 2],
                                         rotation_matrix[1, 1])
                    y_angle = np.arctan2(-rotation_matrix[2, 0], sy)
                    z_angle = 0

                # Convert radians to degrees
                x_angle = np.degrees(x_angle)
                y_angle = np.degrees(y_angle)
                z_angle = np.degrees(z_angle)

                # Display predicted x, y, z and orientation
                position = translation_vector.flatten()
                cv2.putText(frame, f"Position: x={position[0]:.2f}, y={position[1]:.2f}, z={position[2]:.2f}",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
                cv2.putText(frame, f"Orientation: x={x_angle:.2f}, y={y_angle:.2f}, z={z_angle:.2f}",
                            (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
            else:
                cv2.putText(frame, "Pose estimation failed", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    # Save the frame if an object was detected
    if object_detected:
        # Get the current timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        # Create the filename
        filename = f"{output_dir}/detected_{timestamp}.png"
        # Save the frame as a PNG image
        cv2.imwrite(filename, frame)
        print(f"Saved frame: {filename}")

    # Display the frame
    cv2.imshow('frame', frame)
    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
