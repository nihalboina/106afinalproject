import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the black-and-white LEGO base plate image
image_path = "/Users/rohilkhare/106afinalproject/in_gen/sawyer/CV/src/debug/1733993790.4854965.png"  # Replace with actual image path
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Apply GaussianBlur to reduce noise
blurred = cv2.GaussianBlur(image, (5, 5), 0)

# Apply Sobel Edge Detection (X and Y)
sobel_x = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
sobel_y = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
sobel_combined = cv2.convertScaleAbs(sobel_x) + cv2.convertScaleAbs(sobel_y)

# Use binary thresholding on Sobel edges
_, thresh = cv2.threshold(sobel_combined, 100, 255, cv2.THRESH_BINARY)

# Find contours from the Sobel edge-detected image
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Find the largest contour (assume this is the base plate)
if contours:
    largest_contour = max(contours, key=cv2.contourArea)
    
    # Get the rotated bounding box
    rotated_rect = cv2.minAreaRect(largest_contour)
    box = cv2.boxPoints(rotated_rect)
    box = np.int0(box)

    # Calculate the midpoint of the rotated bounding box
    midpoint_x = int((box[0][0] + box[2][0]) / 2)
    midpoint_y = int((box[0][1] + box[2][1]) / 2)

    # Draw the rotated bounding box and midpoint on the original image
    output_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(output_image, [box], 0, (0, 255, 0), 2)
    cv2.circle(output_image, (midpoint_x, midpoint_y), 5, (0, 0, 255), -1)

    # Display results
    fig, ax = plt.subplots(1, 2, figsize=(15, 7))
    ax[0].imshow(cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB))
    ax[0].set_title(f"Detected Plate with Midpoint ({midpoint_x}, {midpoint_y})")

    ax[1].imshow(sobel_combined, cmap="gray")
    ax[1].set_title("Sobel Edges")

    for a in ax:
        a.axis("off")
    plt.show()
else:
    print("No contours detected.")
