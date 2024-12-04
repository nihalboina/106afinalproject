import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from stl import mesh
import numpy as np


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


# Example usage
stl_file = "block.stl"  # Replace with your STL file path
output_top_down = "top_down_view.jpg"
output_top_left = "top_left_corner_view.jpg"

plot_stl_views(stl_file, output_top_down, output_top_left)


def process_image(input_image, output_image, cropped_image):
    # Load the grayscale image
    image = cv2.imread(input_image, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print("Error: Could not load image.")
        return

    # Apply edge detection using the Canny method
    edges = cv2.Canny(image, 100, 200)

    # Find contours in the edged image
    contours, _ = cv2.findContours(
        edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Convert the grayscale image to BGR for drawing colored bounding boxes
    output = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    # Define a function to check if a bounding box is square-ish
    def is_squareish(w, h):
        ratio = max(w, h) / min(w, h)
        return ratio <= 1.33  # Allow up to 33% difference

    # Initialize variables to track the largest square-ish object
    largest_square = None
    max_area = 0

    # Draw bounding boxes around square-ish contours and find the largest
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if is_squareish(w, h):
            # Draw the bounding box
            cv2.rectangle(output, (x, y), (x + w, y + h),
                          (0, 255, 0), 2)  # Green boxes

            # Update the largest square-ish object
            area = w * h
            if area > max_area:
                max_area = area
                largest_square = (x, y, w, h)

    # Save the output image with bounding boxes
    cv2.imwrite(output_image, output)
    print(f"Output saved as {output_image}")

    # Crop and save the largest square-ish object, if found
    if largest_square:
        x, y, w, h = largest_square
        cropped = image[y:y + h, x:x + w]
        cv2.imwrite(cropped_image, cropped)
        print(f"Cropped image saved as {cropped_image}")
    else:
        print("No square-ish objects found for cropping.")


# Input and output file names
input_filename = "image.jpg"
output_filename = "output.jpg"
cropped_filename = "cropped.jpg"

# Process the image
process_image(input_filename, output_filename, cropped_filename)


# now (assuming we have multiple stls, ask gpt which one it is)

# since we have only one right now, just assume it as block.stl
selected_stl = "block.stl"


# Load the images
real_life_image_path = 'cropped.jpg'  # Real-life image
stl_image_path = 'top_left_corner_view.jpg'  # STL top-left corner view

real_life_image = cv2.imread(real_life_image_path, cv2.IMREAD_GRAYSCALE)
stl_image = cv2.imread(stl_image_path, cv2.IMREAD_GRAYSCALE)

# Detect keypoints and descriptors using ORB (Oriented FAST and Rotated BRIEF)
orb = cv2.ORB_create()
keypoints_real, descriptors_real = orb.detectAndCompute(real_life_image, None)
keypoints_stl, descriptors_stl = orb.detectAndCompute(stl_image, None)

# Match features using the Brute-Force matcher
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = bf.match(descriptors_real, descriptors_stl)
matches = sorted(matches, key=lambda x: x.distance)

# Draw matches for visualization
matched_image = cv2.drawMatches(
    real_life_image, keypoints_real, stl_image, keypoints_stl, matches[:20], None, flags=2)

# Find homography if sufficient matches exist
if len(matches) > 10:
    src_pts = np.float32(
        [keypoints_real[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32(
        [keypoints_stl[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    # Compute homography matrix
    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    if H is not None:
        # Decompose the homography matrix to extract rotation and translation
        _, R, t, _ = cv2.decomposeHomographyMat(H, np.eye(3))
    else:
        R, t = None, None
else:
    H, R, t = None, None, None

# Save matched image for inspection
matched_image_path = 'matched_features.jpg'
cv2.imwrite(matched_image_path, matched_image)

# Output results
print(matched_image_path, H, R, t)
