import cv2
import numpy as np
import matplotlib.pyplot as plt
from stl import mesh
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


def plot_stl_views(stl_file, output_image):
    # Load STL file
    stl_mesh = mesh.Mesh.from_file(stl_file)

    # Extract vertices for rendering
    vectors = stl_mesh.vectors
    vertices = np.concatenate(vectors)
    x, y, z = vertices[:, 0], vertices[:, 1], vertices[:, 2]

    # Create a 3D plot for the STL model
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
    ax.view_init(elev=45, azim=-45)

    # Save the view as a JPG
    plt.axis('off')  # Turn off the axis
    plt.savefig(output_image, bbox_inches='tight', dpi=300)
    plt.close()


def process_video_stream():
    # Initialize video capture
    cap = cv2.VideoCapture(0)  # Use 0 for default camera

    # Load STL model image (preprocessed)
    stl_image_path = 'stl_image.jpg'
    stl_file = 'block.stl'  # Replace with your STL file path
    plot_stl_views(stl_file, stl_image_path)
    stl_image = cv2.imread(stl_image_path, cv2.IMREAD_GRAYSCALE)

    # Initialize ORB detector and BFMatcher
    orb = cv2.ORB_create()
    keypoints_stl, descriptors_stl = orb.detectAndCompute(stl_image, None)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    # Define a function to check if a bounding box is square-ish
    def is_squareish(w, h):
        ratio = max(w, h) / min(w, h)
        return ratio <= 1.33  # Allow up to 33% difference

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        # Get frame dimensions
        frame_height, frame_width = frame.shape[:2]
        min_area = (frame_width * frame_height) / \
            100  # 1/100th of total screen size

        # Convert frame to grayscale
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Apply edge detection using the Canny method
        edges = cv2.Canny(image, 100, 200)

        # Find contours in the edged image
        contours, _ = cv2.findContours(
            edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Convert the grayscale image to BGR for drawing colored bounding boxes
        output = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

        # Counter for labeling
        label_count = 1

        # Process each contour
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            area = w * h

            if is_squareish(w, h) and area >= min_area:
                # Crop the detected block
                cropped = image[y:y+h, x:x+w]

                # Detect keypoints and descriptors in the cropped image
                keypoints_real, descriptors_real = orb.detectAndCompute(
                    cropped, None)

                if descriptors_real is not None and descriptors_stl is not None:
                    # Match features using the Brute-Force matcher
                    matches = bf.match(descriptors_real, descriptors_stl)
                    matches = sorted(matches, key=lambda x: x.distance)

                    # Find homography if sufficient matches exist
                    if len(matches) > 10:
                        src_pts = np.float32(
                            [keypoints_real[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
                        dst_pts = np.float32(
                            [keypoints_stl[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

                        # Compute homography matrix
                        H, mask = cv2.findHomography(
                            src_pts, dst_pts, cv2.RANSAC, 5.0)
                        if H is not None:
                            # Decompose the homography matrix to extract rotation and translation
                            num_solutions, Rs, ts, normals = cv2.decomposeHomographyMat(
                                H, np.eye(3))

                            # We'll use the first solution as an example
                            R = Rs[0]
                            t = ts[0]

                            # Convert rotation matrix to Euler angles (Rx, Ry, Rz)
                            def rotationMatrixToEulerAngles(R):
                                sy = np.sqrt(
                                    R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
                                singular = sy < 1e-6

                                if not singular:
                                    x_angle = np.arctan2(R[2, 1], R[2, 2])
                                    y_angle = np.arctan2(-R[2, 0], sy)
                                    z_angle = np.arctan2(R[1, 0], R[0, 0])
                                else:
                                    x_angle = np.arctan2(-R[1, 2], R[1, 1])
                                    y_angle = np.arctan2(-R[2, 0], sy)
                                    z_angle = 0

                                return np.degrees(np.array([x_angle, y_angle, z_angle]))

                            Rx, Ry, Rz = rotationMatrixToEulerAngles(R)

                            # Draw the bounding box
                            cv2.rectangle(output, (x, y), (x + w, y + h),
                                          (0, 255, 0), 2)  # Green boxes

                            # Put a label number
                            cv2.putText(output, f'Block {label_count}', (x, y - 30),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                            # Add text annotations for position and orientation
                            text = f"Pos: [{t[0][0]:.2f}, {t[1][0]:.2f}, {t[2][0]:.2f}]"
                            cv2.putText(output, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
                                        0.5, (0, 255, 0), 2)

                            text = f"Ori: Rx={Rx:.1f}, Ry={Ry:.1f}, Rz={Rz:.1f}"
                            cv2.putText(output, text, (x, y + h + 15), cv2.FONT_HERSHEY_SIMPLEX,
                                        0.5, (0, 255, 0), 2)

                            label_count += 1
                        else:
                            # If homography fails, just draw the bounding box
                            cv2.rectangle(output, (x, y), (x + w, y + h),
                                          (0, 0, 255), 2)  # Red box for failure
                            cv2.putText(output, f'Block {label_count} - Pose Est. Failed',
                                        (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
                                        0.5, (0, 0, 255), 2)
                            label_count += 1
                    else:
                        # Not enough matches
                        cv2.rectangle(output, (x, y), (x + w, y + h),
                                      (0, 0, 255), 2)  # Red box for failure
                        cv2.putText(output, f'Block {label_count} - Not enough matches',
                                    (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
                                    0.5, (0, 0, 255), 2)
                        label_count += 1

        # Display the output frame
        cv2.imshow('Blocks Detection', output)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the capture and destroy all windows
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    process_video_stream()
