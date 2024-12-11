import numpy as np
import cv2

# Camera calibration parameters
camera_calibration = {
    'fx': 627.794983,
    'fy': 626.838013,
    'cx': 360.174988,
    'cy': 231.660996,
    'dist_coeffs': np.array([-0.438799, 0.257299, 0.001038, 0.000384, -0.105028]),
    'width': 752,
    'height': 480
}


def undistort_point(u, v, camera_calibration):
    """
    Undistort a single image point using OpenCV.

    Args:
        u (float): Original x pixel coordinate.
        v (float): Original y pixel coordinate.
        camera_calibration (dict): Dictionary containing camera calibration parameters.

    Returns:
        (float, float): Undistorted pixel coordinates.
    """
    # Prepare the camera matrix
    K = np.array([
        [camera_calibration['fx'], 0, camera_calibration['cx']],
        [0, camera_calibration['fy'], camera_calibration['cy']],
        [0, 0, 1]
    ])

    # Distortion coefficients
    D = camera_calibration['dist_coeffs']

    # OpenCV expects points in the form [[u, v]]
    distorted_point = np.array([[u, v]], dtype=np.float32)

    # Undistort the point
    undistorted = cv2.undistortPoints(distorted_point, K, D, P=K)

    return undistorted[0][0][0], undistorted[0][0][1]


def get_real_world_coordinates(image_x, image_y, fixed_z=-0.108):
    """
    Convert image pixel coordinates to real-world coordinates with a fixed z-coordinate.

    Args:
        image_x (float): X coordinate in the image.
        image_y (float): Y coordinate in the image.
        fixed_z (float): Fixed z-coordinate in the world frame.

    Returns:
        (float, float, float): Real-world coordinates (x, y, z).
    """
    # Step 1: Undistort the image point
    undistorted_x, undistorted_y = undistort_point(
        image_x, image_y, camera_calibration)

    # Step 2: Normalize the undistorted coordinates
    x_normalized = (
        undistorted_x - camera_calibration['cx']) / camera_calibration['fx']
    y_normalized = (
        undistorted_y - camera_calibration['cy']) / camera_calibration['fy']

    # Step 3: Scale by the fixed z to get real-world coordinates
    real_x = x_normalized * fixed_z
    real_y = y_normalized * fixed_z
    real_z = fixed_z

    print(f"Image Coordinates: ({image_x}, {image_y})")
    print(f"Undistorted Coordinates: ({undistorted_x}, {undistorted_y})")
    print(f"Normalized Coordinates: ({x_normalized}, {y_normalized})")
    print(f"Real-World Coordinates: ({real_x}, {real_y}, {real_z})")

    return real_x, real_y, real_z


# Example usage with the known correspondence
# Known image point (450, 18) maps to real-world point (0.754, 0.018, -0.108)
# Use this to validate the scaling
real_world = get_real_world_coordinates(450, 18)
print("Computed Real-World Coordinates:", real_world)

# You can add more points and perform validation or calibration as needed
