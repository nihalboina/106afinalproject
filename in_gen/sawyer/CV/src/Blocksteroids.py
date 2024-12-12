import numpy as np

def image_to_world(u, v, depth, K, R, t):
    """Converts image pixel (u, v) to world coordinates.

    Args:
        u, v: Image pixel coordinates
        depth: Depth value (Z_c) from the camera
        K: Intrinsic matrix (3x3)
        R: Rotation matrix (3x3)
        t: Translation vector (3x1)

    Returns:
        World coordinates (X_w, Y_w, Z_w) as a 1D array.
        Returns None if there's an issue with matrix inversion.
    """
    uv1 = np.array([u, v, 1]).reshape((3, 1))

    try:
        K_inv = np.linalg.inv(K)
    except np.linalg.LinAlgError:
        print("Error: Intrinsic matrix is singular (non-invertible).")
        return None

    normalized_camera_coords = K_inv @ uv1
    camera_coords = depth * normalized_camera_coords
    world_coords = R @ camera_coords + t

    return world_coords.flatten()  # Returns a 1D array

if __name__ == "__main__":
    K = np.array([[627.794983, 0, 360.174988],
                  [0, 626.838013, 231.660996],
                  [0, 0, 1]])

    R = np.eye(3)
    t = np.array([0, 0, 0]).reshape((3, 1))

    u, v = 450, 180
    depth = -0.09

    world_coords = image_to_world(u, v, depth, K, R, t)

    if world_coords is not None:
        print("World Coordinates:", world_coords) # Only prints world coordinates
        # Now world_coords can be used for other purposes
        x_w, y_w, z_w = world_coords # Unpacking the array for easier use
        print(f"x_w: {x_w}, y_w: {y_w}, z_w: {z_w}")
    else:
        print("World coordinate calculation failed.")