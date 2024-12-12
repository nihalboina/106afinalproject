import numpy as np
import cv2


class CameraTransform:
    def __init__(self):
        # Camera intrinsics
        self.K = np.array([
            [627.794983, 0.0, 360.174988],
            [0.0, 626.838013, 231.660996],
            [0.0, 0.0, 1.0]
        ])

        # Distortion coefficients
        self.D = np.array([-0.438799, 0.257299, 0.001038, 0.000384, -0.105028])

        # Camera position in robot frame
        # self.camera_position = np.array([0.681, 0.121, 0.426])

        # Known Z coordinate of the object plane
        self.object_z = -0.09

        # Calculate the transformation from camera to world
        # Since we're given position but not orientation, assuming camera looks straight down
        # You may need to adjust this based on actual camera orientation
        self.R = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0]
        ])

    def pixel_to_world(self, u, v, camera_position=np.array([0.681, 0.121, 0.426])):
        """
        Convert pixel coordinates (u,v) to world coordinates (x,y,z)

        Args:
            u (float): x-coordinate in image
            v (float): y-coordinate in image

        Returns:
            tuple: (x, y, z) coordinates in world frame
        """
        # Step 1: Undistort the point
        point = np.array([[[u, v]]], dtype=np.float32)
        undistorted = cv2.undistortPoints(point, self.K, self.D, P=self.K)
        u_undist = undistorted[0][0][0]
        v_undist = undistorted[0][0][1]

        # Step 2: Convert to normalized image coordinates
        # Note:
        # - Image coordinates (u,v) have origin at top-left
        # - Camera coordinates have origin at center with:
        #   - X axis pointing right
        #   - Y axis pointing down
        #   - Z axis pointing forward
        fx = self.K[0, 0]
        fy = self.K[1, 1]
        cx = self.K[0, 2]
        cy = self.K[1, 2]

        # Calculate ray from camera, accounting for image coordinate system
        x_norm = (u_undist - cx) / fx  # Right is positive
        y_norm = -(v_undist - cy) / fy  # Up is positive in camera frame

        # Step 3: Calculate the scaling factor
        # Since we know the Z coordinate of the object plane and camera position
        # we can calculate the scaling factor
        # Assuming camera looks down
        scale = (self.object_z - camera_position[2]) / (-1.0)

        # Step 4: Calculate 3D point in camera frame
        x_cam = x_norm * scale
        y_cam = y_norm * scale
        z_cam = -scale  # Negative because point is in front of camera

        # Step 5: Transform to world coordinates
        point_cam = np.array([x_cam, y_cam, z_cam])
        point_world = np.dot(self.R, point_cam) + camera_position

        return tuple(point_world)


def main():
    # Create instance of transformer
    transformer = CameraTransform()

    # Example usage
    u, v = 450, 180  # Example pixel coordinates
    world_coords = transformer.pixel_to_world(u, v)
    print(f"Image coordinates (u,v): ({u}, {v})")
    print(f"World coordinates (x,y,z): {world_coords}")

    # For real-time use, you would do something like:
    """
    while True:
        # Get image from camera
        # Process image to get object pixel coordinates (u,v)
        # Convert to world coordinates
        world_coords = transformer.pixel_to_world(u, v)
        # Use world coordinates for robot control
    """


if __name__ == "__main__":
    main()
