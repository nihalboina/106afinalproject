import numpy as np
import cv2
import tf2_ros
import rospy
import tf2_geometry_msgs


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

        # Calculate the transformation from camera to base
        # Since we're given position but not orientation, assuming camera looks straight down
        # You may need to adjust this based on actual camera orientation
        self.R = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0]
        ])

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        # You might need to add a short sleep here to allow the listener to fill the buffer
        rospy.sleep(1.0)

    def pixel_to_base(self, u, v):
        """
        Convert pixel coordinates (u,v) to base coordinates (x,y,z)

        Args:
            u (float): x-coordinate in image
            v (float): y-coordinate in image

        Returns:
            tuple: (x, y, z) coordinates in base frame
        """
        transform = self.tf_buffer.lookup_transform(
            'base', 'right_hand_camera', rospy.Time(0), rospy.Duration(1.0))

        camera_position = np.array([transform.transform.translation.x,
                                   transform.transform.translation.y, transform.transform.translation.z])

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
        z_cam = scale  # Positive because point is in front of camera

        # Step 5: Transform to base coordinates
        point_cam = np.array([x_cam, y_cam, z_cam])
        print(f"camera point: {point_cam}")
        # point_base = np.dot(self.R, point_cam) + camera_position

        point_in_base = tf2_geometry_msgs.do_transform_point(
            point_cam, transform)

        return tuple(point_in_base)


def main():
    # Create instance of transformer
    transformer = CameraTransform()

    # Example usage
    u, v = 450, 180  # Example pixel coordinates
    base_coords = transformer.pixel_to_base(u, v)
    print(f"Image coordinates (u,v): ({u}, {v})")
    print(f"base coordinates (x,y,z): {base_coords}")

    # For real-time use, you would do something like:
    """
    while True:
        # Get image from camera
        # Process image to get object pixel coordinates (u,v)
        # Convert to base coordinates
        base_coords = transformer.pixel_to_base(u, v)
        # Use base coordinates for robot control
    """


if __name__ == "__main__":
    main()
