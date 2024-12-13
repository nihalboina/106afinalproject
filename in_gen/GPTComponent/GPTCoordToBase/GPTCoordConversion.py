import json
import numpy as np
import cv2
import tf2_ros
import rospy
import tf2_geometry_msgs
import geometry_msgs.msg


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
        self.object_z = -0.1

        # Calculate the transformation from camera to base
        # Since we're given position but not orientation, assuming camera looks straight down
        # You may need to adjust this based on actual camera orientation

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

        u_undist = u
        v_undist = v

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
        y_norm = (v_undist - cy) / fy  # Up is positive in camera frame

        # Step 3: Calculate the scaling factor
        # Since we know the Z coordinate of the object plane and camera position
        # we can calculate the scaling factor
        # Assuming camera looks down
        scale = camera_position[2] - self.object_z

        # Step 4: Calculate 3D point in camera frame
        x_cam = x_norm * scale
        y_cam = y_norm * scale
        z_cam = scale  # Positive because point is in front of camera

        # Step 5: Transform to base coordinates
        point_cam = np.array([x_cam, y_cam, z_cam])
        print(f"camera point: {point_cam}")

        point_cam_ps = geometry_msgs.msg.PointStamped()
        point_cam_ps.header.stamp = rospy.Time.now()
        # Ensure this matches your TF frame
        point_cam_ps.header.frame_id = 'right_hand_camera'
        point_cam_ps.point.x = x_cam
        point_cam_ps.point.y = y_cam
        point_cam_ps.point.z = z_cam

        # Step 5: Transform to base coordinates
        point_base = self.tf_buffer.transform(
            point_cam_ps, 'base', rospy.Duration(1.0))

        # Return the transformed point as a tuple
        return (point_base.point.x, point_base.point.y, point_base.point.z)

    def pixel_to_base_batch(self, pixels):
        """
        Convert a batch of pixel coordinates to base coordinates.

        Args:
            pixels (list of tuples): List of (u, v) pixel coordinates.

        Returns:
            list of tuples: List of (x, y, z) base coordinates.
        """
        try:
            transform = self.tf_buffer.lookup_transform(
                'base', 'right_hand_camera', rospy.Time(0), rospy.Duration(1.0))
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
            rospy.logerr(f"TF lookup failed: {e}")
            return [(None, None, None) for _ in pixels]

        camera_position = np.array([
            transform.transform.translation.x,
            transform.transform.translation.y,
            transform.transform.translation.z
        ])

        # Convert to NumPy array for vectorized operations
        pixels_np = np.array(pixels, dtype=np.float32).reshape(-1, 1, 2)

        # Undistort points
        undistorted = cv2.undistortPoints(pixels_np, self.K, self.D, P=self.K)
        undistorted = undistorted.reshape(-1, 2)

        # Camera intrinsics
        fx = self.K[0, 0]
        fy = self.K[1, 1]
        cx = self.K[0, 2]
        cy = self.K[1, 2]

        # Normalized image coordinates
        x_norm = (undistorted[:, 0] - cx) / fx
        # Negative because image y-axis is down
        y_norm = -(undistorted[:, 1] - cy) / fy

        # Scaling factor based on known Z
        # Assuming camera looks straight down
        scale = (camera_position[2] - self.object_z)

        # 3D points in camera frame
        x_cam = x_norm * scale
        y_cam = y_norm * scale
        z_cam = np.full_like(x_cam, scale)

        # Initialize list for transformed points
        points_base = []

        for xc, yc, zc in zip(x_cam, y_cam, z_cam):
            point_cam_ps = geometry_msgs.msg.PointStamped()
            point_cam_ps.header.stamp = rospy.Time.now()
            point_cam_ps.header.frame_id = 'right_hand_camera'
            point_cam_ps.point.x = xc
            point_cam_ps.point.y = yc
            point_cam_ps.point.z = zc

            # Transform to base frame
            try:
                point_base = self.tf_buffer.transform(
                    point_cam_ps, 'base', rospy.Duration(1.0))
                points_base.append(
                    (point_base.point.x, point_base.point.y, point_base.point.z))
            except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
                rospy.logerr("Transform failed for a point")
                points_base.append((None, None, None))

        return points_base

input_file_path = "/Users/rohilkhare/106afinalproject/in_gen/GPTComponent/GPTCoordToBase/Generated_UV_Points.json"
output_file_path = "/Users/rohilkhare/106afinalproject/in_gen/GPTComponent/GPTCoordToBase/ConvertedGPTWorldPts.json"


def main():
    rospy.init_node('convert_uv_to_world', anonymous=True)

    # Create an instance of the transformer
    transformer = CameraTransform()

    # Load the input JSON file
    try:
        with open(input_file_path, 'r') as f:
            uv_data = json.load(f)
    except FileNotFoundError:
        rospy.logerr("Input file not found.")
        return

    # Prepare the output data structure
    output_data = {"world_points": []}

    # Iterate over the UV points and convert each to world coordinates
    for point in uv_data["u_v_points"]:
        block_id = point["block_id"]
        u = point["u"]
        v = point["v"]

        # Transform the point using CameraTransform
        x, y, z = transformer.pixel_to_base(u, v)
        
        # Skip if the transformation failed
        if x is None or y is None or z is None:
            rospy.logwarn(f"Failed to transform block ID {block_id} (u, v)=({u}, {v})")
            continue

        # Add the transformed point to the output data
        output_data["world_points"].append({
            "block_id": block_id,
            "x": round(x, 4),
            "y": round(y, 4),
            "z": round(z, 4)
        })
        
        rospy.loginfo(f"Block {block_id}: (u, v)=({u}, {v}) -> (x, y, z)=({x:.4f}, {y:.4f}, {z:.4f})")

    # Save the transformed points to the output JSON file
    with open(output_file_path, 'w') as f:
        json.dump(output_data, f, indent=4)

    rospy.loginfo(f"Converted points saved to {output_file_path}")


if __name__ == "__main__":
    try:
        main()
    except rospy.ROSInterruptException:
        pass