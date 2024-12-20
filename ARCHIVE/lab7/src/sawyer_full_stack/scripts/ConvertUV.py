import cv2
import numpy as np
import matplotlib.pyplot as plt
import tf2_ros
import rospy
import tf2_geometry_msgs
import geometry_msgs.msg

class CameraTransform:
    def __init__(self):
        self.K = np.array([
            [627.794983, 0.0, 360.174988],
            [0.0, 626.838013, 231.660996],
            [0.0, 0.0, 1.0]
        ])
        self.D = np.array([-0.438799, 0.257299, 0.001038, 0.000384, -0.105028])
        self.object_z = -0.1
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        rospy.sleep(1.0)

    def pixel_to_base(self, u, v):
        transform = self.tf_buffer.lookup_transform(
            'base', 'right_hand_camera', rospy.Time(0), rospy.Duration(1.0))

        camera_position = np.array([transform.transform.translation.x,
                                     transform.transform.translation.y, transform.transform.translation.z])

        point = np.array([[[u, v]]], dtype=np.float32)
        undistorted = cv2.undistortPoints(point, self.K, self.D, P=self.K)
        u_undist, v_undist = undistorted[0][0]

        fx, fy = self.K[0, 0], self.K[1, 1]
        cx, cy = self.K[0, 2], self.K[1, 2]

        x_norm = (u_undist - cx) / fx
        y_norm = (v_undist - cy) / fy
        scale = camera_position[2] - self.object_z

        x_cam, y_cam, z_cam = x_norm * scale, y_norm * scale, scale
        point_cam_ps = geometry_msgs.msg.PointStamped()
        point_cam_ps.header.stamp = rospy.Time.now()
        point_cam_ps.header.frame_id = 'right_hand_camera'
        point_cam_ps.point.x, point_cam_ps.point.y, point_cam_ps.point.z = x_cam, y_cam, z_cam

        point_base = self.tf_buffer.transform(point_cam_ps, 'base', rospy.Duration(1.0))
        return (point_base.point.x, point_base.point.y, point_base.point.z)

def main():
    transformer = CameraTransform()
    u, v = 450, 180
    base_coords = transformer.pixel_to_base(u, v)
    print(f"Image coordinates (u,v): ({u}, {v})")
    print(f"Base coordinates (x,y,z): {base_coords}")

if __name__ == "__main__":
    main()
