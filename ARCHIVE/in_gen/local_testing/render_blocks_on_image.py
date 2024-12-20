import json
import random
import matplotlib.pyplot as plt
import numpy as np
import cv2


# Use the CameraTransform class from the provided camera code
class CameraTransform:
    def __init__(self):
            self.K = np.array([
                [627.794983, 0.0, 360.174988],
                [0.0, 626.838013, 231.660996],
                [0.0, 0.0, 1.0]
            ])
            self.D = np.array([-0.438799, 0.257299, 0.001038, 0.000384, -0.105028])
            self.camera_position = np.array([0.681, 0.121, 0.426])
            self.object_z = -0.09
            self.R = np.eye(3)  # Identity rotation matrix assuming no tilt

    def world_pt_to_pixel(point, K):
        translation_w_to_s = np.array([150, 0, 240])  
        
        R_w_to_s = np.array([[0, -1, 0],  
                            [0, 0, 1],  
                            [1, 0, 0]])  
        
        # Apply translation  and rotation to convert the world point to the sensor frame
        point_s = point - translation_w_to_s
        point_camera = np.dot(R_w_to_s, point_s)

        #P_homogeneous transforms into a 3x1 array which is the homogenoeius repreentation 3D to 2D
        p_homogeneous = np.dot(K,point_camera)

        #Pull out the last eleemnt in this 3x1 matric which is lambda, p_homg = [u,v,lambda]^T
        lambda_ = p_homogeneous[2]  

        #Simple checking
        is_lambda_negative = (lambda_ <= 0) 
        if not is_lambda_negative:
            #Take the U,V matrx, multiply by 1/lambda to get 2D coordinates from homogeneous
            pixel = p_homogeneous[:2] / lambda_
            pixel = pixel.astype(int)  
        else:
            pixel = np.zeros(2, dtype=int) 
        
        return (pixel, is_lambda_negative)

    def pixel_to_world(self, u, v):
        point = np.array([[[u, v]]], dtype=np.float32)
        undistorted = cv2.undistortPoints(point, self.K, self.D, P=self.K)
        u_undist = undistorted[0][0][0]
        v_undist = undistorted[0][0][1]

        fx = self.K[0, 0]
        fy = self.K[1, 1]
        cx = self.K[0, 2]
        cy = self.K[1, 2]

        x_norm = (u_undist - cx) / fx
        y_norm = -(v_undist - cy) / fy

        scale = (self.object_z - self.camera_position[2]) / -1.0

        x_cam = x_norm * scale
        y_cam = y_norm * scale
        z_cam = -scale

        point_cam = np.array([x_cam, y_cam, z_cam])
        point_world = np.dot(self.R, point_cam) + self.camera_position

        return tuple(point_world)


def render_blocks_on_image(json_data, image_path, transformer):
    env_image = cv2.imread(image_path)
    if env_image is None:
        print("Error: Could not load image.")
        return

    env_image_rgb = cv2.cvtColor(env_image, cv2.COLOR_BGR2RGB)
    image_height, image_width, _ = env_image.shape

    plt.figure(figsize=(8, 8))
    plt.imshow(env_image_rgb)

    for block in json_data['blocks']:
        position = block['position']
        color = block['color']
        placement_order = block['placement_order']

        # Project world coordinates to image using the transformer
        img_coords, is_lambda_negative = world_to_image_coordinates(position, transformer)
        if is_lambda_negative:
            continue  # Skip invalid points

        # Ensure points are within image bounds
        u, v = img_coords
        if not (0 <= u < image_width and 0 <= v < image_height):
            continue

        plt.scatter(u, v, label=f"Block {placement_order}", color=color, s=100)
        plt.text(u, v, f"{placement_order}", color='black', fontsize=8, ha='center', va='center')

    plt.axis('off')
    plt.title("Block Placement in the Environment")
    plt.show()



def world_to_image_coordinates(position, transformer):
    """
    Converts world coordinates to image coordinates using CameraTransform.
    """
    X, Y, Z = position

    # Convert world to pixel using the provided CameraTransform
    try:
        img_coords = transformer.pixel_to_world(X, Y)
        u, v, _ = img_coords

        # Check if the depth Z is behind the camera
        if Z <= transformer.object_z:
            return (u, v), True

        return (u, v), False
    except Exception as e:
        print(f"Error projecting point {position}: {e}")
        return (0, 0), True


def world_pt_f_to_pixel(point, f, img_height, img_width):
    """
        Transform a point in world coordinates to the corresponding pixel on the image plane.
        Inputs:
            point: a Numpy array of shape (3,) containing the [x, y, z] coordinates of the point in the world frame
            f: a scalar float for the focal length of the camera
            img_height: height of the image in pixels
            img_width: width of the image in pixels

        Outputs:
            pixel: a Numpy array of shape (2,) containing the [x, y] integer coordinates of the pixel in the image where the point should be drawn
            is_lambda_negative: a Boolean that should be true if the homogeneous pixel coordinates have a negative scaling factor lambda, and false otherwise.
                (This is used to mask out points that lie behind the image plane and therefore shouldn't be plotted)
    """
    c_x = img_width / 2  # Principal point x (center of image)
    c_y = img_height / 2  # Principal point y (center of image)
    
    # Intrinsic matrix for square pixels, no skew distortion
    K = np.array([
        [f, 0, c_x],  # f for both x and y directions, no skew
        [0, f, c_y],  # Principal point at the center
        [0, 0, 1]
    ])
    return world_pt_to_pixel(point, K)
# Example JSON Data
json_data = {
  "blocks": [
    { "block_id": 1, "position": [60.0, 60.0, 0.0], "orientation": [0, 0, 0], "color": "gray", "placement_order": 1 },
    { "block_id": 2, "position": [-60.0, 60.0, 0.0], "orientation": [0, 0, 0], "color": "gray", "placement_order": 2 },
    { "block_id": 3, "position": [60.0, -60.0, 0.0], "orientation": [0, 0, 0], "color": "gray", "placement_order": 3 },
    { "block_id": 4, "position": [-60.0, -60.0, 0.0], "orientation": [0, 0, 0], "color": "gray", "placement_order": 4 },
    { "block_id": 5, "position": [0.0, 60.0, 0.0], "orientation": [0, 0, 0], "color": "gray", "placement_order": 5 },
    { "block_id": 6, "position": [0.0, -60.0, 0.0], "orientation": [0, 0, 0], "color": "gray", "placement_order": 6 },
    { "block_id": 7, "position": [60.0, 0.0, 0.0], "orientation": [0, 0, 0], "color": "gray", "placement_order": 7 },
    { "block_id": 8, "position": [-60.0, 0.0, 0.0], "orientation": [0, 0, 0], "color": "gray", "placement_order": 8 },
    { "block_id": 9, "position": [0.0, 0.0, 0.0], "orientation": [0, 0, 0], "color": "gray", "placement_order": 9 },
    { "block_id": 10, "position": [30.0, 30.0, 40.0], "orientation": [0, 0, 0], "color": "yellow", "placement_order": 10 },
    { "block_id": 11, "position": [-30.0, 30.0, 40.0], "orientation": [0, 0, 0], "color": "yellow", "placement_order": 11 },
    { "block_id": 12, "position": [30.0, -30.0, 40.0], "orientation": [0, 0, 0], "color": "yellow", "placement_order": 12 },
    { "block_id": 13, "position": [-30.0, -30.0, 40.0], "orientation": [0, 0, 0], "color": "yellow", "placement_order": 13 },
    { "block_id": 14, "position": [0.0, 30.0, 40.0], "orientation": [0, 0, 0], "color": "yellow", "placement_order": 14 },
    { "block_id": 15, "position": [0.0, -30.0, 40.0], "orientation": [0, 0, 0], "color": "yellow", "placement_order": 15 },
    { "block_id": 16, "position": [30.0, 0.0, 40.0], "orientation": [0, 0, 0], "color": "yellow", "placement_order": 16 },
    { "block_id": 17, "position": [-30.0, 0.0, 40.0], "orientation": [0, 0, 0], "color": "yellow", "placement_order": 17 },
    { "block_id": 18, "position": [15.0, 15.0, 80.0], "orientation": [0, 0, 0], "color": "gray", "placement_order": 18 },
    { "block_id": 19, "position": [-15.0, 15.0, 80.0], "orientation": [0, 0, 0], "color": "gray", "placement_order": 19 },
    { "block_id": 20, "position": [15.0, -15.0, 80.0], "orientation": [0, 0, 0], "color": "gray", "placement_order": 20 },
    { "block_id": 21, "position": [-15.0, -15.0, 80.0], "orientation": [0, 0, 0], "color": "gray", "placement_order": 21 },
    { "block_id": 22, "position": [0.0, 0.0, 120.0], "orientation": [0, 0, 0], "color": "yellow", "placement_order": 22 }
  ]
}

# Example usage
image_path = "/Users/rohilkhare/106afinalproject/in_gen/local_testing/1733893443.944508.png"
transformer = CameraTransform()
render_blocks_on_image(json_data, image_path, transformer)