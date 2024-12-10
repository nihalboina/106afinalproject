import json
import random
import matplotlib.pyplot as plt
import numpy as np
import cv2

def render_blocks_on_image(json_data, image_path, origin_point=None):
    env_image = cv2.imread(image_path)
    if env_image is None:
        print("Error: Could not load image.")
        return

    env_image_rgb = cv2.cvtColor(env_image, cv2.COLOR_BGR2RGB)
    image_height, image_width, _ = env_image.shape

    json_x_min = min([block['position'][0] for block in json_data['blocks']]) - 3
    json_x_max = max([block['position'][0] for block in json_data['blocks']]) + 3
    json_y_min = min([block['position'][1] for block in json_data['blocks']]) - 3
    json_y_max = max([block['position'][1] for block in json_data['blocks']]) + 3

    json_width = json_x_max - json_x_min
    json_height = json_y_max - json_y_min
    scaling_factor_x = image_width / json_width
    scaling_factor_y = image_height / json_height
    scaling_factor = min(scaling_factor_x, scaling_factor_y)  # Uniform scaling

    if origin_point is None:
        origin_point = [json_x_min, json_y_min]

    plt.figure(figsize=(8, 8))
    plt.imshow(env_image_rgb)

    for block in json_data['blocks']:
        position = block['position']
        color = block['color']
        placement_order = block['placement_order']

        scaled_position = [
            (position[0] - origin_point[0]) * scaling_factor,
            (position[1] - origin_point[1]) * scaling_factor,
        ]

        scaled_position[1] = image_height - scaled_position[1]

        plt.scatter(scaled_position[0], scaled_position[1], label=f"Block {placement_order}", color=color, s=100)
        plt.text(scaled_position[0], scaled_position[1], f"{placement_order}", color='black', fontsize=8, ha='center', va='center')

    plt.axis('off')
    plt.title("Block Placement in the Environment")
    plt.show()


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
image_path = "/Users/rohilkhare/106afinalproject/in_gen/local_testing/image.jpg"  # Path to your environment image
render_blocks_on_image(json_data, image_path)