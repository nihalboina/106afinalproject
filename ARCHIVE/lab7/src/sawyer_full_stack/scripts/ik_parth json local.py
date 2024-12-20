#!/usr/bin/env python

import numpy as np
from numpy import linalg
import sys
import json
from time import sleep

def tuck():
    """
    Tuck the robot arm to the start position. Use with caution
    """
    print('Tucking the arm.')
    sleep(1)
    input("enter")

def pick_or_place():
    print("tucking")
    input("enter")

import json
import time

def get_block_coordinates(json_file, num, retry_interval=5):
    """
    Extracts x, y, z coordinates of item `num` from a JSON file.
    If the file doesn't contain items, waits and keeps trying.

    :param file_path: Path to the JSON file.
    :param num: Item number to extract coordinates for.
    :return: Tuple of (x, y, z) coordinates.
    """
    while True:
        try:
            # Load the JSON file
            with open(json_file, 'r') as file:
                data = json.load(file)

            # Check if the item `num` exists
            if str(num) in data:
                item = data[str(num)]
                position = item["pose"]["position"]
                x = position["x"]
                y = position["y"]
                z = position["z"]
                return x, y, z
            else:
                print(f"Item {num} not found. Retrying...")

        except json.JSONDecodeError:
            print("Invalid JSON format. Retrying...")
        except FileNotFoundError:
            print("File not found. Retrying...")
        except KeyError:
            print("Unexpected data structure. Retrying...")

        # Wait before retrying
        time.sleep(1)

# Example usage
# coordinates = extract_xyz_from_json('data.json', 3)
# print(coordinates)


def get_placement_coordinates(json_file, block_num):
    try:
        # Load the JSON file
        with open(json_file, 'r') as file:
            data = json.load(file)

        # Check if the data has a 'world_points' key and ensure it's a list
        if 'world_points' in data and isinstance(data['world_points'], list):
            # Find the block with the specified block_id
            for block in data['world_points']:
                if block.get('block_id') == block_num:
                    # Extract the x, y, z coordinates
                    x = block.get('x', None)
                    y = block.get('y', None)
                    z = block.get('z', None)

                    if x is not None and y is not None and z is not None:
                        return x, y, z
                    else:
                        raise ValueError(f"Block {block_num} does not contain complete x, y, z coordinates.")
            raise ValueError(f"Block with block_id {block_num} not found.")
        else:
            raise ValueError("The JSON data does not contain a valid 'world_points' list.")

    except json.JSONDecodeError:
        raise ValueError("Failed to decode JSON. Please check the file format.")
    except FileNotFoundError:
        raise ValueError(f"File {json_file} not found.")
    
def main():
    json_file_cv =  "lab7\src\sawyer_full_stack\scripts\detected_blocks.json"  # Replace with your JSON file name
    json_file_gpt = "lab7\src\sawyer_full_stack\scripts\ConvertedGPTWorldPts.json"  # Replace with your JSON file name

    tuck()

    # Calibrate the gripper (other commands won't work unless you do this first)


    while True:
        input('Press [ Enter ]: ')
        
        
        # Set the desired orientation for the end effector HERE
        ###group.set_position_target([0.5, 0.5, 0.0])
        
        z_offset = 0.15

        #get in cv coordinates block 0
        block_pick_num = 0
        try:
            x1, y1, z1 = get_block_coordinates(json_file_cv, block_pick_num)
            print(f"Coordinates of the first block: x={x1}, y={y1}, z={z1}")
        except ValueError as e:
            print(e)

        # pick 1
        pick_or_place()
        
        tuck()

        #get gpt coordinates
        block_num = 1
        try:
            x2, y2, z2 = get_placement_coordinates(json_file_gpt, block_num)
            print(f"Coordinates of place block {block_num}: x={x2}, y={y2}, z={z2}")
        except ValueError as e:
            print(e)

        # place 1
        pick_or_place()
        
        tuck()

        # pick 2
        block_pick_num += 1
        try:
            x3, y3, z3 = get_block_coordinates(json_file_cv, block_pick_num)
            print(f"Coordinates of the first block: x={x3}, y={y3}, z={z3}")
        except ValueError as e:
            print(e)
         
        pick_or_place()
        
        tuck()

        # place 2
        block_num += 1
        try:
            x4, y4, z4 = get_placement_coordinates(json_file_gpt, block_num)
            print(f"Coordinates of block {block_num}: x={x4}, y={y4}, z={z4}")
        except ValueError as e:
            print(e)
        pick_or_place()
        
        tuck()

        # pick 3
        block_pick_num += 1
        try:
            x5, y5, z5 = get_block_coordinates(json_file_cv, block_pick_num)
            print(f"Coordinates of the first block: x={x5}, y={y5}, z={z5}")
        except ValueError as e:
            print(e)
         
        pick_or_place()

        tuck()

        # place 3
        block_num += 1
        try:
            x6, y6, z6 = get_placement_coordinates(json_file_gpt, block_num)
            print(f"Coordinates of block {block_num}: x={x6}, y={y6}, z={z6}")
        except ValueError as e:
            print(e)

        pick_or_place()
        
        tuck()


# Python's syntax for a main() method
if __name__ == '__main__':
    main()
