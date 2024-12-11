import cv2
from stl import mesh
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import time
import os
import glob


import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from stl import mesh
import bpy
import math
import os
import sys
from mathutils import Vector, Euler
from PIL import Image


def plot_stl_views(stl_file, output_top_down, output_top_left, image_size=512):
    # Utility functions
    def clean_scene():
        # Remove all existing objects
        bpy.ops.object.select_all(action='SELECT')
        bpy.ops.object.delete()

    def import_stl(filepath):
        # Import STL mesh
        bpy.ops.import_mesh.stl(filepath=filepath)
        obj = bpy.context.selected_objects[0]
        return obj

    def setup_camera(elev_deg, azim_deg):
        # Create a camera if it doesn't exist
        if "Camera" not in bpy.data.objects:
            bpy.ops.object.camera_add(location=(0, 0, 0))
        camera = bpy.data.objects["Camera"]

        # Set rotation
        # We start with camera looking down -Z axis:
        # Then apply azimuth rotation around Z and elevation around X
        eul = Euler((0, 0, 0), 'XYZ')
        eul.rotate_axis('Z', math.radians(azim_deg))
        eul.rotate_axis('X', math.radians(elev_deg))
        camera.rotation_euler = eul
        return camera

    def fit_camera_to_object(obj, camera, margin=1.2):
        # Compute bounding box
        bbox = [obj.matrix_world @ Vector(corner) for corner in obj.bound_box]
        min_coords = Vector((min([v.x for v in bbox]),
                             min([v.y for v in bbox]),
                             min([v.z for v in bbox])))
        max_coords = Vector((max([v.x for v in bbox]),
                             max([v.y for v in bbox]),
                             max([v.z for v in bbox])))

        center = (min_coords + max_coords) / 2.0
        diag = (max_coords - min_coords).length

        bpy.context.scene.camera = camera
        fov = camera.data.angle  # field of view in radians
        distance = (diag/2) / math.tan(fov/2) * margin

        # Move camera along its local -Z axis
        direction = camera.matrix_world.to_quaternion() @ Vector((0.0, 0.0, -1.0))
        camera.location = center - direction * distance

    def setup_light():
        # Create a Sun light if none exists
        bpy.ops.object.light_add(type='SUN', location=(0, 0, 10))
        light = bpy.context.active_object
        light.data.energy = 5.0

        # White world background
        bpy.context.scene.world.use_nodes = True
        bg = bpy.context.scene.world.node_tree.nodes["Background"]
        bg.inputs[0].default_value = (1, 1, 1, 1)
        bg.inputs[1].default_value = 1.0

    def setup_material(obj):
        # Create a simple gray material
        mat = bpy.data.materials.new(name="GrayMaterial")
        mat.diffuse_color = (0.5, 0.5, 0.5, 1.0)
        if len(obj.data.materials) == 0:
            obj.data.materials.append(mat)
        else:
            obj.data.materials[0] = mat

    def render_to_file(filepath):
        bpy.context.scene.render.filepath = filepath
        bpy.ops.render.render(write_still=True)
        # Convert to grayscale
        img = Image.open(filepath)
        gray = img.convert('L')
        gray.save(filepath)

    # Begin setup
    clean_scene()
    obj = import_stl(stl_file)
    setup_material(obj)
    setup_light()

    # Set render settings
    scene = bpy.context.scene
    scene.render.engine = 'CYCLES'  # or 'BLENDER_EEVEE'
    scene.cycles.samples = 64
    scene.render.resolution_x = image_size
    scene.render.resolution_y = image_size
    scene.render.resolution_percentage = 100

    # Render top-down view
    cam_top_down = setup_camera(elev_deg=90, azim_deg=0)
    fit_camera_to_object(obj, cam_top_down)
    render_to_file(output_top_down)

    # Render top-left view
    cam_top_left = setup_camera(elev_deg=45, azim_deg=-45)
    fit_camera_to_object(obj, cam_top_left)
    render_to_file(output_top_left)


def setup():
    # Generate views for all objects in the folder and place into the generated_views folder
    path_to_objects = "../sawyer/CV/src/object_stls"
    path_to_views = "../sawyer/CV/src/generated_views"
    files_and_objects = os.listdir(path_to_objects)

    for obj in files_and_objects:
        if not obj.lower().endswith('.stl'):
            continue
        wout_stl = obj.replace('.stl', '')
        full_obj_path = f"{path_to_objects}/{obj}"
        output_top_down = f"{path_to_views}/{wout_stl}_top_down_view.png"
        output_top_left = f"{path_to_views}/{wout_stl}_top_left_corner_view.png"
        if not (os.path.exists(output_top_down) and os.path.exists(output_top_left)):
            plot_stl_views(full_obj_path, output_top_down, output_top_left)


def run_cv(current_frame, background, number_objects=1):
    # If no background is available, just return the current frame
    if background is None:
        return current_frame, None

    # Convert to grayscale
    gray_current = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
    gray_background = cv2.cvtColor(background, cv2.COLOR_BGR2GRAY)

    # Compute absolute difference between current frame and background
    diff = cv2.absdiff(gray_background, gray_current)

    # Threshold the difference image
    _, thresh = cv2.threshold(diff, 50, 255, cv2.THRESH_BINARY)

    # Morphological operations to clean up noise
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

    # Find contours
    contours, _ = cv2.findContours(
        thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    rectangles = []
    # Basic criterion: bounding boxes that are not too small
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w > 20 and h > 20:  # arbitrary minimal size filter
            rectangles.append((x, y, w, h))

    # Sort rectangles by area in descending order
    rectangles = sorted(rectangles, key=lambda r: -(r[2]*r[3]))

    matched_object = None
    if len(rectangles) > 0:
        # Attempt object recognition using template matching
        path_to_views = "../sawyer/CV/src/generated_views"
        generated_views = glob.glob(os.path.join(path_to_views, "*_view.png"))

        for rect_ind in range(min(len(rectangles), number_objects)):
            x, y, w, h = rectangles[rect_ind]
            roi = current_frame[y:y+h, x:x+w]
            roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

            best_match_score = 0.0
            best_view = None
            scores = []  # Store (view_file, score) for debugging

            for view_file in generated_views:
                template = cv2.imread(view_file, cv2.IMREAD_GRAYSCALE)
                if template is None:
                    continue

                # Resize template to match ROI size
                template_resized = cv2.resize(template, (w, h))

                res = cv2.matchTemplate(
                    roi_gray, template_resized, cv2.TM_CCOEFF_NORMED)
                _, max_val, _, _ = cv2.minMaxLoc(res)

                scores.append((view_file, max_val))

                if max_val > best_match_score:
                    best_match_score = max_val
                    best_view = view_file

            # After checking all views
            if best_match_score >= 0.9:
                matched_object = best_view
                cv2.rectangle(current_frame, (x, y),
                              (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(current_frame, f"Matched: {os.path.basename(best_view)} ({best_match_score:.2f})",
                            (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            else:
                # Print no match and the best score, plus the scores for each template
                cv2.rectangle(current_frame, (x, y),
                              (x+w, y+h), (0, 0, 255), 2)
                no_match_text = f"No match ({best_match_score:.2f})"
                cv2.putText(current_frame, no_match_text,
                            (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

                # Print scores for each template tried (for debugging)
                # This will print in the console
                print("No match for this object. Scores were:")
                for vf, sc in scores:
                    print(
                        f"  Template: {os.path.basename(vf)}, Score: {sc:.2f}")

    return current_frame, matched_object


if __name__ == "__main__":
    # Setup
    setup()

    # Open the default camera
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Cannot access the camera")
        exit(1)

    background = None
    print("Press 'b' to set the background. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to read from camera. Exiting...")
            break

        processed_frame, matched_object = run_cv(frame, background)

        # Display the processed frame
        cv2.imshow("Camera", processed_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('b'):
            # Capture the current frame as background
            background = frame.copy()
            print("Background set.")

    cap.release()
    cv2.destroyAllWindows()
