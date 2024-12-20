import cv2
import numpy as np
import open3d as o3d

# Function to render the STL model to an image and get its edges


def render_stl_to_edge_image(mesh, width=500, height=500):
    # Set up Open3D visualizer
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=width, height=height, visible=False)
    vis.add_geometry(mesh)
    ctr = vis.get_view_control()
    ctr.set_zoom(0.8)
    vis.poll_events()
    vis.update_renderer()

    # Capture image from the visualizer
    img = vis.capture_screen_float_buffer(False)
    vis.destroy_window()
    img = np.asarray(img)
    img = (img * 255).astype(np.uint8)

    # Convert the image from RGB to grayscale
    gray_model = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # Use Canny edge detection to get edges of the model
    edges_model = cv2.Canny(gray_model, 50, 150)

    return edges_model


# Load the STL model
mesh = o3d.io.read_triangle_mesh('block.stl')
if mesh.is_empty():
    print("Failed to load mesh. Please check the STL file.")
    exit()
mesh.compute_vertex_normals()

# Render the STL model to get its edge image
model_edges = render_stl_to_edge_image(mesh)

# Find contours in the model edge image
model_contours, _ = cv2.findContours(
    model_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
if len(model_contours) == 0:
    print("No contours found in model edge image.")
    exit()
model_cnt = max(model_contours, key=cv2.contourArea)

# Compute shape descriptors (Hu moments) of the model's contour
model_hu_moments = cv2.HuMoments(cv2.moments(model_cnt)).flatten()

# Initialize the camera
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open camera")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    # Convert the frame to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Use Canny edge detection on the grayscale frame
    edges_frame = cv2.Canny(gray_frame, 50, 150)

    # Find contours in the edge image
    contours, _ = cv2.findContours(
        edges_frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Create a copy of the frame to draw on
    output_frame = cv2.cvtColor(gray_frame, cv2.COLOR_GRAY2BGR)

    for cnt in contours:
        # Ignore small contours that may be noise
        if cv2.contourArea(cnt) < 500:
            continue

        # Compute shape descriptors (Hu moments)
        moments = cv2.moments(cnt)
        if moments['m00'] == 0:
            continue
        hu_moments = cv2.HuMoments(moments).flatten()

        # Compute similarity measure using cv2.matchShapes
        similarity = cv2.matchShapes(
            model_cnt, cnt, cv2.CONTOURS_MATCH_I1, 0.0)

        # Compute score as inverse of similarity measure
        # The lower the similarity value, the better the match
        # We invert and scale to get a percentage
        score = 1 / (1 + similarity)
        percentage = score * 100

        # Limit percentage to [0, 100]
        percentage = min(max(percentage, 0), 100)

        # Get bounding box around the contour
        x, y, w, h = cv2.boundingRect(cnt)

        # Draw bounding box and percentage chance
        cv2.rectangle(output_frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.putText(output_frame, f"{percentage:.2f}%", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    # Display the output frame
    cv2.imshow('Output', output_frame)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
