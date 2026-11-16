import open3d as o3d
from transformers import pipeline
from PIL import Image
import numpy as np
from ultralytics import YOLO

# Step 1: Load the GPU-accelerated YOLO model
model = YOLO("yolo11n_openvino_model/")

# Step 2: Detect objects in our saved image
results = model.predict(
    source="capture.jpg",
    device="intel:gpu",
    verbose=False
)

detected_objects = []

# Step 3: Extract the bounding boxes
for box in results[0].boxes:

    # Get the object's name
    class_id = int(box.cls[0])
    name = model.names[class_id]

    # Get bounding box coordinates
    x1, y1, x2, y2 = box.xyxy[0].tolist()

    # Calculate centre of bounding box
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2

    print(f"\nDetected: {name}")
    print(f"Bounding box: {x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f}")
    print(f"Centre: ({center_x:.1f}, {center_y:.1f})")

    # Store this object's detection information
    detected_object = {
        "name": name,
        "center_x": center_x,
        "center_y": center_y
    }

    # TODO: Add detected_object to detected_objects
    detected_objects.append(detected_object)


# Load the depth estimation model
print("\nLoading depth model...")

depth_model = pipeline(
    task="depth-estimation",
    model="depth-anything/Depth-Anything-V2-Small-hf",
    device="cpu"
)

# Generate depth predictions
image = Image.open("capture.jpg").convert("RGB")
result = depth_model(image)

depth = result["predicted_depth"]
depth = depth.squeeze().cpu().numpy()

# Store the 3D positions of all detected objects
objects_3d = []

# Camera parameters
width, height = image.size

fx = width * 0.9
fy = width * 0.9

cx = width / 2
cy = height / 2

# Process each detected object
for obj in detected_objects:

    # TODO 1: Retrieve the object's centre coordinates
    center_x = obj["center_x"]
    center_y = obj["center_y"]

    # Convert to integer pixel coordinates
    x = int(center_x)
    y = int(center_y)

    # Get a small region around the centre
    radius = 10

    h, w = depth.shape

    left = max(0, x - radius)
    right = min(w, x + radius + 1)

    top = max(0, y - radius)
    bottom = min(h, y + radius + 1)

    depth_region = depth[top:bottom, left:right]

    valid_depth = depth_region[
        np.isfinite(depth_region) & (depth_region > 0)
    ]

    if valid_depth.size == 0:
        continue

    # Calculate median relative depth
    object_depth = float(np.median(valid_depth))

    # Convert to pseudo-distance
    z = 1.0 / max(object_depth, 0.001)

    # Calculate approximate 3D position
    X = (center_x - cx) * z / fx
    Y = -(center_y - cy) * z / fy
    Z = -z

    # Store the results
    objects_3d.append({
        "name": obj["name"],
        "position": [X, Y, Z],
        "depth": object_depth
    })

    print(f"{obj['name']}: ({X:.4f}, {Y:.4f}, {Z:.4f})")

print("Total 3D objects:", len(objects_3d))


# Load our existing point cloud
point_cloud = o3d.io.read_point_cloud(
    "scene_downsampled.ply"
)

# Begin with the point cloud
geometries = [point_cloud]

# Process every 3D object
for obj in objects_3d:

    # TODO 1: Retrieve the object's 3D position
    position = obj["position"]

    # Create a red sphere
    marker = o3d.geometry.TriangleMesh.create_sphere(
        radius=0.015
    )

    marker.compute_vertex_normals()
    marker.paint_uniform_color([1, 0, 0])

    # TODO 2: Move marker to the object's position
    marker.translate(position)

    # Create an approximate bounding box
    center = np.array(position)

    half_size = np.array([
        0.075,
        0.125,
        0.05
    ])

    bbox = o3d.geometry.AxisAlignedBoundingBox(
        min_bound=center - half_size,
        max_bound=center + half_size
    )

    bbox.color = [0, 1, 0]

    # TODO 3: Add both geometries to the list
    geometries.append(marker)
    geometries.append(bbox)

# Display the complete scene
print("Number of geometries:", len(geometries))

o3d.visualization.draw_geometries(
    geometries,
    window_name="3D Vision - Multiple Objects",
    width=1000,
    height=700
)


# Load our existing point cloud
point_cloud = o3d.io.read_point_cloud(
    "scene_downsampled.ply"
)

# Begin with the point cloud
geometries = [point_cloud]

# Process every 3D object
for obj in objects_3d:

    # TODO 1: Retrieve the object's 3D position
    position = obj["position"]

    # Create a red sphere
    marker = o3d.geometry.TriangleMesh.create_sphere(
        radius=0.015
    )

    marker.compute_vertex_normals()
    marker.paint_uniform_color([1, 0, 0])

    # TODO 2: Move marker to the object's position
    marker.translate(position)

    # Create an approximate bounding box
    center = np.array(position)

    half_size = np.array([
        0.075,
        0.125,
        0.05
    ])

    bbox = o3d.geometry.AxisAlignedBoundingBox(
        min_bound=center - half_size,
        max_bound=center + half_size
    )

    bbox.color = [0, 1, 0]

    # TODO 3: Add both geometries to the list
    geometries.append(marker)
    geometries.append(bbox)

# Display the complete scene
print("Number of geometries:", len(geometries))

o3d.visualization.draw_geometries(
    geometries,
    window_name="3D Vision - Multiple Objects",
    width=1000,
    height=700
)

# Get depth at the detected person's centre

# Sample a small region around the bounding box centre
x = int(center_x)
y = int(center_y)

radius = 10

# Clamp coordinates to the depth map boundaries
h, w = depth.shape

left = max(0, x - radius)
right = min(w, x + radius + 1)

top = max(0, y - radius)
bottom = min(h, y + radius + 1)

# Extract depth values from the region
depth_region = depth[top:bottom, left:right]

# Ignore invalid values
valid_depth = depth_region[
    np.isfinite(depth_region) & (depth_region > 0)
]

if valid_depth.size == 0:
    raise ValueError("No valid depth values found")

# Use median instead of a single pixel
person_depth = float(np.median(valid_depth))

print(f"Median relative depth: {person_depth:.4f}")


print(f"Person relative depth: {person_depth:.4f}")


# Step 1: Get image dimensions
width, height = image.size

# Step 2: Approximate camera parameters
fx = width * 0.9
fy = width * 0.9

cx = width / 2
cy = height / 2

# Step 3: Convert relative depth to pseudo-distance
z = 1.0 / max(float(person_depth), 0.001)

# Step 4: Convert 2D pixel coordinates into 3D
X = (center_x - cx) * z / fx
Y = -(center_y - cy) * z / fy
Z = -z

# Step 5: Display the result
print("\n--- 3D OBJECT POSITION ---")
print(f"Object: {name}")
print(f"X: {X:.4f}")
print(f"Y: {Y:.4f}")
print(f"Z: {Z:.4f}")


# Step 1: Load the saved point cloud
point_cloud = o3d.io.read_point_cloud(
    "scene_downsampled.ply"
)

# Step 2: Create a red marker
marker = o3d.geometry.TriangleMesh.create_sphere(
    radius=0.015
)

marker.compute_vertex_normals()
marker.paint_uniform_color([1, 0, 0])
marker.translate([X, Y, Z])

# Step 3: Calculate bounding box dimensions
center = np.array([X, Y, Z])

box_width = 0.15
box_height = 0.25
box_depth = 0.10

half_size = np.array([
    box_width / 2,
    box_height / 2,
    box_depth / 2
])

min_point = center - half_size
max_point = center + half_size

# Step 4: Create the bounding box
bbox = o3d.geometry.AxisAlignedBoundingBox(
    min_bound=min_point,
    max_bound=max_point
)

bbox.color = [0, 1, 0]

# Step 5: Store geometries
geometries = []

# TODO 1: Add the point cloud to geometries
geometries.append(point_cloud)

# TODO 2: Add the red marker to geometries
geometries.append(marker)

# TODO 3: Add the green bounding box to geometries
geometries.append(bbox)

# Step 6: Verify our geometry list
print("Number of geometries:", len(geometries))

# Step 7: Display everything
o3d.visualization.draw_geometries(
    geometries,
    window_name="3D Object Recognition",
    width=1000,
    height=700
)
