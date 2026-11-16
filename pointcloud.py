
import cv2
import numpy as np
import open3d as o3d
from PIL import Image
from transformers import pipeline

# Step 1: Load the original image
image = cv2.imread("multi_objects.jpg")

if image is None:
    raise FileNotFoundError("multi_objects.jpg not found!")

height, width = image.shape[:2]

# Step 2: Load the depth estimation model
print("Loading depth model...")

depth_model = pipeline(
    task="depth-estimation",
    model="depth-anything/Depth-Anything-V2-Small-hf",
    device="cpu"
)

# Step 3: Generate relative depth predictions
print("Estimating depth...")

rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
pil_image = Image.fromarray(rgb_image)

result = depth_model(pil_image)

raw_depth = result["predicted_depth"]
raw_depth = raw_depth.squeeze().cpu().numpy()

# Step 4: Resize depth map if necessary
depth = cv2.resize(raw_depth, (width, height))

# Step 5: Convert relative depth to pseudo-distance
# Depth Anything V2 relative depth: larger usually means nearer
depth = np.maximum(depth, 0.001)
z = 1.0 / depth

# Step 6: Approximate camera parameters
fx = width * 0.9
fy = width * 0.9

cx = width / 2
cy = height / 2

# Step 7: Generate pixel coordinates
u, v = np.meshgrid(
    np.arange(width),
    np.arange(height)
)

# Step 8: Project pixels into 3D
x = (u - cx) * z / fx
y = (v - cy) * z / fy

# Step 9: Combine X, Y, Z into 3D points
points = np.stack(
    (x, -y, -z),
    axis=-1
).reshape(-1, 3)

# Step 10: Get the original pixel colours
colors = rgb_image.reshape(-1, 3) / 255.0

# Step 11: Create Open3D point cloud
point_cloud = o3d.geometry.PointCloud()

point_cloud.points = o3d.utility.Vector3dVector(points)
point_cloud.colors = o3d.utility.Vector3dVector(colors)


# Step 12: Reduce the number of points using a voxel grid
print("Original points:", len(point_cloud.points))

voxel_size = 0.01

downsampled_cloud = point_cloud.voxel_down_sample(
    voxel_size=voxel_size
)

print("Downsampled points:", len(downsampled_cloud.points))

# Save the optimised point cloud
o3d.io.write_point_cloud(
    "scene_downsampled.ply",
    downsampled_cloud
)

print("Optimised point cloud saved!")

# Step 12: Display point cloud

# Save the generated 3D point cloud
o3d.io.write_point_cloud(
    "scene.ply",
    point_cloud
)

print("3D point cloud saved to scene.ply!")

print("Displaying 3D point cloud...")


# Create an interactive visualiser
vis = o3d.visualization.Visualizer()

vis.create_window(
    window_name="3D Vision - Optimised Point Cloud",
    width=1000,
    height=700
)

# Add the optimised point cloud
vis.add_geometry(downsampled_cloud)


# Create a 3D coordinate system
coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
    size=0.2,
    origin=[0, 0, 0]
)

# Add coordinate axes to the visualiser
vis.add_geometry(coordinate_frame)

# Set background colour
render_option = vis.get_render_option()
render_option.background_color = np.array([0.05, 0.05, 0.08])
render_option.point_size = 2.0

# Display the scene
vis.run()

# Close the visualiser
vis.destroy_window()

