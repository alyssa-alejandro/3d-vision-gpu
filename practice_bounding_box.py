
import open3d as o3d

# Define the minimum and maximum corners
min_point = [0, 0, 0]
max_point = [1, 2, 1]

# Create a 3D bounding box
bbox = o3d.geometry.AxisAlignedBoundingBox(
    min_bound=min_point,
    max_bound=max_point
)

# TODO 1: Set the bounding box colour to green
bbox.color = [0, 1, 0]  # RGB for green

# TODO 2: Move the bounding box by (2, 0, 0)
bbox.translate([2, 0, 0])

# Display the bounding box
o3d.visualization.draw_geometries([bbox])
