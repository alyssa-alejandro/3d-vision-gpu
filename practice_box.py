
import open3d as o3d

# Create a 3D box
box = o3d.geometry.TriangleMesh.create_box(
    width=1.0,
    height=1.5,
    depth=0.5
)

# TODO 1: Make the box green
box.paint_uniform_color([0, 1, 0])  # RGB for green

# TODO 2: Move the box to (2, 0, 0)
box.translate([2, 0, 0])

# Display the box
o3d.visualization.draw_geometries([box])
