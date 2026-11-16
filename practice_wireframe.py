
import open3d as o3d

# Create a solid box
box = o3d.geometry.TriangleMesh.create_box(
    width=1.0,
    height=1.5,
    depth=0.5
)

# Convert the box edges into a wireframe
wireframe = o3d.geometry.LineSet.create_from_triangle_mesh(box)

# TODO 1: Make the wireframe green
wireframe.paint_uniform_color([0, 1, 0])  # RGB for green

# TODO 2: Move the wireframe by (2, 0, 0)
wireframe.translate([2, 0, 0])

# Display only the wireframe
o3d.visualization.draw_geometries([wireframe])
