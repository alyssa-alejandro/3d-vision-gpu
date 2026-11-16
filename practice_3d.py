
import open3d as o3d

# Sphere A
sphere_a = o3d.geometry.TriangleMesh.create_sphere(
    radius=0.2
)

# TODO 1: Make sphere A red
sphere_a.paint_uniform_color([1, 0, 0])  # RGB for red

# TODO 2: Move sphere A to (1, 0, 0)
sphere_a.translate([1, 0, 0])

# Sphere B
sphere_b = o3d.geometry.TriangleMesh.create_sphere(
    radius=0.3
)

# TODO 3: Make sphere B blue
sphere_b.paint_uniform_color([0, 0, 1])  # RGB for blue

# TODO 4: Move sphere B to (0, 2, 0)
sphere_b.translate([0, 2, 0])

# Display both spheres
o3d.visualization.draw_geometries(
    [sphere_a, sphere_b]
)
