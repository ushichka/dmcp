import plyfile
from tqdm.auto import tqdm
import numpy as np

def write_ply(filename, points):
    """writes point cloud as ply"""
    num_points = points.shape[0]

    # Write PLY header
    header = (
        "ply\n"
        "format ascii 1.0\n"
        f"element vertex {num_points}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "end_header\n"
    )

    # Write vertices (points)
    with open(filename, "w") as f:
        f.write(header)
        for point in points:
            f.write(f"{point[0]:.6f} {point[1]:.6f} {point[2]:.6f}\n")

def load_ply(filename):
    """load ply as point cloud"""

    # Load PLY file
    plydata = plyfile.PlyData.read(filename)

    # Extract vertices (points)
    vertices = plydata['vertex']
    num_points = len(vertices)
    
    # Extract x, y, z coordinates of points
    points = []
    for i in tqdm(range(num_points)):
        x = vertices[i][0]
        y = vertices[i][1]
        z = vertices[i][2]
        points.append([x, y, z])

    return np.array(points)