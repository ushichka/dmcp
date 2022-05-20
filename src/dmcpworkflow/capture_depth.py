from PIL import Image
from .h_backproject_mesh import get_interactive_camera, capture_scene
import math
import pyvista as pv
import argparse
import numpy as np
import scipy.misc
import scipy.io
import scipy.linalg as la

def generate_depth_map(mesh,imK):
    K, pose, shape = get_interactive_camera(mesh,imK)
    img, depth = capture_scene(mesh, imK, pose[:3,:3], pose[:3,-1], shape[1], shape[0])
    extr = la.inv(np.vstack((pose,[0,0,0,1])))[:3,:]
    P = K @ extr
    return depth, K, P

if __name__ == "__main__":
    # cli args
    parser = argparse.ArgumentParser(description='create depth map from mesh')
    #parser.add_argument('mesh_path', metavar='N', type=str, nargs='?', default="C:/Users/Julian/Nextcloud/Uni/Depth for Thermal Images/data_raw/lidar/lidar_roi.ply",
    #                    help='the path for mesh file')
    parser.add_argument('--mesh')
    parser.add_argument('--outIm')
    parser.add_argument('--outK')
    parser.add_argument('--outP')

    args = parser.parse_args()

    mesh_path = args.mesh
    mesh = pv.read(mesh_path)
    depth_map, K, P = generate_depth_map(mesh)

    # save as array
    np.savetxt(args.outIm, depth_map, delimiter=",")
    np.savetxt(args.outK, K, delimiter=",")
    np.savetxt(args.outP, P, delimiter=",")

    print(f"captured data saved to:\n -- {args.outIm}\n -- {args.outK}\n -- {args.outP}")
