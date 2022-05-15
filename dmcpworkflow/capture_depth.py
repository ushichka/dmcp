from PIL import Image
from h_backproject_mesh import main as capture_depth
import math
import pyvista as pv
import argparse
import numpy as np
import scipy.misc
import scipy.io

def generate_depth_map(mesh_path):
    mesh = pv.read(mesh_path)
    plotter = pv.Plotter(off_screen=False, notebook=False)
    actor = plotter.add_mesh(mesh, color="grey")
    def clicked(event):
        up = plotter.camera.GetViewUp()
        forward = plotter.camera.GetDirectionOfProjection()
        right = np.cross(forward, up)
        right = right / np.linalg.norm(right)
        T = plotter.camera.GetPosition()

    plotter.track_click_position(callback=clicked, side='left', viewport=True)

    w, h = plotter.window_size

    plotter.show()


    T = plotter.camera.GetPosition()
    T = [T[0], T[1], T[2]]

    up = plotter.camera.GetViewUp()
    forward = plotter.camera.GetDirectionOfProjection()
    right = np.cross(forward, up)
    right = right / np.linalg.norm(right)
    u = up
    f = forward
    r = right
    R = np.array([[r[0], -u[0], f[0]],
                [r[1], -u[1], f[1]],
                [r[2], -u[2], f[2]]])

    C = np.hstack((R, np.array([T]).T))
    C = np.vstack((C, [0, 0, 0, 1]))


    E = np.linalg.inv(C)
    E = E[0:3, 0:4]

    K = np.array([[526, 0, 320], [0, 526, 256], [0, 0, 1]])

    wcx, wcy = plotter.camera.GetWindowCenter()

    cx = w * wcx/-2+float(w)/2
    cy = h * wcy/-2+float(h)/2

    # convert the focal length to view angle and set it
    view_angle = plotter.camera.GetViewAngle()

    # it was (2* ,math.tan...) but 1*... seems to work
    f_x = -w / (1 * math.tan(view_angle/2.0))
    f_y = -h / (1 * math.tan(view_angle/2.0))

    K = np.array([[f_x, 0, cx],
                [0, f_y, cy],
                [0, 0,  1]])


    P = np.matmul(K, E)

    # capture photo
    n_rows = h
    n_cols = w


    depth_map = capture_depth(mesh_path, P, K, n_rows, n_cols, False)

    return depth_map, K, P

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

    depth_map, K, P = generate_depth_map(mesh_path)

    # save as array
    np.savetxt(args.outIm, depth_map, delimiter=",")
    np.savetxt(args.outK, K, delimiter=",")
    np.savetxt(args.outP, P, delimiter=",")

    print(f"captured data saved to:\n -- {args.outIm}\n -- {args.outK}\n -- {args.outP}")
