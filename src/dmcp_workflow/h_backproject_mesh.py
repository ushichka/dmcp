import numpy as np
import pyvista as pv
import scipy.linalg as la
import scipy.io as sio
from scipy.signal import medfilt2d
import argparse
import math


def trans_to_matrix(trans):
    """ Convert a numpy.ndarray to a vtk.vtkMatrix4x4 """
    return pv.vtkmatrix_from_array(trans)


def make_vtk_camera(w, h, intrinsic, extrinsic, plotter):
    """reference: https://github.com/pyvista/pyvista/issues/1215"""

    plotter.window_size = [w, h]

    #
    # intrinsics
    #

    cx = intrinsic[0, 2]
    cy = intrinsic[1, 2]
    f = intrinsic[0, 0]

    # convert the principal point to window center (normalized coordinate system) and set it
    wcx = -2*(cx - float(w)/2) / w
    wcy = 2*(cy - float(h)/2) / h
    plotter.camera.SetWindowCenter(wcx, wcy)

    # convert the focal length to view angle and set it
    view_angle = 180 / math.pi * (2.0 * math.atan2(h/2.0, f))
    plotter.camera.SetViewAngle(view_angle)

    #
    # extrinsics
    #

    # apply the transform to scene objects
    plotter.camera.SetModelTransformMatrix(trans_to_matrix(extrinsic))

    # the camera can stay at the origin because we are transforming the scene objects
    plotter.camera.SetPosition(0, 0, 0)

    # look in the +Z direction of the camera coordinate system
    plotter.camera.SetFocalPoint(0, 0, 1)

    # the camera Y axis points down
    plotter.camera.SetViewUp(0, -1, 0)

    #
    # near/far plane
    #

    # ensure the relevant range of depths are rendered
    # depth_min = 0.1
    # depth_max = 100
    # p.camera.SetClippingRange(depth_min, depth_max)
    # # depth_min, depth_max = p.camera.GetClippingRange()
    plotter.renderer.ResetCameraClippingRange()

    # p.show()
    # p.render()
    plotter.store_image = True  # last_image and last_image_depth

# generate depth map for all cameras


def capture_depth(mesh, P, K, n_rows, n_cols):
    plotter = pv.Plotter(off_screen=True, notebook=False)
    actor = plotter.add_mesh(mesh, color="grey")

    E = la.inv(K) @ P[0:3, :]
    E = np.vstack((E, [0, 0, 0, 1]))

    make_vtk_camera(n_cols, n_rows, K, E, plotter)

    plotter.show()
    depth_img = plotter.get_image_depth()
    depth_img = depth_img * -1  # values are negative when read from plotter
    return depth_img

    # set invalid to nan
    # !Note: assumes maximum values are invalid!
    y, x = np.where(depth_img == np.nanmax(depth_img))
    depth_img[y, x] = np.nan
    return depth_img


def filter_nan(I):
    I = I.copy()
    I = medfilt2d(I, 5)

    mask = np.isnan(I)
    I[mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), I[~mask])
    return I


def main(filename, P, K, n_rows, n_cols, fn=True):
    mesh = pv.read(filename)  # example r'data/formatted/lidar_roi.ply'
    depth_img = capture_depth(mesh, P, K, n_rows, n_cols)
    if fn:
        depth_img = filter_nan(depth_img)

    return depth_img


def toNpMatrix(string):
    return eval("np.array("+string+")")


if __name__ == "__main__":

    dsc = "Takes depth_image of mesh using given pinhole camera as model. Maximum depth will be set to nan."
    parser = argparse.ArgumentParser(description=dsc)

    parser.add_argument('filename', help='mesh file')
    parser.add_argument('P', help='pinhole projection matrix')
    parser.add_argument('K', help='camera intrinsic matrix')
    parser.add_argument('n_rows', help="height of image", type=int)
    parser.add_argument('n_cols', help="width of image", type=int)

    parser.add_argument("-fn", "--FilterNan", default=True,
                        help="If True, NaN values are interpolated.")

    parser.add_argument("-o", "--out", default="backprojected.mat",
                        help="output matlab file", required=False)

    args = parser.parse_args()
    depth_img = main(args.filename, toNpMatrix(args.P), toNpMatrix(args.K),
                     args.n_rows, args.n_cols, args.FilterNan)

    sio.savemat(args.out, {"depth_map": depth_img})

    #import matplotlib.pyplot as plt
    # plt.figure()
    #plt.imshow(depth_img, origin="lower")
    # plt.show()