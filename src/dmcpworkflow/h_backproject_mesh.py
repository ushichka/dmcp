import numpy as np
import pyrender
import pyvista as pv
import scipy.linalg as la
import scipy.io as sio
from scipy.signal import medfilt2d
from scipy.spatial.transform import Rotation
import argparse
import math

def extract_pyrender_pinhole(v: pyrender.Viewer):
    cn : pyrender.Node = v._camera_node
    
    rot_quat = cn.rotation
    rot_mat = Rotation.from_quat(rot_quat).as_matrix()
    rote = [-180,0,0.0]
    r_corr = Rotation.from_euler("xyz",rote, degrees=True).as_matrix() # pyrender uses different convention
    rot_mat = rot_mat @ r_corr
    trans = np.array([cn.translation]).T
    pose = np.hstack((rot_mat, trans))
    print(pose)

    fx = cn.camera.fx
    fy = cn.camera.fy
    cx = cn.camera.cx
    cy = cn.camera.cy
    K = [
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0,  1]
        ]
    K = np.array(K)
    return K, pose

def get_interactive_camera(mesh: pyrender.Mesh, K :np.ndarray):
    scene = pyrender.Scene(ambient_light=np.ones(4))
    _meshnode = scene.add(mesh)
    cx = K[0, 2]
    cy = K[1, 2]
    fx = K[0, 0]
    fy = K[1, 1]

    cam = pyrender.camera.IntrinsicsCamera(fx=fx, fy=fy, cx=cx, cy=cy,znear=10, zfar=10000000.0)
    r = Rotation.from_euler("xyz",[180,0,0.0], degrees=True).as_matrix()
    t = np.array([[0,0,0]]).T
    pose = np.hstack((r,t))
    pose = np.vstack((pose,[0,0,0,1]))
    scene.add(cam,pose=pose)

    v = pyrender.Viewer(scene, run_in_thread=True)

    while v.is_active:
        pass

    

    K, pose = extract_pyrender_pinhole(v)
    height = v.height
    width = v.width
    shape = (height, width)
    return K, pose, shape

def capture_scene(mesh: pyrender.Mesh, K: np.ndarray, R: np.ndarray, T: np.ndarray, width: int, height: int):
    intrinsics = K
    cx = intrinsics[0, 2]
    cy = intrinsics[1, 2]
    fx = intrinsics[0, 0]
    fy = intrinsics[1, 1]
    cam = pyrender.IntrinsicsCamera(fx = fx, fy =fy, cx=cx, cy=cy,znear=10, zfar=10000000.0)
    cam_orig = cam
    rote = [180,0,0.0]
    r = Rotation.from_euler("xyz",rote, degrees=True).as_matrix()
    r = R @ r
    t = np.array([T]).T
    print(r)
    print(t)
    pose = np.hstack((r,t))
    pose = np.vstack((pose,[0,0,0,1]))
    #
    #print(cam_pose_orig)
    #print(pose)

    #==============================================================================
    l = 1.0
    scene = pyrender.Scene(ambient_light=np.array([l, l, l, l]))
    #cam_node = scene.add(cam_orig, pose=cam_pose_orig)
    cam_node = scene.add(cam_orig, pose=pose)


    #==============================================================================
    # Rendering offscreen from that camera
    #==============================================================================

    meshnode = scene.add(mesh)
    #v = Viewer(scene, central_node=drill_node)
    s = 3
    r = pyrender.OffscreenRenderer(viewport_width=width, viewport_height=height)
    color, depth = r.render(scene)
    r.delete()

    return np.asarray(color), np.asarray(depth)

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
