from turtle import distance
import numpy as np
import scipy.linalg as la
from perspective import solve_PnP, horn_affine_transformation

def dm_to_world(dm: np.ndarray, dmK: np.ndarray, dmP: np.ndarray, dmPts: np.ndarray):
    if dmP.shape != (3,4):
        raise Exception(f"dmP shape must be 3,4 bit is {dmP.shape}")

    if dmK.shape != (3,3):
        raise Exception("dmK shape must be 3,3")

    if dmPts.shape[1] != 2:
        raise Exception("dmPts must have 2 columns")


    world_points = []
    for i in range(dmPts.shape[0]):
        px, py = dmPts[i,:]
        pt_camera_space = dm[round(py), round(px)] * np.matmul(la.inv(dmK), [px, py, 1])
        pt_camera_space_hat = np.stack((pt_camera_space, [1]))
        extrinsic_matrix = np.matmul(la.inv(dmK), dmP)
        extrinsic_matrix_hat = np.vstack((extrinsic_matrix,[0,0,0,1]))
        pose_matrix = la.inv(extrinsic_matrix_hat)[:3,:]
        pt_world_space = np.matmul(pose_matrix, pt_camera_space_hat)

        world_points.append(pt_world_space)

    world_points = np.array(world_points).astype("float32")
    return world_points



def dmcp(K_native: np.ndarray,P_native: np.ndarray, box_native_x_native: np.ndarray, box_world: np.ndarray):
    # box is annotated points
    if P_native.shape != (3,4):
        raise Exception(f"P_native shape must be 3,4 bit is {P_native.shape}")

    if K_native.shape != (3,3):
        raise Exception("K_native shape must be 3,3")

    if box_native_x_native.shape[1] != 2:
        raise Exception("box_native_x_native must have 2 columns")

    if box_world.shape[1] != 3:
        raise Exception("box_world must have 3 columns")




    # DMCP Step 1 calibrate camera in world space using annotations
    pose_matrix = solve_PnP(box_world,box_native_x_native,K_native)
    pose_matrix_hat = np.vstack((pose_matrix, [0,0,0,1]))
    extrinsic_matrix_world = la.inv(pose_matrix_hat)[:3,:]

    # DMCP Step 2 compute registering transform
    # DMCP Step 2.1 transform world points into camera space
    box_world_hat = np.hstack((box_world, np.ones((box_world.shape[0],1))))

    box_world_camera = np.matmul(extrinsic_matrix_world, box_world_hat.T).T


    # DMCP Step 2.2 transform camera points into native space
    extrinsic_matrix_native = np.matmul(la.inv(K_native), P_native) 
    extrinsic_matrix_native_hat = np.vstack((extrinsic_matrix_native,[0,0,0,1]))
    pose_matrix = la.inv(extrinsic_matrix_native_hat)
    camera_pose_matrix_native = pose_matrix

    box_world_camera_hat = np.hstack((box_world_camera,np.ones((box_world_camera.shape[0],1))))
    box_native_tf = np.matmul(camera_pose_matrix_native, box_world_camera_hat.T).T[:,:3]

    # DMCP Step 2.3

    A_tf = horn_affine_transformation(box_native_tf, box_world)

    A_tf_hat = np.vstack((A_tf,[0,0,0,1]))

    return A_tf_hat
