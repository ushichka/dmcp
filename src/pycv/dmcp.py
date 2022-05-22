from turtle import distance
import numpy as np
import scipy.linalg as la
from sklearn.preprocessing import scale
from src.pycv.perspective import calibrate_dlt, solve_PnP, horn_affine_transformation
import cv2

def dm_to_world(dm: np.ndarray, dmK: np.ndarray, dmP: np.ndarray, dmPts: np.ndarray):
    if dmP.shape != (3,4):
        raise Exception(f"dmP shape must be 3,4 bit is {dmP.shape}")

    if dmK.shape != (3,3):
        raise Exception("dmK shape must be 3,3")

    if dmPts.shape[1] != 2:
        raise Exception("dmPts must have 2 columns")

    extrinsic_matrix = np.matmul(la.inv(dmK), dmP)
    extrinsic_matrix_hat = np.vstack((extrinsic_matrix,[0,0,0,1]))
        
    pose_matrix = la.inv(extrinsic_matrix_hat)[:3,:] 

    world_points = []
    for i in range(dmPts.shape[0]):
        px, py = dmPts[i,:]
        distance = dm[round(py), round(px)]
        pt_camera_space =  distance * np.matmul(la.inv(dmK), [px, py, 1])
        pt_camera_space_hat = np.append(pt_camera_space, 1)
        #print(f"dist {distance} cam_hat {pt_camera_space_hat}")

        pt_world_space = np.matmul(pose_matrix, pt_camera_space_hat)

        world_points.append(pt_world_space)

    world_points = np.array(world_points).astype("float32")
    return world_points



def dmcp(K_native: np.ndarray,P_native: np.ndarray, box_native_x_native: np.ndarray, box_world: np.ndarray, return_raw_pose=False):
    # box is annotated points
    if P_native.shape != (3,4):
        raise Exception(f"P_native shape must be 3,4 bit is {P_native.shape}")

    if K_native.shape != (3,3):
        raise Exception("K_native shape must be 3,3")

    if box_native_x_native.shape[1] != 2:
        raise Exception("box_native_x_native must have 2 columns")

    if box_world.shape[1] != 3:
        raise Exception("box_world must have 3 columns")

    
    def estimate_scaling(K1, P1, K2, P2):
        E1 = np.vstack((la.inv(K1) @ P1, [0, 0, 0, 1]))
        E2 = np.vstack((la.inv(K2) @ P2, [0, 0, 0, 1]))

        # ref: https://math.stackexchange.com/questions/237369/given-this-transformation-matrix-how-do-i-decompose-it-into-translation-rotati/417813
        sv_1 = [la.norm(E1[0:3, 0]), la.norm(E1[0:3, 1]), la.norm(E1[0:3, 2])] # vector of each norm of column in rotation matrix
        sv_2 = [la.norm(E2[0:3, 0]), la.norm(E2[0:3, 1]), la.norm(E2[0:3, 2])]

        sv_1= np.array(sv_1)
        sv_2 = np.array(sv_2)

        scale_factor = la.norm(sv_2) / la.norm(sv_1)
        return scale_factor





    # DMCP Step 1 calibrate camera in world space using annotations
    pose_matrix = solve_PnP(box_world,box_native_x_native,K_native)
    raw_pose = pose_matrix.copy()
    #P = calibrate_dlt(box_native_x_native, box_world)
    #extr = la.inv(K_native) @ P
    #extr_hat = np.vstack((extr,[0,0,0,1]))
    #pose_matrix = la.inv(extr_hat)[:3,:]

    # estimated pose
    print("estimated pose"),
    print(pose_matrix)
    pose_matrix_hat = np.vstack((pose_matrix, [0,0,0,1]))
    extrinsic_matrix_world = la.inv(pose_matrix_hat)[:3,:]
    P = K_native @ extrinsic_matrix_world

    # DMCP Step 2 compute registering transform
    # DMCP Step 2.1 transform world points into camera space
    
    scale_factor = estimate_scaling(K_native,P,K_native, P_native)
    print(f"scale factor {scale_factor}")

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

    retval, scale = cv2.estimateAffine3D(box_native_tf, box_world,force_rotation=False)
    A_tf = retval
    A_tf[:3,:3] = A_tf[:3,:3]* scale_factor

    A_tf_hat = np.vstack((A_tf,[0,0,0,1]))

    #print("retval\n",retval,"scale\n", scale)
    if return_raw_pose:
        return raw_pose, A_tf_hat

    return A_tf_hat
