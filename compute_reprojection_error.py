import argparse
from operator import matmul
import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt
from matplotlib.backend_bases import MouseButton
import colorcet as cc

parser = argparse.ArgumentParser(description='show reprojection error between annotated world points and annotated image points')
parser.add_argument("--cps", required=True)
parser.add_argument("--dm", required=True)
parser.add_argument("--dmK", required=True)
parser.add_argument("--dmP", required=True)
parser.add_argument("--imP", required=True)
parser.add_argument("--transform", required=True)


args = parser.parse_args()

path_cps = args.cps
path_dm  = args.dm
path_dmK = args.dmK
path_dmP = args.dmP
path_imP = args.imP
path_transform = args.transform

cps: np.ndarray = np.loadtxt(path_cps,delimiter=",")
dm: np.ndarray = np.loadtxt(path_dm,delimiter=",")[-1:0:-1,:]
dmK: np.ndarray = np.loadtxt(path_dmK,delimiter=",")
dmP: np.ndarray = np.loadtxt(path_dmP,delimiter=",")
imP: np.ndarray = np.loadtxt(path_imP,delimiter=",")
transform: np.ndarray = np.loadtxt(path_transform,delimiter=",")

imP = np.vstack((imP,np.array([0,0,0,1])))

def dm_point_to_camera_point(x,y): 
    pointline = np.matmul(la.inv(dmK), np.array([x, y, 1]))
    depth = dm[round(y), round(x)]
    return depth * pointline

def extract_camera_pose_matrix(K,P):
    cam_ext_matrix = np.matmul(la.inv(K) ,P)
    cam_ext_matrix = np.vstack((cam_ext_matrix,np.array([0,0,0,1])))
    cam_pose_matrix = la.inv(cam_ext_matrix)
    cam_pose_matrix = cam_pose_matrix[0:3, :]
    return cam_pose_matrix
def camera_point_to_world_point(px,py,pz, K, P):
    wp = np.matmul(extract_camera_pose_matrix(K,P), np.array([px, py, pz, 1]))
    return wp

annotated_camera_points = [dm_point_to_camera_point(cps[i,2],cps[i,3]) for i in range(cps.shape[0])]
annotated_world_points = [camera_point_to_world_point(p[0],p[1],p[2],dmK,dmP) for p in annotated_camera_points]

# convert camera Projection matrix using estimated transform
P_world_space_hat = np.matmul(imP, la.inv(transform))
P_world_space = P_world_space_hat[:3,:]

# backproject annotated points to estimated camera
projs_hat = [np.matmul(P_world_space, np.array([p[0], p[1], p[2], 1])) for p in annotated_world_points ]
projs = [np.array([p[0], p[1]]) / p[2] for p in projs_hat]
projs = np.array(projs)
