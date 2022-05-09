import argparse
from operator import matmul
import numpy as np
import scipy.linalg as la
import math
import matplotlib.pyplot as plt
from matplotlib.backend_bases import MouseButton
import colorcet as cc

parser = argparse.ArgumentParser(description='show reprojection error between annotated world points and annotated image points')
parser.add_argument("--cps", required=True)
parser.add_argument("--dm", required=True)
parser.add_argument("--dmK", required=True)
parser.add_argument("--dmP", required=True)
parser.add_argument("--im", required=True)
parser.add_argument("--imP", required=True)
parser.add_argument("--transform", required=True)
parser.add_argument("--Pdlt")
parser.add_argument("--outErrs", required=True)
parser.add_argument("--outScatter", required=True)
parser.add_argument("--outBar", required=True)

args = parser.parse_args()

path_cps = args.cps
path_dm  = args.dm
path_dmK = args.dmK
path_dmP = args.dmP
path_im  = args.im
path_imP = args.imP
path_transform = args.transform
path_Pdlt = args.Pdlt
path_outErrs = args.outErrs
path_outScatter = args.outScatter
path_outBar = args.outBar

cps: np.ndarray = np.loadtxt(path_cps,delimiter=",")
dm: np.ndarray = np.loadtxt(path_dm,delimiter=",")[-1:0:-1,:]
dmK: np.ndarray = np.loadtxt(path_dmK,delimiter=",")
dmP: np.ndarray = np.loadtxt(path_dmP,delimiter=",")
im: np.ndarray = np.loadtxt(path_im,delimiter=",")[-1:0:-1,:]
imP: np.ndarray = np.loadtxt(path_imP,delimiter=",")
transform: np.ndarray = np.loadtxt(path_transform,delimiter=",")
Pdlt = None
if path_Pdlt != None:
    Pdlt: np.ndarray = np.loadtxt(path_Pdlt,delimiter=",")


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

if type(Pdlt) != type(None):
    P_world_space = Pdlt

ns = la.null_space(P_world_space)
ns = ns / ns[-1]
print(f"P_trans_null \n{ns}")

# backproject annotated points to estimated camera
projs_hat = [np.matmul(P_world_space, np.array([p[0], p[1], p[2], 1])) for p in annotated_world_points ]
projs = [np.array([p[0], p[1]]) / p[2] for p in projs_hat]
#projs = np.array(projs)

# reprojection error
repr_err = [ math.sqrt((projs[i][0] - cps[i,0])**2 + (projs[i][1] - cps[i,1])**2) for  i in range(len(projs))]
projs = np.array(projs)

# save reprojection error
np.savetxt(path_outErrs, np.array(repr_err),fmt="%05.2f")

# visualize
## scatter
plt.figure()
plt.imshow(im,origin="lower",cmap=cc.cm.gouldian)
plt.scatter(cps[:,0], cps[:,1],marker="o", c="green", label="original annotation")
plt.scatter(projs[:,0],projs[:,1], marker="x", c="red", label="backprojected annotation")
plt.legend(loc="upper right")
plt.savefig(path_outScatter, dpi=300)

## Bar
plt.figure()
plt.bar(np.arange(len(repr_err)), repr_err, label="reprojection error")
plt.legend(loc="upper right")
plt.savefig(path_outBar, dpi=300)

print(f"reprojection error data saved to:\n -- {path_outErrs}\n -- {path_outScatter}\n -- {path_outBar}")