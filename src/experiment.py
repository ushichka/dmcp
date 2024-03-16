from argparse import ArgumentError
import os
from cv2 import imread
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import scipy.linalg as la
import pyvista as pv
import imageio.v3 as iio
np.set_printoptions(formatter={"float": "{:10.2f}".format})
import trimesh as tr
import pyrender as pr
import hashlib
import math
import colorcet as cc
import click

from src.dmcpworkflow.capture_depth import generate_depth_map
from src.dmcpworkflow.annotate_points import annotate
from src.pycv.dmcp import dm_to_world
from src.pycv.dmcp import dmcp

def loadtxt_safe(path,delimiter=","):
    try:
        return np.loadtxt(path, delimiter=delimiter)
    except Exception as e:
        print(f"cannot read {path}\n{e}")
        return None

class Experiment:

    def __init__(self, dir,mesh_path) -> None:
        """
            workspace directory to load and store dmcp data
            Expects dmIm.csv, dmK.csv and dmP.csv containing a reference image with respective intrinsics and perspective projection matrix.

            (Without calibration, P = KE, where E=[R|T] is the extrinsic camera matrix can be ignored by setting E to the identity)
        """
        self.mesh_path = mesh_path
        self.exp_dir = dir

        self.path_lidar_hash = dir + os.sep + "lidar_hash.md5"
        ## captured images for depth map
        self.path_dmIm = dir + os.sep + "dmIm.csv"
        self.path_dmK = dir + os.sep + "dmK.csv"
        self.path_dmP = dir + os.sep + "dmP.csv"
        ## annotation paths
        self.path_imIm = dir + os.sep + "imIm.csv"
        self.path_cps = dir + os.sep + "cps.csv"
        ## alignment paths
        self.path_imK = dir + os.sep + "imK.csv"
        self.path_imP = dir + os.sep + "imP.csv"
        self.path_Pdlt = dir + os.sep + ".Pdlt.csv"
        self.path_transform = dir + os.sep + "transform.csv"
        ## reprojection error paths
        self.path_reprErrs = dir + os.sep + "reprErrs.csv"
        self.path_reprScatter = dir + os.sep + "reprScatter.png"
        self.path_reprBar = dir + os.sep + "reprBar.png"


# DEPTH MAP IO
    def load_dmIm(self):
            dmIm = loadtxt_safe(self.path_dmIm, delimiter=",")
            return dmIm
    def load_dmK(self):
        dmK = loadtxt_safe(self.path_dmK, delimiter=",")
        return dmK

    def load_dmP(self):
        dmP = loadtxt_safe(self.path_dmP, delimiter=",")
        return dmP

    def save_dmIm(self,dmIm: np.ndarray):
        dmIm = np.savetxt(self.path_dmIm,dmIm, delimiter=",")
        return dmIm

    def save_dmK(self,dmK: np.ndarray):
        dmK = np.savetxt(self.path_dmK,dmK,  delimiter=",")
        return dmK

    def save_dmP(self, dmP: np.ndarray):
        dmP = np.savetxt(self.path_dmP,dmP,  delimiter=",")
        return dmP
# IMAGE IO
    def load_imIm(self):
        imIm = loadtxt_safe(self.path_imIm, delimiter=",")
        return imIm

    def load_imK(self):
        imK = loadtxt_safe(self.path_imK, delimiter=",")
        return imK

    def load_imP(self):
        imP = loadtxt_safe(self.path_imP, delimiter=",")
        return imP

    def save_imIm(self, imIm : np.ndarray):
        imIm = np.savetxt( self.path_imIm,imIm, delimiter=",")
        return imIm

    def save_imK(self, imK: np.ndarray):
        imK = np.savetxt(self.path_imK,imK,  delimiter=",")
        return imK

    def save_imP(self, imP: np.ndarray):
        imP = np.savetxt( self.path_imP,imP,delimiter=",")
        return imP

# CPS IO

    def load_cps(self):
        cps = loadtxt_safe(self.path_cps, delimiter=",")
        return cps

    def save_cps(self, cps: np.ndarray):
        cps = np.savetxt(self.path_cps, cps,delimiter=",")
        return cps
# Mesh IO
    def save_hash(self):
        lidar_hash = hashlib.md5(open(self.mesh_path,'rb').read()).hexdigest()
        with open(self.path_lidar_hash, "w") as hashfile:
            hashfile.write(f"{lidar_hash}\n")

# Transform IO
    def load_transform(self):
        transform = loadtxt_safe(self.path_transform, delimiter=",")
        return transform

    def save_transform(self,transform: np.ndarray):
        transform = np.savetxt(self.path_transform,transform, delimiter=",")
        return transform

# METHODS

    def exec_dmcp(self, step=None,znear=0.1, zfar=100000):

        imIm = self.load_imIm()
        imK = self.load_imK()
        imP = self.load_imP()

        #if imIm == None or imK == None or imP == None:
        #    raise ArgumentError(f"directory '{self.exp_dir}' not valid expected at least imIm imK and imP to be present.")

        print("reading mesh")
        trmesh = tr.load_mesh(self.mesh_path)
        #print(type(trmesh))
        if type(trmesh) == tr.points.PointCloud:
            ovMesh = pr.Mesh.from_points(points=trmesh.vertices,colors = trmesh.colors)
        else:
            ovMesh = pr.Mesh.from_trimesh(trmesh)


        #%% dmcp workflow
        #%% generate depth map
        print("generating depth map")
        dmIm, dmK, dmP = generate_depth_map(ovMesh,imK,znear=znear,zfar=zfar)

        #%% annotate points
        print("annotate points")
        #mpl.use("TKAgg")
        cps = annotate(imIm, dmIm)

        #%% project to world
        
        pts_world = dm_to_world(dmIm, dmK, dmP, cps[:,2:])
        pts_world

        #%% dmcp step
        print("executing sparse correspondence alignment (SCA)")

        _raw_pose, trans = dmcp(imK, imP, cps[:,:2], pts_world,return_raw_pose=True)
        print(f"transformation\n{trans}")
        
        #%% save data
        self.save_dmIm(dmIm)
        self.save_dmK(dmK)
        self.save_dmP(dmP)
        self.save_hash()

        self.save_cps(cps)

        self.save_transform(trans)

    def visualize_3D(self):
        dmIm = self.load_dmIm()
        dmK = self.load_dmK()
        dmP = self.load_dmP()
        cps = self.load_cps()
        imP = self.load_imP()
        trans = self.load_transform()

        pts_world = dm_to_world(dmIm, dmK, dmP, cps[:,2:])

        P_est = imP @ la.inv(trans)
        pose_est = la.null_space(P_est) / la.null_space(P_est)[-1]

        position_est = pose_est[:3].flatten()
        print(position_est)

        sv_pos_est = pv.Sphere(radius=0.3, center=position_est)
        pvMesh = pv.read(self.mesh_path)
        pvPts = pv.PolyData(pts_world)
        pl = pv.Plotter(notebook=False)
        try:
            pl.add_mesh(pvMesh, rgb=True,scalars="RGB")
        except:
            pl.add_mesh(pvMesh)
        pl.add_mesh(pvPts, color="green", render_points_as_spheres=True,point_size=1)
        pl.add_mesh(sv_pos_est, color="blue")
        pl.show()

    
    def visualize_reprojection(self,block=True):
        imScatter = iio.imread(self.path_reprScatter)
        imBar = iio.imread(self.path_reprBar)


        plt.figure(f"reprojection errors {self.exp_dir}")
        plt.subplot(1,2,1)
        plt.imshow(imScatter,origin="upper")
        plt.subplot(1,2,2)
        plt.imshow(imBar,origin="upper")
        plt.axis("off")
        plt.show(block=block)
        plt.close()

    def compute_reprojection_error(self):
        dmIm = self.load_dmIm()
        dmK = self.load_dmK()
        dmP = self.load_dmP()
        cps = self.load_cps()
        imP = self.load_imP()
        imIm = self.load_imIm()
        trans = self.load_transform()
        imP = np.vstack((imP,np.array([0,0,0,1])))

        def dm_point_to_camera_point(x,y): 
            pointline = np.matmul(la.inv(dmK), np.array([x, y, 1]))
            depth = dmIm[round(y), round(x)]
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
        P_world_space_hat = np.matmul(imP, la.inv(trans))
        P_world_space = P_world_space_hat[:3,:]

        ns = la.null_space(P_world_space)
        ns = ns / ns[-1]

        # backproject annotated points to estimated camera
        projs_hat = [np.matmul(P_world_space, np.array([p[0], p[1], p[2], 1])) for p in annotated_world_points ]
        projs = [np.array([p[0], p[1]]) / p[2] for p in projs_hat]
        #projs = np.array(projs)

        # reprojection error
        repr_err = [ math.sqrt((projs[i][0] - cps[i,0])**2 + (projs[i][1] - cps[i,1])**2) for  i in range(len(projs))]
        projs = np.array(projs)

        # save reprojection error
        np.savetxt(self.path_reprErrs, np.array(repr_err),fmt="%05.2f")

        # visualize
        ## scatter
        fig = plt.figure()
        plt.imshow(imIm,origin="upper",cmap=cc.cm.gouldian)
        plt.scatter(cps[:,0], cps[:,1],marker="o", c="green", label="original annotation")
        plt.scatter(projs[:,0],projs[:,1], marker="x", c="red", label="backprojected annotation")
        plt.axis('off')
        plt.savefig(self.path_reprScatter,bbox_inches="tight", pad_inches=0,dpi=150)
        plt.close()

        ## Bar
        plt.figure()
        plt.bar(np.arange(len(repr_err)), repr_err, label="reprojection error")
        #plt.legend(loc="upper right")
        plt.axis("image")
        plt.savefig(self.path_reprBar, bbox_inches="tight",dpi=150)
        plt.close()

@click.command()
@click.argument("expdir", type=click.Path(exists=True),required=False)
@click.option("--mesh", type=click.Path(exists=True),required=True,envvar="MESH")
@click.option('--repr', 'action', flag_value='repr')
@click.option('--pose', 'action', flag_value='pose')
@click.option('--exec', 'action', flag_value='exec')
def cli(expdir, mesh, action):
    if expdir == None:
        expdir = os.getcwd()

    print(f"experiment {expdir}")
    exp = Experiment(expdir, mesh)
    #print(f"action {action}")
    if action == "repr":
        exp.compute_reprojection_error()
        exp.visualize_reprojection(True)
    if action == "pose":
        exp.visualize_3D()
    if action == "exec":
        exp.exec_dmcp()
        
if __name__ == "__main__":
    cli(auto_envvar_prefix="DMCP")