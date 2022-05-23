from argparse import ArgumentError
import os
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
import click

from src.dmcpworkflow.capture_depth import generate_depth_map
from src.dmcpworkflow.annotate_points import annotate
from src.pycv.dmcp import dm_to_world
from src.pycv.dmcp import dmcp


class Experiment:

    def __init__(self, dir,mesh_path) -> None:
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
            dmIm = np.loadtxt(self.path_dmIm, delimiter=",")
            return dmIm
    def load_dmK(self):
        dmK = np.loadtxt(self.path_dmK, delimiter=",")
        return dmK

    def load_dmP(self):
        dmP = np.loadtxt(self.path_dmP, delimiter=",")
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
        imIm = np.loadtxt(self.path_imIm, delimiter=",")
        return imIm

    def load_imK(self):
        imK = np.loadtxt(self.path_imK, delimiter=",")
        return imK

    def load_imP(self):
        imP = np.loadtxt(self.path_imP, delimiter=",")
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
        cps = np.loadtxt(self.path_cps, delimiter=",")
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
        transform = np.loadtxt(self.path_transform, delimiter=",")
        return transform

    def save_transform(self,transform: np.ndarray):
        transform = np.savetxt(self.path_transform,transform, delimiter=",")
        return transform

# METHODS

    def exec_dmcp(self):

        imIm = self.load_imIm()
        imK = self.load_imK()
        imP = self.load_imP()

        #if imIm == None or imK == None or imP == None:
        #    raise ArgumentError(f"directory '{self.exp_dir}' not valid expected at least imIm imK and imP to be present.")

        print("reading mesh")
        ovMesh = pr.Mesh.from_trimesh(tr.load_mesh(self.mesh_path))


        #%% dmcp workflow
        #%% generate depth map
        print("generating depth map")
        dmIm, dmK, dmP = generate_depth_map(ovMesh,imK,znear=0.00001,zfar=100)

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

    def visualize_dmcp(self):
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


        sv_pos_est = pv.Sphere(radius=0.25, center=position_est)
        pvMesh = pv.read(self.mesh_path)
        pvPts = pv.PolyData(pts_world)
        pl = pv.Plotter(notebook=False)
        pl.add_mesh(pvMesh, color="dimgrey")
        pl.add_mesh(pvPts, color="lightblue", render_points_as_spheres=True,point_size=25)
        pl.add_mesh(sv_pos_est, color="yellowgreen")
        pl.show()

@click.command()
@click.argument("expdir", type=click.Path(exists=True),required=False)
@click.option("--mesh", type=click.Path(exists=True),required=True)
@click.option("--show", is_flag=True, default=False, help="visualize experiment")
def cli(expdir, mesh, show):
    if expdir == None:
        expdir = os.getcwd()

    exp = Experiment(expdir, mesh)
    if show:
        exp.visualize_dmcp()

if __name__ == "__main__":
    cli()