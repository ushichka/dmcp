#%% imports
from email.mime import base
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import scipy.linalg as la
import pyvista as pv
import imageio.v3 as iio

from src.pyushichka.load_data import loadImageUndistorted
np.set_printoptions(formatter={"float": "{:10.2f}".format})
import trimesh as tr
import pyrender as pr
from src.pyushichka import loadCalibration, loadImage
from src.experiment import Experiment
import click
import pathlib
import os
import re

import cv2
#import open3d as o3d
import trimesh


def load_calib(file_path):
    with open(file_path,'r') as file:
        lines = file.readlines()
        K = parse_intrinsic(lines[0])
        doofs = parse_doffs(lines[2])
        baseline = parse_baseline(lines[3])
        return K, doofs, baseline

def parse_intrinsic(line):
     found_numbers = list(re.findall(r"(\d+\.?\d*)",line.strip()))[1:]
     K = np.array([float(n) for n in found_numbers]).reshape(3,3)
     return K

def parse_doffs(line):
    doffs = re.findall(r"(\d+\.?\d*)",line.strip())[0]
    doffs = float(doffs)
    return doffs

def parse_baseline(line):
    baseline = re.findall(r"(\d+\.?\d*)",line.strip())[0]
    baseline = float(baseline)
    return baseline

def load_dm(K, doffs, baseline,file_path):
    pfm = iio.imread(file_path,plugin="PFM-FI").astype(np.float32)

    f = (K[0,0] + K[1,1]) / 2.0 # assume equality of f in x and y

    for y in range(pfm.shape[0]):
        for x in range(pfm.shape[1]):
            disp = pfm[y,x]
            if disp != 0:
                Z = baseline * f / (disp + doffs)
            else:
                Z = np.nan
            pfm[y,x] = Z

    dm = np.flip(pfm,axis=0).copy()
    return dm

def load_data(dir_path):
    dir_path = pathlib.Path(dir_path)
    dispm_path = dir_path.joinpath("disp0.pfm")
    img_path = dir_path.joinpath("im0.png")
    calib_path = dir_path.joinpath("calib.txt")

    K, doffs, baseline = load_calib(calib_path)
    dm = load_dm(K, doffs, baseline, dispm_path)
    im = iio.imread(img_path)
    im = np.asarray(im)
    return K, im, dm

def create_middlebury_mesh(K,im,dm):
    pts = []
    invK = la.inv(K)
    for y in range(dm.shape[0]):
        for x in range(dm.shape[1]):
            dist = dm[y,x]
            pt = dist * (invK @ [x,y,1])
            pts.append(pt)

    pts = np.array(pts)
    #pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts))
    colors = np.reshape(im,(-1,3))#/ 255.0
    #print(colors)
    #pcd.colors = o3d.utility.Vector3dVector(o3d.utility.Vector3dVector(colors))
    #äprint("estimating normals")
    #pcd.estimate_normals();
    #print("computing alpha shape")
    #mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd,10000)
    #tMesh = trimesh.Trimesh(np.asarray(mesh.vertices), np.asarray(mesh.triangles),vertex_normals=np.asarray(mesh.vertex_normals),vertex_colors=colors)
    #tMesh = trimesh.Trimesh(vertices = pts, vertex_colors=colors)
    #return tMesh
    #print("delunay")
    step = 1
    cloud = pv.PolyData(pts[::step,:])
    cloud["RGB"] = colors[::step,:]
    #volume = cloud.delaunay_3d(alpha=200.)
    #print("delunay finished")
    #mesh = volume.extract_geometry()
    return cloud #mesh
    #tMesh = trimesh.Trimesh(np.asarray(mesh.verts), np.asarray(mesh.faces),vertex_colors=colors)
    #return tMesh
    #tMesh.export("/data/middlebury/middlebury.ply")
    
@click.group()
def cli():
    pass

@cli.command()
@click.argument('scene',
              type=click.Path(exists=True))
@click.option('--mesh',
              type=click.Path(exists=False),required=True,envvar="MESH")
def create_mesh(scene, mesh):
    K, im, dm = load_data(scene)

    tMesh = create_middlebury_mesh(K,im,dm)
    #tMesh.export(mesh)
    tMesh.save(mesh,binary=True,texture="RGB")


@cli.command()
@click.argument('scene',
              type=click.Path(exists=True))
@click.argument('out',
              type=click.Path(exists=False))
@click.option('--mesh',
              type=click.Path(exists=False),required=True,envvar="MESH")
def experiment(scene, out, mesh):
    if not os.path.isdir(out):
        os.makedirs(out)
    
    K, im, dm = load_data(scene)
    P = K @ np.eye(4)[:3,:]

    im_gray = cv2.cvtColor(im,cv2.COLOR_RGB2GRAY)

    exp = Experiment(out, mesh)
    exp.save_imIm(im_gray)
    exp.save_imK(K)
    exp.save_imP(P)
    plt.figure()
    plt.imshow(im)
    plt.show()

    exp.exec_dmcp()

    exp.compute_reprojection_error()
    plt.show()


if __name__ == "__main__":
    cli(auto_envvar_prefix="DMCP")
