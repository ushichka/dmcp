#%% imports
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import scipy.linalg as la
import pyvista as pv
import imageio.v3 as iio
np.set_printoptions(formatter={"float": "{:10.2f}".format})
import trimesh as tr
import pyrender as pr

#%% load data
from src.pyushichka import loadCalibration, loadImage
ushichka_dir = "/home/julian/data/ushichka/2018-08-19/"
cam = 1 # 2nd camera
imK, imP = loadCalibration(cam, ushichka_dir)
imIm, _path_im = loadImage(cam, 0, ushichka_dir)

print(f"intrinsics\n{imK}\npinhole projection matrix\n{imP}\n")
plt.figure()
plt.imshow(imIm)
plt.show()

from src.dmcpworkflow.capture_depth import generate_depth_map
mesh_path = "/home/julian/code/uni/ushichka-registration/data/formatted/lidar_roi.ply"
print("reading mesh")
ovMesh = pr.Mesh.from_trimesh(tr.load_mesh(mesh_path))


#%% Visualize Mesh

#s = pr.Scene()
#s.add(ovMesh)
#l = pr.light.PointLight(color=(0.5,0.25,0.125),intensity=15)
#s.add(l)
#
#v = pr.Viewer(s)


#%% dmcp workflow
print("generating depth map")
dmIm, dmK, dmP = generate_depth_map(ovMesh,imK,znear=0.00001,zfar=100)