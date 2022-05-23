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

mesh_path = "/home/julian/code/uni/ushichka-registration/data/formatted/lidar_roi.ply"

experiment_dir = "/tmp/exp"

#%% setup experiment
from src.experiment import Experiment
exp = Experiment(experiment_dir, mesh_path)
exp.save_imIm(imIm)
exp.save_imK(imK)
exp.save_imP(imP)

exp.exec_dmcp()

exp.visualize_dmcp()

