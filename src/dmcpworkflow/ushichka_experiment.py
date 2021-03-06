#%% imports
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

def ushichka_experiment(ushichka_dir, mesh_path, experiment_dir,cam,step):

    exp = Experiment(experiment_dir, mesh_path)
    if step == None or step == 0:
        imK, imP = loadCalibration(cam, ushichka_dir)
        imIm = loadImageUndistorted(cam, 0, ushichka_dir) # 0 is first image
        exp.save_imIm(imIm)
        exp.save_imK(imK)
        exp.save_imP(imP)
        plt.figure()
        plt.imshow(imIm)
        plt.show()

    if step == None or step == 1:
        exp.exec_dmcp()

    if step == None or step == 2:
        exp.compute_reprojection_error()
        plt.show()
        #exp.visualize_reprojection()

@click.command()
@click.argument('experiment_dir',
              type=click.Path(exists=False))
@click.argument('ushichka_dir',
              type=click.Path(exists=True))
@click.argument('cam',
                type=click.IntRange(0,2))
@click.option('--mesh',
              type=click.Path(exists=True),required=True)
@click.option("--step", default=None, type=click.IntRange(0,2))

def cli(experiment_dir,ushichka_dir,cam,mesh,step):
    recording_name = pathlib.Path(ushichka_dir).name
    experiment_dir = experiment_dir + os.sep + str(recording_name)+"--"+f"cam{cam}"
    if not os.path.isdir(experiment_dir):
        os.makedirs(experiment_dir)
    ushichka_experiment(ushichka_dir, mesh, experiment_dir, cam,step)

if __name__ == "__main__":
    cli(auto_envvar_prefix="DMCP")
