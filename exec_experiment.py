import argparse
import os
import numpy as np
import subprocess
import hashlib
from src.pyushichka.load_data import loadCalibration, loadImage
import shutil
import matplotlib.pyplot as plt
import pathlib
import colorcet as cc

parser = argparse.ArgumentParser(description='execute dmcp on ushichka')
parser.add_argument('--dir', help="folder where data belonging to experiment gets stored")
parser.add_argument('--mesh', help="the mesh file to align to")
parser.add_argument('--recording', help="root folder (date) of ushichka recording")
parser.add_argument('--cam', help="index of camera to use")
parser.add_argument('--step', default=-1)

args = parser.parse_args()

cam = int(args.cam)
recording = args.recording
mesh_path = args.mesh
dir = args.dir
step=int(args.step)

# specify subdir for experiment named after recording
recording_name = pathlib.Path(recording).name
dir = dir + os.sep + str(recording_name)+"--"+f"cam{cam}"

# dir paths
## lidar md5 hash
path_lidar_hash = dir + os.sep + "lidar_hash.md5"
## captured images for depth map
path_dmIm = dir + os.sep + "dmIm.csv"
path_dmK = dir + os.sep + "dmK.csv"
path_dmP = dir + os.sep + "dmP.csv"
## annotation paths
path_imIm = dir + os.sep + "imIm.csv"
path_cps = dir + os.sep + "cps.csv"
## alignment paths
path_imK = dir + os.sep + "imK.csv"
path_imP = dir + os.sep + "imP.csv"
path_Pdlt = dir + os.sep + ".Pdlt.csv"
path_transform = dir + os.sep + "transform.csv"
## reprojection error paths
path_reprErrs = dir + os.sep + "reprErrs.csv"
path_reprScatter = dir + os.sep + "reprScatter.png"
path_reprBar = dir + os.sep + "reprBar.png"
path_reprErrsPdlt = dir + os.sep + ".reprErrsPdlt.csv"
path_reprScatterPdlt = dir + os.sep + ".reprScatterPdlt.png"
path_reprBarPdlt = dir + os.sep + ".reprBarPdlt.png"

# prepare experiment folder for full run
if step == -1:
    # make sure dir exists
    if not os.path.isdir(dir):
        os.makedirs(dir)

    # read image from ushichka and move to experiment dir
    img, im_path_orig = loadImage(cam, 0, recording)
    shutil.copyfile(im_path_orig, path_imIm)

    # create im calibration files
    K, P = loadCalibration(cam, recording)
    np.savetxt(path_imK, K, fmt='%4.6f', delimiter=',')
    np.savetxt(path_imP, P, fmt='%4.6f', delimiter=',')

    # store hash of lidar file for reference
    lidar_hash = hashlib.md5(open(mesh_path,'rb').read()).hexdigest()
    with open(path_lidar_hash, "w") as hashfile:
        hashfile.write(f"{lidar_hash}\n")

# capture depth map
if step == -1 or step == 1:
    print("STEP 1:")
    # also show reference image
    plt.figure("Preview for Orientation")
    img, im_path_orig = loadImage(cam, 0, recording)
    plt.imshow(img, cmap=cc.cm.get("gouldian_r"))
    plt.show(block=True)
    subprocess.run(["python", "dmcpworkflow/capture_depth.py", "--mesh", f"{mesh_path}", "--outIm", f"{path_dmIm}", "--outK", f"{path_dmK}", "--outP", f"{path_dmP}"])

# annotate cps
if step == -1 or step == 2:
    print("STEP 2:")
    subprocess.run(["python", "dmcpworkflow/annotate_points.py", "--im", f"{path_imIm}", "--dm", f"{path_dmIm}", "--out", f"{path_cps}"])

# execute alignment
if step == -1 or step == 3:
    print("STEP 3:")
    subprocess.run(["python", "--project=.", "exec_dmcp.jl", "--imK", f"{path_imK}", "--imP", f"{path_imP}", "--dmK", f"{path_dmK}", "--dmP", f"{path_dmP}", "--dmIm", f"{path_dmIm}", "--cps", f"{path_cps}", "--out", f"{path_transform}", "--outPdlt", f"{path_Pdlt}"])

# execute alignment
if step == -1 or step == 4:
    print("STEP 4:")
    # normal reprojection error
    subprocess.run([
        "python", "-m", "compute_reprojection_error", 
        "--cps", f"{path_cps}",
        "--dm", f"{path_dmIm}",
        "--dmK", f"{path_dmK}",
        "--dmP", f"{path_dmP}",
        "--im", f"{path_imIm}",
        "--imP", f"{path_imP}",
        "--transform", f"{path_transform}",
        "--outErrs", f"{path_reprErrs}",
        "--outScatter", f"{path_reprScatter}",
        "--outBar", f"{path_reprBar}"
        ])

    # compute reprojection error for PDlt only
    subprocess.run([
        "python", "-m", "compute_reprojection_error", 
        "--cps", f"{path_cps}",
        "--dm", f"{path_dmIm}",
        "--dmK", f"{path_dmK}",
        "--dmP", f"{path_dmP}",
        "--im", f"{path_imIm}",
        "--imP", f"{path_imP}",
        "--Pdlt", f"{path_Pdlt}",
        "--transform", f"{path_transform}",
        "--outErrs", f"{path_reprErrsPdlt}",
        "--outScatter", f"{path_reprScatterPdlt}",
        "--outBar", f"{path_reprBarPdlt}"
        ])