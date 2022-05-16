import os
import numpy as np

class Experiment:

    def __init__(self, dir) -> None:
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

    def load_dmIm(self):
            dmIm = np.loadtxt(self.path_dmIm, delimiter=",")
            return dmIm
    def load_dmK(self):
        dmK = np.loadtxt(self.path_dmK, delimiter=",")
        return dmK

    def load_dmP(self):
        dmP = np.loadtxt(self.path_dmP, delimiter=",")
        return dmP

    def load_imK(self):
        imK = np.loadtxt(self.path_imK, delimiter=",")
        return imK

    def load_imP(self):
        imP = np.loadtxt(self.path_imP, delimiter=",")
        return imP

    def load_cps(self):
        cps = np.loadtxt(self.path_cps, delimiter=",")
        return cps

    def load_imIm(self):
        imIm = np.loadtxt(self.path_imIm, delimiter=",")
        return imIm