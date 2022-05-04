# -*- coding: utf-8 -*-
"""
Created on Mon Aug 30 21:30:47 2021

@author: Julian
"""

import os
import numpy as np
from scipy.io import loadmat


def loadCalibration(i, data_root):
    """ Uses calibration round 1
        Usage: from pyushichka import loadCalibration
    """
    data_root = data_root + os.sep + os.sep # otherwise basename might not work
    date = os.path.basename(os.path.dirname(data_root))
    path_dltCoefs = data_root + os.sep + "video_calibration" +os.sep + "calibration_output" + os.sep + "round1"+os.sep+f"{date}_round1_1pt1wandscore_dltCoefs.csv"
    path_easyWand = data_root + os.sep + "video_calibration" +os.sep + "calibration_output" + os.sep + "round1" + os.sep + f"{date}_round1_1pt1wandscore_easyWandData.mat"
    #extractIntrinsics()
    #print(date)
    #print(path_easyWand)
    
    #with h5.File(path_dvProject) as file_dvProject:
    #    file_dvProject = file_dvProject['udExport/data']

    file_easyWand = loadmat(path_easyWand, struct_as_record = False,squeeze_me=True)['easyWandData']
    
    #print(file_easyWand.principalPoints)

    K = extractIntrinsics(0, file_easyWand)
    P = extractProjection(0, path_dltCoefs)

    return K,P
    #print(K)
    #print(P)

def extractIntrinsics(i, wand):
    pp = [-1, -1]
    if   i == 0:
        pp = wand.principalPoints[0:2]
    elif i == 1:
        pp = wand.principalPoints[2:4]
    elif i == 2:
        pp = wand.principalPoints[4:6]
    else:
        raise Exception(f"invalid camera index {i}")
        
    f = wand.focalLengths[i]
    K = [[f,0,pp[0]],[0,f,pp[1]],[0,0,1]]
    K = np.array(K).astype(np.float32)
    return K

def extractProjection(i, path_dltCoefs):
    arr = np.loadtxt(path_dltCoefs,
                 delimiter=",", dtype=np.float32)
    
    P = np.append(arr[:,i],[1])
    P = np.reshape(P,(4,3)).T

    #np.set_printoptions(precision=3, suppress=True)

    return P
