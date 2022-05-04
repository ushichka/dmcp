# -*- coding: utf-8 -*-
"""
Created on Mon Aug 30 21:30:47 2021

@author: Julian
"""

import os
import numpy as np
from pathlib import Path
from scipy.io import loadmat


def loadCalibration(i, data_root):
    """ Uses calibration round 1
        Usage: from pyushichka import loadCalibration
    """
    path_calib_out = data_root + os.sep + "video_calibration" +os.sep + "calibration_output" + os.sep + "round1" 
    path_dltCoefs = list(Path(path_calib_out).rglob('*_dltCoefs.csv'))[-1] # use last for legacy reasons
    path_easyWand = list(Path(path_calib_out).rglob('*_easyWandData.mat'))[-1]
    #extractIntrinsics()
    #print(date)
    #print(path_easyWand)
    
    #with h5.File(path_dvProject) as file_dvProject:
    #    file_dvProject = file_dvProject['udExport/data']

    file_easyWand = loadmat(path_easyWand, struct_as_record = False,squeeze_me=True)['easyWandData']
    
    #print(file_easyWand.principalPoints)

    K = extractIntrinsics(i, file_easyWand)
    P = extractProjection(i, path_dltCoefs)

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
    P = np.reshape(P,(3,4))

    #np.set_printoptions(precision=3, suppress=True)

    return P
