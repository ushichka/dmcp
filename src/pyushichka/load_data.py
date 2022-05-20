# -*- coding: utf-8 -*-
"""
Created on Mon Aug 30 21:30:47 2021

@author: Julian
"""

import os
import numpy as np
from pathlib import Path
from scipy.io import loadmat

def loadImage(camera,image, data_root):
    camera = int(camera) +1 # expect 0 indexed but K1/2/3 are 1 indexed
    path_images = data_root + os.sep + "image" + os.sep + "raw_images"+ os.sep
    path_img_list = list(Path(path_images).rglob(f"*K{camera}*512x640shape.csv"))
    
    # the raw_image dir is not always there
    if len(path_img_list) != 0:
        path_img = path_img_list[image]
    else:
        path_images = data_root + os.sep + "image" + os.sep
        path_img_list = list(Path(path_images).rglob(f"*K{camera}*512x640shape.csv"))
        path_img = path_img_list[image]
    
    im = np.loadtxt(path_img,
                 delimiter=",", dtype=np.float32)
    return im, path_img
    


def loadCalibration(i, data_root):
    """ Uses calibration round 1
        Usage: from pyushichka import loadCalibration
        IMPORTANT: ´~/´ expansion does currently **NOT** work!
    """
    path_calib_out = data_root + os.sep + "video_calibration" +os.sep + "calibration_output" + os.sep + "round1" #TODO: use last intead of 1
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
