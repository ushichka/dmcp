# -*- coding: utf-8 -*-
"""
Created on Mon Aug 30 21:30:47 2021

@author: Julian
"""

import os
import numpy as np
from pathlib import Path
from scipy.io import loadmat
import cv2
import scipy.linalg as la
import pathlib
import requests
from tqdm.auto import tqdm

def download_raw(target_dir: pathlib.Path):
    """ download complete ushichka data from "https://zenodo.org/api/records/6620671/files-archive"
    
        target_dir: directory to place downloaded files into (creates if not existing)
    """
    target_dir = pathlib.Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    url = "https://zenodo.org/api/records/6620671/files-archive"
    # Streaming, so we can iterate over the response.
    response = requests.get(url, stream=True)

    # Sizes in bytes.
    total_size = int(response.headers.get("content-length", 5153464980))
    block_size = 1024

    with tqdm(total=total_size, unit="B", unit_scale=True) as progress_bar:
        with open(target_dir / "raw.zip", "wb") as file:
            for data in response.iter_content(block_size):
                progress_bar.update(len(data))
                file.write(data)

    if total_size != 0 and progress_bar.n != total_size:
        raise RuntimeError(f"Could not download file total_size={total_size} progress_bar.n={progress_bar.n}")
    
    import zipfile
    with zipfile.ZipFile(target_dir / "raw.zip", 'r') as zip_ref:
        zip_ref.extractall(target_dir)

    # Removing the raw zip file
    os.remove(target_dir / "raw.zip")

    # Recursively unzip other files in the directory
    for root, _, files in os.walk(target_dir):
        for file in files:
            if file.endswith('.zip'):
                zip_ref = zipfile.ZipFile(os.path.join(root, file), 'r')
                zip_ref.extractall(root)
                zip_ref.close()
                os.remove(os.path.join(root, file))
    


def loadImage(camera,image, data_root):
    """ load image from data in ushichka directory 
        camera: number of camera (0-2)
        image: number of image ()
    """
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

def loadImageUndistorted(camera, image, data_root):
    image_distorted, _imPath = loadImage(camera, image, data_root)
    imK, _imP = loadCalibration(camera,data_root)
    dist = np.array([-0.3069,0.1134,0,0]) # in the opencv format [k1, k2, p1, p2, k3]) 
    dst = cv2.undistort(image_distorted,imK,dist,None,imK)
    return dst
    

def fix_rotation(K,P):
    extrinsic = la.inv(K) @ P
    shifter_mat = np.row_stack(([1,0,0],
                                [0,1,0],
                                [0,0,-1]))

    eRot = extrinsic[:3,:3] 

    eRot = eRot @ shifter_mat


    print(eRot, "\n",extrinsic[:3,-1])
    extrinsic_fixed = np.hstack((eRot, np.array([extrinsic[:3,-1]]).T))
    P_fixed = K @ extrinsic_fixed
    
    P_fixed = P.copy()
    P_fixed[:3,:3] = P_fixed[:3,:3] @ shifter_mat

    return P_fixed

def print_det(K,P):
    extrinsic = la.inv(K) @ P
    print("determinant",la.det(extrinsic[:3,:3]))

def loadCalibration(i, data_root):
    """ Uses calibration round 1
        Usage: from pyushichka import loadCalibration
        IMPORTANT: ´~/´ expansion does currently **NOT** work!
    """
    path_calib_out = data_root + os.sep + "video_calibration" +os.sep + "calibration_output"
    path_calib_out = list(pathlib.Path(path_calib_out).glob('round*'))[-1]
    path_dltCoefs = list(Path(path_calib_out).rglob('*_dltCoefs.csv'))[-1] # use newest
    path_easyWand = list(Path(path_calib_out).rglob('*_easyWandData.mat'))[-1]
    #extractIntrinsics()
    #print(date)
    #print(path_easyWand)
    
    #with h5.File(path_dvProject) as file_dvProject:
    #    file_dvProject = file_dvProject['udExport/data']

    file_easyWand = loadmat(path_easyWand, struct_as_record = False,squeeze_me=True)['easyWandData']
    #print(file_easyWand.principalPoints)

    K = extractIntrinsics(i, file_easyWand)
    P = extractProjection(i, file_easyWand)
    #print_det(K,P)
    #P = fix_rotation(K,P)
    #print_det(K,P)

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

def extractProjection(i, wand):
    #arr = np.loadtxt(path_dltCoefs,
    #             delimiter=",", dtype=np.float32)
    
    def cam_centre_from_dlt(coefs):
        '''
        
        
        Reference
        ---------
        * http://www.kwon3d.com/theory/dlt/dlt.html Equation 25
        '''
            
        m1 = np.array([[coefs[0],coefs[1],coefs[2]],
                    [coefs[4],coefs[5],coefs[6]],
                    [coefs[8],coefs[9],coefs[10]]])
        m2 = np.array([-coefs[3], -coefs[7], -1]).T

        xyz = np.matmul(np.linalg.inv(m1),m2)
        return xyz

    def transformation_matrix_from_dlt(coefs):
        '''
        Based on the DLTcameraPosition.m function written by Ty Hedrick. 
        
        Parameters
        ----------
        coefs : (11,Ncamera) np.array
        
        Returns
        -------
        T : (4,4) np.array
            Transformation matrix
        Z : float
            Distance of camera centre behind image plane
        ypr : (3,) np.array
            yaw, pitch, roll angles in degrees
        
        
        Notes
        -----
        I guess this function is based on the equations described in 
        Kwon3d (http://www.kwon3d.com/theory/dlt/dlt.html#3d).
                
        The transformation matrix T -
        
        
        
        ''' 
        D = (1/(coefs[8]**2+coefs[9]**2+coefs[10]**2))**0.5;
        #D = D[0]; # + solution
        
        Uo=(D**2)*(coefs[0]*coefs[8]+coefs[1]*coefs[9]+coefs[2]*coefs[10]);
        Vo=(D**2)*(coefs[4]*coefs[8]+coefs[5]*coefs[9]+coefs[6]*coefs[10]);
        #print(f'D: {D}, Uo: {Uo}, Vo:{Vo}')
        du = (((Uo*coefs[8]-coefs[0])**2 + (Uo*coefs[9]-coefs[1])**2 + (Uo*coefs[10]-coefs[2])**2)*D**2)**0.5;
        dv = (((Vo*coefs[8]-coefs[4])**2 + (Vo*coefs[9]-coefs[5])**2 + (Vo*coefs[10]-coefs[6])**2)*D**2)**0.5;
        
        #du = du[0]; # + values
        #dv = dv[0]; 
        Z = -1*np.mean([du,dv]) # there should be only a tiny difference between du & dv
        
        row1 = [(Uo*coefs[8]-coefs[0])/du ,(Uo*coefs[9]-coefs[1])/du ,(Uo*coefs[10]-coefs[2])/du]
        row2 = [(Vo*coefs[8]-coefs[4])/dv ,(Vo*coefs[9]-coefs[5])/dv ,(Vo*coefs[10]-coefs[6])/dv] 
        row3 = [coefs[8] , coefs[9], coefs[10]]
        T3 = D*np.array([row1,
                        row2,
                        row3])

        dT3 = np.linalg.det(T3);
        
        if dT3 < 0:
            T3=-1*T3;
        
        xyz = cam_centre_from_dlt(coefs)
        
        T = np.zeros((4,4))
        T[:3,:3] = np.linalg.inv(T3);
        T[3,:]= [xyz[0], xyz[1], xyz[2], 1]
        
            
        # % compute YPR from T3
        # %
        # % Note that the axes of the DLT based transformation matrix are rarely
        # % orthogonal, so these angles are only an approximation of the correct
        # % transformation matrix
        # %  - Addendum: the nonlinear constraint used in mdlt_computeCoefficients below ensures the
        # %  orthogonality of the transformation matrix axes, so no problem here
        alpha = np.arctan2(T[1,0],T[0,0])
        beta = np.arctan2(-T[2,0], (T[2,1]**2+T[2,2]**2)**0.5)
        gamma = np.arctan2(T[2,1],T[2,2])
        
        ypr = np.rad2deg([gamma,beta,alpha]);

        return T, Z, ypr

    def make_rotation_mat_fromworld(Rc, C):
        '''
        Parameters
        ----------
        Rc : 3x3 np.array
            Rotation matrix wrt the world coordinate system
        C : (1,3) or (3,) np.array
            Camera XYZ in world coordinate system
        Returns
        -------
        camera_rotation: 4x4 np.array
            The final camera rotation and translation matrix which 
            converts the world point to the camera point
        
        References
        ----------
        * Simek, Kyle, https://ksimek.github.io/2012/08/22/extrinsic/ (blog post) accessed 14/2/2022
        
        Example
        -------
        
        > import track2trajectory.synthetic_data as syndata
        > Rc, C = ..... # load and define the world system camera rotations and camera centre
        > rotation_mat = make_rotation_mat_fromworld(Rc, C)
        '''
        camera_rotation = np.zeros((4,4))
        camera_rotation[:3,:3] = Rc.T
        camera_rotation[:3,-1] = -np.matmul(Rc.T,C)
        camera_rotation[-1,-1] = 1 
        return camera_rotation

    #print(P_my.flatten()[:-1])
    coefs = wand.coefs[:,i]
    camera_pose_T , _, _ = transformation_matrix_from_dlt(coefs)
    shifter_mat = np.row_stack(([1,0,0,0],
                                [0,1,0,0],
                                [0,0,-1,0],
                                [0,0,0,1]))

    #print(camera_pose_T)
    shifted_rotmat = np.matmul(camera_pose_T, shifter_mat)[:3,:3]
    extrinsic_matrix = make_rotation_mat_fromworld(shifted_rotmat, camera_pose_T[-1,:3])


    extrinsic_matrix = (np.array([[1,0,0],[0,-1,0],[0,0,1]]) @ extrinsic_matrix[:3,:])

    #print(extrinsic_matrix)
    K = extractIntrinsics(i, wand)
    P = K @ extrinsic_matrix[:3,:]
    P

    return P
