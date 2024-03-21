# -*- coding: utf-8 -*-
"""
Comparing DMCP to some other methods to align cameras and point-clouds
======================================================================
JJ has performed alignment with various other algorithms and here we'll
'groundtruth' their performance using the fact that the microphone array in 
the Ushichka dataset was either 'on' the cave walls or captured as part of the 
inverted-T array of the LiDAR scan. 

The LiDAR scan data scan and camera derived microphone xyz positions are
all available at this link: https://zenodo.org/records/6620671

@author: Thejasvi Beleyur
Code released under MIT License
"""
import glob
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np 
import os
import pandas as pd
import pyvista as pv
import seaborn as sns
from common_funcs import find_closest_points_distances, icp_register
from common_funcs import run_pre_and_post_icp_steps

#%%
# First load the triangulated mesh which represents a sub-section of the cave
# The lidar_roi.ply is in the uploaded Zenodo folder (see link above)
mesh = pv.read('..\\..\\projectushichka\\thermo_lidar_alignment\\data\\lidar_roi.ply')
#%%
# Search around for all the transformation matrices and store the paths 
# in one dictionary 

experiment_nights = glob.glob('..\\experiments\\2018*')

transform_matrices = {}
for expt_night in experiment_nights:
    for root, dirs, files in os.walk(expt_night):
        #foldername = os.path.split(root)[-1]
        date_method = root.split('\\')[2]
        for name in files:
            if 'transform' in name:
                if transform_matrices.get(date_method) is None:
                    transform_matrices[date_method] = []
                fullpath = os.path.join(root,name)
                transform_matrices[date_method].append(fullpath)

#%%
# Search for all the mic xyz positions.
# Instead of the given path here - put the path the downloaded folder.
camera_reconstr_folder = os.path.join('..','..',
                                      'projectushichka',
                                      'thermo_lidar_alignment',
                                      'dmcp_mic_alignment_for_ms')
micxyz_pts = {}
surfacexyz_pts = {}
for root, dirs, filenames in os.walk(camera_reconstr_folder):
    for filename in filenames:
        mics_and_xyz_in_filename = np.logical_and('mic' in filename, 'xyz' in filename)
        surface_in_filename = 'surface' in filename
        if mics_and_xyz_in_filename:
            expt_date = filename.split('_')[2]
            # fix a typo in the data
            if '2081' in expt_date:
                expt_date = expt_date.replace('2081', '2018')
            micxyz_pts[expt_date] = os.path.join(root,filename)
        if surface_in_filename:
            surfacexyz_pts[expt_date] = os.path.join(root, filename)
#%%
import time
points_distance_preicp = {}
points_distance_posticp= {}

preicp_xyz_points = {}
posticp_xyz_points = {}

if not os.path.exists('mic-to-cave-results'):
    os.mkdir('mic-to-cave-results')


experiment_dates = [os.path.split(X)[-1] for X in experiment_nights]

preicp_results = {}
posticp_results = {}

for each_night in experiment_dates:
    print(each_night)
    points_distance_preicp[each_night] = []
    points_distance_posticp[each_night] = []
    preicp_xyz_points[each_night] = []
    posticp_xyz_points[each_night] = []
    # mic points xyz
    mic_file = micxyz_pts[each_night]
    mic_3dpoints = pd.read_csv(mic_file).dropna()
    # cave points xyz
    cave3dpoints = surfacexyz_pts.get(each_night)
    if cave3dpoints is not None:
        cave_3dpoints = pd.read_csv(surfacexyz_pts[each_night]).dropna()
        points3d = pd.concat([mic_3dpoints, cave_3dpoints]).reset_index(drop=True)
    else:
        points3d = mic_3dpoints.copy()
    
    mic_xyz = [points3d.loc[each,:].to_numpy() for each in points3d.index]
    mic_xyzh = [np.append(each, 1) for each in mic_xyz]
    for each_camera_transmat in transform_matrices[each_night]:
        method = each_camera_transmat.split('\\')[3]
        identifier = each_night+'_'+method
        
        
        print(each_camera_transmat)
        st = time.time()
        # Now move the mic from camera calibration space to LiDAR space.
        try:
            A = pd.read_csv(each_camera_transmat, header=None,delim_whitespace=True).to_numpy()
            np.multiply(A,np.ones((4,4))) # test if the data has been parsed correctly
        except:
            A = pd.read_csv(each_camera_transmat, header=None,delimiter=',').to_numpy()
            np.multiply(A,np.ones((4,4)))
        
        pre_post_dist, preposticp_xyz, icp_refine_transmat = run_pre_and_post_icp_steps(mic_xyzh,
                                                                                    mesh, A,
                                                                                    max_distance=1.5)
        preicp_distances, posticp_distances = pre_post_dist
        
        # points_distance_preicp[each_night].append(preicp_distances)
        # points_distance_posticp[each_night].append(posticp_distances)
        preicp_xyz, posticp_xyz = preposticp_xyz
        
        
        dataset = pd.DataFrame(data=None,
                               index=range(len(preicp_xyz)),
                               columns=['method-date','prex','prey','prez','predist','postx','posty','postz','postdist'])
        dataset['method-date'] = identifier
        dataset.loc[:,'prex':'prez'] = np.array(preicp_xyz).reshape(-1,3)
        dataset.loc[:,'postx':'postz'] = np.array(posticp_xyz).reshape(-1,3)
        dataset.loc[:,'predist'] = preicp_distances
        dataset.loc[:,'postdist'] = posticp_distances
        
        dataset.to_csv(os.path.join('mic-to-cave-results', f'{identifier}_results.csv'))
        
        # preicp_xyz_points[each_night].append(preicp_xyz)
        # posticp_xyz_points[each_night].append(posticp_xyz)
        print(f'pre-post icp median dists to scan: {np.median(preicp_distances), np.median(posticp_distances)}')
        print(f'one evening one cam run time: {time.time()-st} \n')
