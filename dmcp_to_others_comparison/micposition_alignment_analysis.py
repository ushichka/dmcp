# -*- coding: utf-8 -*-
"""
Mic alignment: thermal-LiDAR
============================

Some thoughts:
    * It seems like the 3D alignment errors and 2D reprojection errors
    are rather correlated - does this make sense?
    * The transforms from diff cameras have points on slightly 'tilted'
    planes - or on completely different planes. 

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
mesh = pv.read('lidar/lidar_roi.ply')
#%%
experiment_nights = glob.glob('2018*/')
trans_mats = {}
for each_night in experiment_nights:
    file_template = os.path.join(each_night,'**','transform.csv')
    trans_mats[each_night] = glob.glob(file_template)

#%%
import time
points_distance_preicp = {}
points_distance_posticp= {}

preicp_xyz_points = {}
posticp_xyz_points = {}
for each_night in experiment_nights:
    print(each_night)
    points_distance_preicp[each_night] = []
    points_distance_posticp[each_night] = []
    preicp_xyz_points[each_night] = []
    posticp_xyz_points[each_night] = []
    # mic points xyz
    mic_file = glob.glob(os.path.join(each_night, '*mic*xyzpts.csv'))
    mic_3dpoints = pd.read_csv(mic_file[0]).dropna()
    # cave points xyz
    cavepoint_file = glob.glob(os.path.join(each_night, '*surface*xyz*.csv'))
    if len(cavepoint_file)>0:
        cave_3dpoints = pd.read_csv(cavepoint_file[0]).dropna()
        points3d = pd.concat([mic_3dpoints, cave_3dpoints]).reset_index(drop=True)
    else:
        points3d = mic_3dpoints.copy()
    mic_xyz = [mic_3dpoints.loc[each,:].to_numpy() for each in mic_3dpoints.index]
    mic_xyzh = [np.append(each, 1) for each in mic_xyz]
    for each_camera_transmat in trans_mats[each_night]:
        print(each_camera_transmat)
        st = time.time()
        A = pd.read_csv(each_camera_transmat, header=None).to_numpy()
        # Now move the mic from camera calibration space to LiDAR space.
        pre_post_dist, preposticp_xyz, icp_refine_transmat = run_pre_and_post_icp_steps(mic_xyzh,
                                                                                    mesh, A,
                                                                                    max_distance=1.5)
        preicp_distances, posticp_distances = pre_post_dist
        points_distance_preicp[each_night].append(preicp_distances)
        points_distance_posticp[each_night].append(posticp_distances)
        preicp_xyz, posticp_xyz = preposticp_xyz
        preicp_xyz_points[each_night].append(preicp_xyz)
        posticp_xyz_points[each_night].append(posticp_xyz)
        print(f'one evening one cam run time: {time.time()-st}')

#%% 
# Save all the   
reformatted_preicp_dists = {}
reformatted_posticp_dists = {}

for date, multi_camera_dists in points_distance_preicp.items():
    for i,camera_dists in enumerate(multi_camera_dists):
        keyname = date+ f'camera_{i}'
        reformatted_preicp_dists[keyname] = -999*np.ones(25)
        reformatted_preicp_dists[keyname][:len(camera_dists)] = camera_dists


for date, multi_camera_dists in points_distance_posticp.items():
    for i,camera_dists in enumerate(multi_camera_dists):
        keyname = date+ f'camera_{i}'
        reformatted_posticp_dists[keyname] = -999*np.ones(25)
        reformatted_posticp_dists[keyname][:len(camera_dists)] = camera_dists

preicp_dists = pd.DataFrame(data=reformatted_preicp_dists)
preicp_dists = preicp_dists.apply(lambda x: np.where(x==-999, np.nan, x))
preicp = pd.melt(preicp_dists)

posticp_dists = pd.DataFrame(data=reformatted_posticp_dists)
posticp_dists = posticp_dists.apply(lambda x: np.where(x==-999, np.nan, x))
posticp = pd.melt(posticp_dists)
# calculate median distances
preicp_median = preicp.groupby(by='variable').apply(np.nanmedian)
posticp_median = posticp.groupby(by='variable').apply(np.nanmedian)

#%%
preicp['date'] = preicp['variable'].apply(lambda X: X[:10])
ninety_pctile_distances = np.nanpercentile(preicp['value'], [2.5, 50, 90, 97.5])
# date-wise stats
preicp.groupby('date').apply(lambda X: np.nanpercentile(X['value'], [ 100]))
#%%
num_entries = lambda X: len(X['value'][~np.isnan(X['value'])])

#%% Data format for these plots
# The Pandas DataFrame is expectd to be a 'long' format dataframe with 
# one row per entry. Here 'variable' is the column holding the evening\camera id
# and value is the actual distance. 
# For e.g. variable holds [2018-07-21\camera_0,2018-07-21\camera_0,2018-07-21\camera_1,
# 2018-07-21\camera_1...]
# corresponding to each of the mic/cave points on 2018-07-21

plt.figure(figsize=(7,3)) # width, height
sns.stripplot(x='variable', y='value', data=preicp,  edgecolor='none', jitter=True,
              alpha=0.5, size=3)
for i,med in enumerate(preicp_median):
    plt.hlines(med, i-0.25, i+0.25)
sns.despine(); plt.yscale('log')
plt.yticks(ticks=[1e-3, 1e-2, 1e-1, 5e-1, 1, 1e1],
           labels=[0.001, 0.01, 0.1, 0.5, 1, 10],
           fontsize=9)
plt.ylabel('Distance to nearest\n mesh point, m', fontsize=9, labelpad=-10);
plt.xticks(range(21),['1','2','3']*7, fontsize=9); plt.xlim(-0.5,20.5)
plt.xlabel('Date & camera ID', labelpad=22)
date_x = np.arange(-0.2, 18, 3)
all_samplesizes = list(preicp.groupby('variable').apply(num_entries))[::3]
for x, samplesize, night in zip(date_x, all_samplesizes, experiment_nights):
    plt.text(x, 1e-4, night[:-1]+f'\n({samplesize})', fontsize=9,
             multialignment='center')
plt.tight_layout()
plt.savefig('preicp_meshdistances.eps')


plt.figure(figsize=(7,3))
sns.stripplot(x='variable', y='value', data=posticp,  edgecolor='none', jitter=True,
              alpha=0.5, size=3)
for i,med in enumerate(posticp_median):
    plt.hlines(med, i-0.25, i+0.25)
sns.despine(); plt.yscale('log')
plt.yticks(ticks=[1e-3, 1e-2, 1e-1, 5e-1, 1, 1e1],
           labels=[0.001, 0.01, 0.1, 0.5, 1, 10],
           fontsize=9)
plt.ylabel('Distance to nearest\n mesh point, m', fontsize=9, labelpad=-10);
plt.xticks(range(21),['1','2','3']*7, fontsize=9); plt.xlim(-0.5,20.5)
plt.xlabel('Date & camera ID', labelpad=22)
date_x = np.arange(-0.2, 18, 3)
all_samplesizes = list(posticp.groupby('variable').apply(num_entries))[::3]
for x, samplesize, night in zip(date_x, all_samplesizes, experiment_nights):
    plt.text(x, 1e-4, night[:-1]+f'\n({samplesize})', fontsize=9,
             multialignment='center')
plt.tight_layout()
plt.savefig('posticp_meshdistances.eps')

#%%
# TODO:
# Get the mean XYZ from each of the transforms for 2018-08-17 and plot each point
# in the mesh view.

colour_set = ['orange','green','red']
#for night, cavepts in preicp_xyz_points.items():
night = list(preicp_xyz_points.keys())[7]
cavepts = preicp_xyz_points[night]
plotter = pv.Plotter()
plotter.add_mesh(mesh, show_edges=False, color=True, opacity=0.5)
plotter.camera.position = (5.0, -1.02, -1)
plotter.camera.azimuth = 5
plotter.camera.roll = -90
plotter.camera.elevation = 5 #-15
plotter.camera.view_angle = 45
for i, each in enumerate(cavepts):
    for every in each:
        plotter.add_mesh(pv.Sphere(radius=0.05, center=every),
                                   color=colour_set[i])
plotter.add_text('DMCP mic+cave points alignment '+night[:-1], 'lower_right')
plotter.save_graphic(night[:-1]+'_cavepts.pdf')
plotter.show()
plotter.clear()

#%% 
# How far apart are the microphones when compared across the different cameras?

