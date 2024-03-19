# -*- coding: utf-8 -*-
"""
Common functions to analyse 3D point and LiDAR fit
==================================================

@author: Thejasvi Beleyur
Code released under MIT License
"""
import numpy as np
import open3d as o3d


def run_pre_and_post_icp_steps(points, mesh, A, **kwargs):
    '''
    Performs important steps relevant to the paper.

    Parameters
    ----------
    points : list
        List with homogeneous coordinates x,y,z,1 of each point.
    mesh : pv.DataSet
    A : np.array
        4 x 4 np.array transformation matrix
    max_distance : float>0, optional
        See :code:`icp_register`

    Returns
    -------
    pre_post_distances : list
        List with sub-lists. First list is the distance between input points
        and nearest mesh point, second is the distance between transformed points
        and nearest mesh point.
    pre_post_points : list
        List with sub-lists. Each sub-list has the xyz points after first 
        (pre-icp) and second transformation (post-icp).
    posticp_transmat : (4,4) np.array

    See Also
    --------
    common_funcs.icp_register
    common_funcs.find_closest_points_distances
    '''
    mic_lidar = [np.matmul(A, each)[:-1] for each in points]
    mic_mesh_distance = find_closest_points_distances(mic_lidar, mesh)
    posticp_transmat = icp_register(mic_lidar, mesh, **kwargs)
    posticp_points = [np.matmul(posticp_transmat, np.append(each, 1))[:-1] for each in mic_lidar]
    mic_mesh_distance2 = find_closest_points_distances(posticp_points, mesh)
    pre_post_distances = [mic_mesh_distance, mic_mesh_distance2]
    pre_post_points = [mic_lidar, posticp_points]
    return pre_post_distances, pre_post_points, posticp_transmat


def find_closest_points_distances(points, tgt_mesh):
    '''

    Parameters
    ----------
    points : list
        List with np.arrays. Each np.array has 3 entries with xyz mic point
        positions
    tgt_mesh : pyvista.DataSet
        Output of pv.read.

    Returns
    -------
    mic_mesh_distance : list
        List with distances between given microphone positions and nearest
        mesh point.
    '''
    mic_mesh_distance = []
    for micpoint in points:
        index = tgt_mesh.find_closest_point(micpoint)
        mesh_xyz = tgt_mesh.points[index]
        diff = mesh_xyz - micpoint
        distance = np.linalg.norm(diff)
        mic_mesh_distance.append(distance)
    return mic_mesh_distance


def icp_register(mic_lidar, mesh, **kwargs):
    '''
    Performs ICP registration of points and outputs a transformation matrix.

    Parameters
    ----------
    mic_lidar : list
        List with np.array containing xyz of each mic position.
    mesh : pyvista.DataSet
        The output of pv.read - after loading the raw .ply file.
    max_distance : float>0, optional
        Max distance of mic positions from the LiDAR points. I think
        this helps the algorithm focus on which points to use to find the
        correspondences. Defaults to 0.5 m.

    Returns
    -------
    transformation_mat : np.array
        The transformation matrix which aligns the mic points to the LiDAR.
    '''
    threshold = kwargs.get('max_distance', 0.15)
    trans_init = np.eye(4)  # give a 'blank' identity transformation matrix
    mic_lidar_xyz = np.array(mic_lidar).reshape(-1, 3)

    source_pcd = o3d.geometry.PointCloud()
    source_pcd.points = o3d.utility.Vector3dVector(mic_lidar_xyz)
    target_pcd = o3d.geometry.PointCloud()
    target_pcd.points = o3d.utility.Vector3dVector(np.array(mesh.points))

    reg_p2p = o3d.pipelines.registration.registration_icp(
                    source_pcd, target_pcd, threshold, trans_init,
                    o3d.pipelines.registration.TransformationEstimationPointToPoint())

    transformation_mat = reg_p2p.transformation
    return transformation_mat
