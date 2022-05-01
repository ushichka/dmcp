# -*- coding: utf-8 -*-
"""
Mesh and image
==============
Module to simultaneously visualise and annotate the 3D mesh and 2D
image.

@author: thejasvi
Created on Sun May  1 07:29:35 2022
"""
import pyvista as pv
from pyvista import examples
import matplotlib.pyplot as plt

mesh_path = 'demo/lidar_roi.ply'
image_path = 'demo/imIm.png'

def callback(x):
    print(x)
    print(f'camera position: {plotter.camera.position}')
    print(f'camera az,rol,elev: {plotter.camera.azimuth},{plotter.camera.roll},\
          {plotter.camera.elevation}')
    print(f'camera view angle, focal point: {plotter.camera.view_angle,plotter.camera.focal_point}')
    return x

# enable_point_picking stores the picked point in 
# plotter.picked_point !!
def smallsphere(*xyz):
    data, point_id = xyz
    centre = cave_mesh.points[point_id]
    globe = pv.Sphere(0.05,center=centre)
    sphere = plotter.add_mesh(globe)
    print(centre)

plotter = pv.Plotter()
plotter.add_text("Render Window 0", font_size=30)
cave_mesh = pv.read(mesh_path)
plotter.add_mesh(cave_mesh)
plotter.enable_point_picking(use_mesh=True, callback=smallsphere)


# plotter.subplot(0, 1)
# image = pv.read(image_path)
# #plotter.add_text("Render Window 1", font_size=30)
# plotter.add_mesh(image)

# plotter.subplot(1, 0)
# plotter.add_text("Render Window 2", font_size=30)
# sphere = pv.Sphere()
# plotter.add_mesh(sphere, scalars=sphere.points[:, 2])
# plotter.add_scalar_bar("Z")
# # plotter.add_axes()
# plotter.add_axes(interactive=True)

# plotter.subplot(1, 1)
# plotter.add_text("Render Window 3", font_size=30)
# plotter.add_mesh(pv.Cone(), color="g", show_edges=True)
# plotter.show_bounds(all_edges=True)


plt.figure()
ffim = plt.imread(image_path)
plt.imshow(ffim)

# Display the window
plotter.show()

