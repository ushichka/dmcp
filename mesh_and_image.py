# -*- coding: utf-8 -*-
"""
Mesh and image
==============
Module to simultaneously visualise and annotate the 3D mesh and 2D
image.

TODO: 
    > Implement point picking on image
    > Implement point deletion in mesh
    

@author: thejasvi
Created on Sun May  1 07:29:35 2022
"""
import pyvista as pv
from pyvista import examples
import matplotlib.pyplot as plt
from matplotlib.backend_bases import MouseButton

mesh_path = 'demo/lidar_roi.ply'
image_path = 'demo/imIm.png'

point_num_mesh = 1
point_num_image = 1 

def callback(x):
    print(x)
    print(f'camera position: {plotter.camera.position}')
    print(f'camera az,rol,elev: {plotter.camera.azimuth},{plotter.camera.roll},\
          {plotter.camera.elevation}')
    print(f'camera view angle, focal point: {plotter.camera.view_angle,plotter.camera.focal_point}')
    return x

plotter = pv.Plotter()
mesh_points = {}
image_points = {}
# enable_point_picking stores the picked point in 
# plotter.picked_point !!
def label_mesh_points(*xyz):
    global point_num_mesh
    global mesh_points 
    data, point_id = xyz
    centre = cave_mesh.points[point_id]
    mesh_points[point_num_mesh] = centre
    #globe = pv.Sphere(0.05,center=centre)
    #sphere = plotter.add_mesh(globe)
    plotter.add_point_labels(centre, f'{point_num_mesh}',
                             italic=True, font_size=20,
                           point_color='red', point_size=20,
                           render_points_as_spheres=True,
                           always_visible=True, shadow=True)

    point_num_mesh += 1
    print(centre)

def on_click(event):
    global point_num_image
    global image_points
    if event.button is MouseButton.RIGHT:
        x,y = event.xdata, event.ydata
        print('hi',x,y)
        ax = plt.gca()
        ax.plot(x,y,'r*')
        plt.text(x+0.5, y+0.5, f"{point_num_image}")
        image_points[point_num_image] = (x,y)
        point_num_image += 1 
        


cave_mesh = pv.read(mesh_path)
plotter.add_mesh(cave_mesh)
plotter.enable_point_picking(use_mesh=True, callback=label_mesh_points)


plt.figure()
ffim = plt.imread(image_path)
plt.imshow(ffim)
plt.connect('button_press_event', on_click)
plt.title('Right-click mouse to annotate points.')

# Display the window
plotter.show()

# check if both dictionaries have the same length

# format both dictionaries. 

