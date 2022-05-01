# -*- coding: utf-8 -*-
"""
Mesh and image
==============
Module to simultaneously visualise and annotate the 3D mesh and 2D
image.

@author: Thejasvi Beleyur
Created on Sun May  1 07:29:35 2022
"""
import argparse
import pandas as pd
import os
import pyvista as pv
import matplotlib.pyplot as plt
from matplotlib.backend_bases import MouseButton


#%% Define the input files and paths

# mesh_path = 'demo/lidar_roi.ply'
# image_path = 'demo/imIm.png'


parser = argparse.ArgumentParser(description='Annotate corresponding mesh and image points')
parser.add_argument('--mesh', help='Path to mesh file')
parser.add_argument('--image', help='Path to image file (csv or image)')
parser.add_argument('--sep', default=',', help='Which separator to use to parse csv file. Defaults to ;')
parser.add_argument('--output_path', default='', help='Where the mesh_annotated_points and image_annotated_points will be saved. Defaults to working directory')
args = parser.parse_args()

mesh_path = args.mesh
image_path = args.image
output_path = args.output_path
#%% 
print(f'HIIII {image_path}, {image_path[-3:]}')
if image_path[-3:]=='csv':
    img_data = pd.read_csv(image_path, header=None, sep=args.sep).to_numpy(dtype='float64')
else:
    img_data = plt.imread(image_path)

#%%

point_num_mesh = 1
point_num_image = 1 

plotter = pv.Plotter()
mesh_points = {}
image_points = {}

def label_mesh_points(*xyz):
    global point_num_mesh
    global mesh_points 
    data, point_id = xyz
    centre = cave_mesh.points[point_id]
    mesh_points[point_num_mesh] = centre
    #globe = pv.Sphere(0.05,center=centre)
    #sphere = plotter.add_mesh(globe)
    try:
        plotter.add_point_labels(centre, f'{point_num_mesh}',
                                 italic=True, font_size=20,
                               point_color='red', point_size=20,
                               render_points_as_spheres=True,
                               always_visible=True, shadow=True)
        print(f'Mesh point {point_num_mesh} picked: {centre}')
        point_num_mesh += 1
    except ValueError:
        print(f'Try again {xyz}')

cave_mesh = pv.read(mesh_path)
plotter.add_mesh(cave_mesh)
plotter.enable_point_picking(use_mesh=True, callback=label_mesh_points)

#%% Image part of things
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
        print(f'Image point {point_num_image} picked: {(x,y)}')
        point_num_image += 1 
if __name__ == '__main__':
    plt.figure()
    plt.imshow(img_data)
    plt.connect('button_press_event', on_click)
    plt.title('Right-click mouse to annotate points.')
    plt.show(block=False)
    
    # Display the window
    plotter.show()
    
    # check if both dictionaries have the same length
    img_x, img_y = [], []
    for key, data in image_points.items():
        x,y = data
        img_x.append(x)
        img_y.append(y)
    point_x, point_y, point_z = [], [], []
    for pointnum, xyz in mesh_points.items():
        x,y,z = xyz
        point_x.append(x)
        point_y.append(y)
        point_z.append(z)

    # save into csv files
    mesh_annotated = pd.DataFrame(data={'x':point_x, 'y':point_y,
                                        'z':point_z})
    mesh_annotated.to_csv(os.path.join(output_path, 
                                       'mesh_points_annotated.csv'))
    
    image_annotated = pd.DataFrame(data={'x':img_x, 'y':img_y})
    image_annotated.to_csv(os.path.join(output_path,
                                        'image_points_annotated.csv'))
    

