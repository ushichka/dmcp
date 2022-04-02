from julia import Julia
jl = Julia(compile='min')
from julia import Main as jl
import scipy.io
import scipy.misc
import numpy as np

# demo data from thesis
mat = scipy.io.loadmat("demo/dmcp_inputs_demo.mat")

# call julia implementation
jl.include("src/dmcp_alg.jl")
A = jl.exec_dmcp(mat["K_th"].astype(np.float64), mat["P_th"].astype(np.float64), mat["I_dm"].astype(np.float32), mat["K_dm"].astype(np.float64), mat["P_dm"].astype(np.float64), mat["cps"].astype(np.float64))

print(A)

## show mesh
import pyvista as pv
mesh_path = "C:/Users/Julian/Nextcloud/Uni/Depth for Thermal Images/data_raw/lidar/lidar_roi.ply"
mesh = pv.read(mesh_path)

plotter = pv.Plotter(off_screen=False, notebook=False)
actor = plotter.add_mesh(mesh, color="grey")
#plotter.camera.SetPosition((0,0,0))

# look in the +Z direction of the camera coordinate system
#plotter.camera.SetFocalPoint(0, 0, 1)

# the camera Y axis points down
#plotter.camera.SetViewUp(0, -1, 0)

def clicked(event):
    up = plotter.camera.GetViewUp()
    forward = plotter.camera.GetDirectionOfProjection()
    right = np.cross(forward, up)
    right = right / np.linalg.norm(right)
    T = plotter.camera.GetPosition()
    print("pos ",T)
    print("up ", up)
    print("forward " , forward)
    print("right ", right)

plotter.track_click_position(callback=clicked,side='left', viewport=True)

w, h = plotter.window_size

plotter.show()


T = plotter.camera.GetPosition()
T = [T[0], T[1], T[2]]

#R = np.array(   [[1, 0, 0],
#                 [0, 1, 0],
#                 [0, 0, 1]])

up = plotter.camera.GetViewUp()
forward = plotter.camera.GetDirectionOfProjection()
right = np.cross(forward, up)
right = right / np.linalg.norm(right)
u = up
f = forward
r = right
R = np.array(  [[r[0],-u[0],f[0]],
                [r[1],-u[1],f[1]],
                [r[2],-u[2],f[2]]])

print("pos ",T)
print("up ", up)
print("forward " , forward)
print("right ", right)


C = np.hstack((R, np.array([T]).T))
C = np.vstack((C,[0, 0, 0, 1]))

print("C \n", C)

E = np.linalg.inv(C)
E = E[0:3,0:4]
print("E \n", E)

K = np.array([[526, 0, 320],[0, 526, 256],[0,0,1]])

import math
#w, h = plotter.last_image.shape
wcx, wcy = plotter.camera.GetWindowCenter()

cx =  w*  wcx/-2+float(w)/2
cy =  h*  wcy/-2+float(h)/2

# convert the focal length to view angle and set it
view_angle = plotter.camera.GetViewAngle()
print("va", w, h)

f_x = -w / (2 * math.tan(view_angle/2.0))
f_y = -h / (2 * math.tan(view_angle/2.0))

K = np.array(  [[f_x, 0, cx],
                [0, f_y, cy],
                [0, 0,  1]])

print("K\n",K)


P = np.matmul(K, E)

print("P \n", P)

# capture photo
from h_backproject_mesh import main as capture_depth
n_rows = 512
n_cols = 640


depth_map = capture_depth(mesh_path,P,K,n_rows,n_cols, False)

# show plot
#import matplotlib.pyplot as plt
#plt.figure()
#plt.imshow(depth_map)
#plt.show()

# save as array
np.savetxt('export/dm.csv', depth_map, delimiter=",")

# save as image
from PIL import Image
im = Image.fromarray(depth_map)
im = im.convert('RGBA')
im.save('export/dm.png', "PNG")

print("DATA SAVED TO export/")
print("closing program...")