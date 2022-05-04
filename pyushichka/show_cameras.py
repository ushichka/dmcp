import numpy as np
import scipy.linalg as la
import pyvista as pv


from load_data import loadCalibration

def from_homog(c):
    c = c / float(c[-1])
    return c[0:-1]

def extractDir(K,P):
    E = np.matmul(la.inv(K), P)
    E = np.vstack((E,[0,0,0,1])) 
    C = la.inv(E)
    return -1*C[:-1,2]

def extractCenter(P):
    C = la.null_space(P)
    C = from_homog(C)
    return C

def loadCalib(i, recording_path):
    K, P = loadCalibration(i, recording_path)
    dir = extractDir(K,P)
    pos = extractCenter(P)
    return pos, dir 

def createCone(pos, dir):
    return pv.Cone(center=pos, direction=dir)


path = "/home/julian/data/ushichka/2018-08-18"
colorlist = ["red", "green", "blue"]

p = pv.Plotter()

for i in range(3):
    pos, dir = loadCalib(i, path)
    cone = createCone(pos, dir)
    p.add_mesh(cone, show_edges=True, color=colorlist[i])

p.show()

