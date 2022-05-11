#%%
import numpy as np
import scipy.linalg as la

def calibrate_dlt(img_pts, world_pts):
    """ each row one point """
    
    # build knows system matrix 2k x 12 A that determines coefficients
    rows = []
    for i in range(img_pts.shape[0]):

        # 2D Point
        u = img_pts[i,0]
        v = img_pts[i,1]

        # 3D Point
        X = world_pts[i,0]
        Y = world_pts[i,1]
        Z = world_pts[i,2]

        rows.append([-X, -Y, -Z, -1, 0, 0, 0, 0, u * X, u * Y, u * Z ,u])
        rows.append([0, 0, 0, 0, -X, -Y, -Z, -1, v * X, v * Y, v * Z ,v])
    rows = np.array(rows)
    A = rows


    # solve homogeneous linear system Ax = 0. x Represent the coefficients of the camera matrix
    # solve in least squares sense regarding reprojection error using SVD 
    U, S, V = la.svd(A)
    V = V.T
    #P = V[-1,:]
    #P = np.array(P)
    #P = np.reshape(P,(3,4))
    P = np.reshape(V[:, -1], (3, 4)) # 11 is 12th element

    return P

def decompose_perspective_projection_matrix(P):
    H = P[:3,:3]
    Q, R = np.linalg.qr(np.linalg.inv(H))
    # K, R = linalg.rq(H)
    Rotation = Q.T
    K = np.linalg.inv(R)
    K = K/K[2,2]
    R = Rotation
    T = np.dot(-1*np.linalg.inv(H), P[:,3])
    return K, R, T


def rot2eul(R):
    beta = -np.arcsin(R[2,0])
    alpha = np.arctan2(R[2,1]/np.cos(beta),R[2,2]/np.cos(beta))
    gamma = np.arctan2(R[1,0]/np.cos(beta),R[0,0]/np.cos(beta))
    return np.array((alpha, beta, gamma))

def eul2rot(theta) :

    R = np.array([[np.cos(theta[1])*np.cos(theta[2]),       np.sin(theta[0])*np.sin(theta[1])*np.cos(theta[2]) - np.sin(theta[2])*np.cos(theta[0]),      np.sin(theta[1])*np.cos(theta[0])*np.cos(theta[2]) + np.sin(theta[0])*np.sin(theta[2])],
                  [np.sin(theta[2])*np.cos(theta[1]),       np.sin(theta[0])*np.sin(theta[1])*np.sin(theta[2]) + np.cos(theta[0])*np.cos(theta[2]),      np.sin(theta[1])*np.sin(theta[2])*np.cos(theta[0]) - np.sin(theta[0])*np.cos(theta[2])],
                  [-np.sin(theta[1]),                        np.sin(theta[0])*np.cos(theta[1]),                                                           np.cos(theta[0])*np.cos(theta[1])]])

    return R

def test_calibrate_dlt():
    """from https://github.com/BonJovi1/Camera-Calibration/blob/master/code.ipynb"""
    worldcoo = [ (0,0,0), (0,28,0), (56,0,0), (56,28,0), (56,84,0), (84,84,0), (112,56,0),
             (112,84,0), (84,56,0), (84,112,0), (0,28,28), (0,28,56), (0,56,28), (0,56,56), 
             (0,56,84), (0,56,112), (0,112,0), (0,112,28), (0,112,56), (0,112,84), (0,112,112) 
           ]

    imagecoo = [ (1549, 1599), (1547, 1763), (1797, 1625), (1793, 1807), (1785, 2156), (1918, 2196),
             (2069, 2051), (2061, 2233), (1928, 2015), (1915, 2366), (1413, 1781), (1280, 1807),
             (1415, 1958), (1283, 1981), (1139, 2013), (990, 2041), (1541, 2251), (1420, 2287),
             (1292, 2320), (1149, 2356), (1005, 2401)
           ]

    imagecoo = np.array(imagecoo)

    worldcoo = np.array(worldcoo)

    P = calibrate_dlt(imagecoo, worldcoo)

    P_target = [[-0.00106739, -0.00031384 , 0.00275453 ,-0.69612992] , [ 0.00060642, -0.00311488 , 0.00049928, -0.71790235],
                [ 0.00000053, -0.00000021 , 0.00000047 ,-0.00045042]]

    P_target = np.array(P_target)

    diff = sum(sum(np.diff(P_target - P)))
    assert diff < 1e-8

if __name__ == "__main__":
    P = test_calibrate_dlt()

