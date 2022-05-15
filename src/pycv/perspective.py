#%%
import numpy as np
import math
import scipy.linalg as la
import cv2

def solve_PnP(world, native, K, distCoeffs = np.array([[0, 0, 0, 0]]).astype("float32")):
    if world.shape[0] > 4:
        world = world[:4,:]
        native = native[:4,:]
    retval, rvec, tvec = cv2.solveP3P(world.astype("float32"), native.astype("float32"), K.astype("float32"),distCoeffs, flags=cv2.SOLVEPNP_P3P)
    # as up to 4 solutions
    rvec = rvec[1]
    tvec = tvec[1]
    print(rvec)
    print(tvec)
    R, _ = cv2.Rodrigues(rvec)
    T = tvec
    camera_extrinsic_matrix = np.hstack((R,T))
    camera_extrinsic_matrix_hat = np.vstack((camera_extrinsic_matrix,[0,0,0,1]))
    camera_pose_matrix = la.inv(camera_extrinsic_matrix_hat)[:3,:]
    return camera_pose_matrix

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

def reprojection_error(P, image_points, world_points):
    projs_hat = [np.matmul(P, np.array([p[0], p[1], p[2], 1])) for p in world_points ]
    projs = [np.array([p[0], p[1]]) / p[2] for p in projs_hat]

    # reprojection error
    repr_err = [ math.sqrt((projs[i][0] - image_points[i,0])**2 + (projs[i][1] - image_points[i,1])**2) for  i in range(len(projs))]
    projs = np.array(projs)
    return np.array(repr_err)

def horn_affine_transformation(P,Q):
    P = P.T
    Q = Q.T
    if P.shape != Q.shape:
        print("Matrices P and Q must be of the same dimensionality")
        raise Exception("matrices must be same shape")
    centroids_P = np.mean(P, axis=1)
    centroids_Q = np.mean(Q, axis=1)
    A = P - np.outer(centroids_P, np.ones(P.shape[1]))
    B = Q - np.outer(centroids_Q, np.ones(Q.shape[1]))
    C = np.dot(A, B.transpose())
    U, S, V = np.linalg.svd(C)
    R = np.dot(V.transpose(), U.transpose())
    L = np.eye(3)
    if(np.linalg.det(R) < 0):
        L[2][2] *= -1
    R = np.dot(V.transpose(), np.dot(L, U.transpose()))
    t = np.dot(-R, centroids_P) + centroids_Q
    T = np.array([t]).T
    return np.hstack((R,T))

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

