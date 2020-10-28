import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from math import degrees

from time import time

# '''
# Args: 
#     points: (Nx3) array of points
# Returns:
#      
# '''
# def fit_plane(points):
#     m = np.mean(points, axis=0)
#     U, S, V_T = np.linalg.svd((points - m))
#
#     print(V_T.shape)
#     
#     n = V_T[-1]
#
#     d = -n.dot(m)
#     a, b, c = n.flatten()
#
#     return a, b, c, d

# points = np.load("cloud.npy")

def fit_plane(points):
    # fig = plt.figure()
    # ax = Axes3D(fig)

    # X = g_points[:,0]
    # Y = g_points[:,1]
    # Z = g_points[:,2]

    from sklearn import linear_model

    ransac = linear_model.RANSACRegressor(residual_threshold=0.1)
    ransac.fit(points[:,0:2], points[:,2])

    # Data for three-dimensional scattered points

    # # Fit the plane
    # t0 = time()
    # a, b, c, d = fit_plane(g_points)
    # t1 = time()
    # print(t1 - t0)

    # Display plane
    x = np.arange(np.min(points[:,0]), np.max(points[:,0]), 0.25)
    y = np.arange(np.min(points[:,1]), np.max(points[:,1]), 0.25)
    xs, ys = np.meshgrid(x, y)

    # zs = -1./c * ((a*xs) + (b*ys) + d)
    print(xs.shape, ys.shape)
    zs = ransac.predict(np.concatenate((xs.reshape(-1, 1), ys.reshape(-1, 1)), axis=1))
    zs = zs.reshape(xs.shape)

    m = ransac.inlier_mask_

    X = points[m,0]
    Y = points[m,1]
    Z = points[m,2]

    max_range = np.array([X.max()-X.min(), Y.max()-Y.min(), Z.max()-Z.min()]).max() / 2.0

    mid_x = (X.max()+X.min()) * 0.5
    mid_y = (Y.max()+Y.min()) * 0.5
    mid_z = (Z.max()+Z.min()) * 0.5
    # ax.set_xlim(mid_x - max_range, mid_x + max_range)
    # ax.set_ylim(mid_y - max_range, mid_y + max_range)
    # ax.set_zlim(mid_z - max_range, mid_z + max_range)

    m2 = np.bitwise_not(ransac.inlier_mask_)

    X2 = points[m2,0]
    Y2 = points[m2,1]
    Z2 = points[m2,2]

    # ax.plot_wireframe(xs, ys, zs, color=(0., 0., 0., 1.))

    # ax.scatter(X2, Y2, Z2, c=Z2, cmap='summer')
    # ax.scatter(X, Y, Z, c=Z, cmap='copper')

    # print("coef", ransac.estimator_.coef_)
    # print("intercept", ransac.estimator_.intercept_)
    # print(ransac.inlier_mask_.shape)
    # print(points.shape)

    # print(ransac.estimator_.coef_)
    # print(ransac.estimator_.intercept_)

    a, b = ransac.estimator_.coef_

    n = (-a, -b, 1)
    n /= np.linalg.norm(n) 

    roll = np.arctan2(n[1], n[2])
    pitch = np.arctan2(n[0], n[2])

    # print(n)
    print(degrees(roll), degrees(pitch))

    # plt.show()

    return float(roll), float(pitch)



