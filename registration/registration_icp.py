import copy

import numpy as np
import open3d
from sklearn.neighbors import KDTree
from scipy.spatial.transform import Rotation

def get_best_fit_transform(A, B):
    '''
    Calculates the least-squares best-fit transform between corresponding 3D points A->B
    Input:
      A: Nx3 numpy array of corresponding 3D points
      B: Nx3 numpy array of corresponding 3D points
    Returns:
      T: 4x4 homogeneous transformation matrix
      R: 3x3 rotation matrix
      t: 3x1 column vector
    '''

    assert len(A) == len(B)

    # translate points to their centroids
    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)
    AA = A - centroid_A
    BB = B - centroid_B

    # rotation matrix
    H = np.dot(AA.T, BB)
    U, S, Vt = np.linalg.svd(H)
    R = np.dot(Vt.T, U.T)

    # special reflection case
    if np.linalg.det(R) < 0:
       Vt[2,:] *= -1
       R = np.dot(Vt.T, U.T)

    # translation
    t = centroid_B.T - np.dot(R, centroid_A.T)

    # homogeneous transformation
    T = np.identity(4)
    T[0:3, 0:3] = R.T
    T[[3], 0:3] = t.T

    return T, R, t


def get_NN_in_target(xyz1, xyz2):

    tree = KDTree(xyz2)

    distance, indices_NN = tree.query(xyz1, k=1, return_distance=True)

    indices_NN = indices_NN.flatten()

    return distance, indices_NN


def my_icp(xyz1, xyz2):
    p1 = open3d.geometry.PointCloud()
    p2 = open3d.geometry.PointCloud()
    p2.points = open3d.utility.Vector3dVector(xyz2)
    xyz1 = copy.deepcopy(xyz1).T
    xyz2 = copy.deepcopy(xyz2).T

    R_init = np.array([[1.0, 0.0, 0.0],
                       [0.0, 1.0, 0.0],
                       [0.0, 0.0, 1.0]])
    t_init = (np.mean(xyz2, axis=1) - np.mean(xyz1, axis=1)).reshape(-1, 1)
    T = np.identity(4)
    T[[3], 0:3] = t_init.T

    R_ = R_init
    t_ = t_init
    R = R_init
    t = np.zeros((3, 1))
    Cost = []
    for i in range(500):
        t = t.reshape(-1, 1)
        t_ = t_.reshape(-1, 1)
        xyz1 = np.dot(R_, xyz1) + t_
        R = np.dot(R_, R)
        t = np.dot(R_, t) + t_
        distance, indices = get_NN_in_target(xyz1.T, xyz2.T)
        Cost.append(np.mean(distance))
        if i > 1 and (Cost[-1] > Cost[-2] + 10**-4 or np.linalg.norm(Cost[-1] - Cost[-2]) < 0.01):
            break

        T, R_, t_ = get_best_fit_transform(xyz1.T, xyz2.T[indices])

        p1.points = open3d.utility.Vector3dVector(xyz1.T)
    # open3d.visualization.draw_geometries([p1, p2])

    T = np.identity(4)
    T[0:3, 0:3] = R
    T[[3], 0:3] = t.T
    return T, R, t


if __name__ == "__main__":
    P1 = open3d.io.read_point_cloud("icp_samples/P1_2.ply")
    P2 = open3d.io.read_point_cloud("icp_samples/P2_2.ply")
    open3d.visualization.draw_geometries([P1, P2])

    T, R, t = my_icp(np.asarray(P1.points), np.asarray(P2.points))
    print("Finished testing")

