from functional_map_.signatures.hks import get_hks_minmax
from functional_map_.signatures.wks import get_wks_minmax
from functional_map_.signatures.fpfh import get_fpfh
# import additional modules used in this tutorial
import numpy as np


def get_descriptor_value(basis_functions_1, natural_frequencies_1, xyz, maxBasis, descriptor_weights, options={}):
    """

    :param basis_functions_1: numpy array (n * n), basis function value
    :param natural_frequencies_1: numpy array (n), natural frequencies
    :param xyz: numpy array (n * 3), coordinates of vertices
    :param maxBasis: int, the number of basis function values we take for each vertex
    :param descriptor_weights: the weights of HKS, WKS and XYZ signatures
    :param options: the optional options for the parameter
    :return: numpy array (n * ?)
    """
    if "hks_ts" not in options:
        hks_ts = np.array([ 0.001, 0.01, 0.05, 0.1, 1, 5, 10])
    else:
        hks_ts = options['hks_ts']
    hks_value = get_hks_minmax(basis_functions_1, natural_frequencies_1, maxBasis, hks_ts)
    if "wks_energy_num" not in options:
        wks_energy_num = 10
    else:
        wks_energy_num = options["wks_energy_num"]
    wks_value = get_wks_minmax(basis_functions_1, natural_frequencies_1, maxBasis, wks_energy_num)

    if "voxel_size" in options:
        voxel_size = options['voxel_size']
    else:
        voxel_size = (np.max(xyz[:, 2]) - np.min(xyz[:, 2])) / 100
    fpfh_value = get_fpfh(xyz, voxel_size)
    return np.hstack([ descriptor_weights[0] * hks_value,
                       descriptor_weights[1] * wks_value,
                       descriptor_weights[2] * fpfh_value,
                       descriptor_weights[3] * xyz])


def get_functional_representation(target_function, basis_function, area, dim):
    """
    :param target_function: (n * n) the descriptor function to be projected
    :param basis_function: (n * n) Laplacian basis functions
    :param area: (n * n) the fem area for each pair of vertices
    :param dim: int
    :return:
    """
    A = basis_function[:, :dim]
    b = target_function
    residual = [0,0]
    x, residual, rank, s = np.linalg.lstsq(A, b, rcond=None)

    # x = np.linalg.inv(A.T.dot(A)).dot(A.T).dot(b)
    residual[0] = np.mean(np.abs(A.dot(x) - target_function))
    return x, residual[0]


def get_transition_matrix(value_rep_1, value_rep_2, natural_frequencies_1, natural_frequencies_2, alpha, dim):

    C_init = np.random.rand(dim, dim)

    P1 = value_rep_1
    P2 = value_rep_2
    L1 = natural_frequencies_1[:dim]
    L2 = natural_frequencies_2[:dim]

    C = C_init
    A_fixed = P1.dot(P1.T)
    B = P1.dot(P2.T)

    # if np.linalg.cond(A_fixed) < 1 / sys.float_info.epsilon:
    #     C = (np.linalg.inv(A_fixed).dot(B)).T
    # else:
    #     C = (np.linalg.pinv(A_fixed).dot(B)).T

    for i in range(dim):
        A = np.diag( alpha * (L1 -L2[i])**2 ) + A_fixed
        C[i, :] = np.linalg.inv(A).dot(B[:, i])
    residual = np.linalg.norm(C.dot(P1) - P2) + alpha * np.linalg.norm(C.dot(np.diag(L1)) - np.diag(L2).dot(C))
    print("::fitting residual: ", residual/(dim * dim))
    return C