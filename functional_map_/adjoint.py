import numpy as np
import scipy as sp
from utils import get_transition_matrix
# from icp import icp_refine
from sklearn.neighbors import KDTree


def C_to_p2p(C, basis_function1, basis_function2):
    dim = C.shape[0]
    tree = KDTree(basis_function1[:, :dim] @ C.T)
    matches = tree.query(basis_function2[:, :dim], k=1, return_distance=False).flatten()
    return matches


def p2p_to_C(matches, basis_function1, basis_function2):
    return scipy.linalg.lstsq(basis_function2, basis_function1[matches, :])[0]

def icp_iteration(eigvects1, eigvects2, FM):
    """
    Performs an iteration of ICP.
    Conversion from a functional map to a pointwise map is done by comparing
    embeddings of dirac functions on the second mesh Phi_2.T with embeddings
    of dirac functions of the first mesh (transported with the functional map) C@Phi_1.T

    Parameters
    -------------------------
    FM        : (k2,k1) functional map in reduced basis
    eigvects1 : (n1,k1') first k' eigenvectors of the first basis  (k1'>k1).
    eigvects2 : (n2,k2') first k' eigenvectors of the second basis (k2'>k2)

    Output
    --------------------------
    FM_refined : (k2,k1) An orthogonal functional map after one step of refinement
    """
    k2, k1 = FM.shape
    p2p = C_to_p2p(FM, eigvects1, eigvects2)
    C_icp = p2p_to_C(p2p, eigvects1[:, :k1], eigvects2[:, :k2])
    U, _, VT = scipy.linalg.svd(C_icp)
    return U @ np.eye(k2, k1) @ VT


def icp_refine(eigvects1, eigvects2, FM, nit=10, tol=1e-10, verbose=False):
    """
    Refine a functional map using the standard ICP algorithm.
    Parameters
    --------------------------
    eigvects1 : (n1,k1') first k' eigenvectors of the first basis  (k1'>k1).
    eigvects2 : (n2,k2') first k' eigenvectors of the second basis (k2'>k2)
    FM        : (k2,k1) functional map in reduced basis
    nit       : int - Number of iterations to perform. If not specified, uses the tol parameter
    tol       : float - Maximum change in a functional map to stop refinement
                (only used if nit is not specified)
    Output
    ---------------------------
    FM_icp    : ICP-refined functional map
    """
    current_FM = FM.copy()
    iteration = 1

    if nit is not None and nit > 0:
        myrange = range(nit)
    else:
        myrange = range(100)

    for i in myrange:
        FM_icp = icp_iteration(eigvects1, eigvects2, current_FM)

        if nit is None or nit == 0:
            if np.max(np.abs(current_FM - FM_icp)) <= tol:
                break

        current_FM = FM_icp.copy()
    return current_FM


def get_adjoint_matrix(value_rep_1, value_rep_2,
                       natural_frequencies_1, natural_frequencies_2,
                       basis_functions_1, basis_functions_2,
                       alpha, dim, icp=False):

    C = get_transition_matrix(value_rep_2, value_rep_1, natural_frequencies_2, natural_frequencies_1, alpha, dim)
    if icp:
        C = icp_refine(basis_functions_2[:, :dim], basis_functions_1[:, :dim], C, nit=100, tol=10 ** -3)

    return C.T
