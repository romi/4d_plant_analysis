import copy
import time
import numpy as np
from sklearn.neighbors import KDTree
from p2p_matching_in_organ.landmark_utils import get_mesh_landmarks


def evaluate_p2p_matching(mesh1, mesh2, skel_landmarks1, skel_landmarks2, T21, T12 = None,
                          voxel_size=1.0,
                          sample_proportion=0.1,
                          verbose=True):
    """

    :param sample_proportion:
    :param voxel_size:
    :param mesh1: the target mesh
    :param mesh2: the source mesh  len(mesh.points) == len(p2p)
    :param skel_landmarks1: landmarks of mesh1
    :param skel_landmarks2: landmarks of mesh2
    :param p2p: the ith point of mesh2  <---> p2p[i] point of mesh1
    :return:
    """
    if skel_landmarks2.shape[0] > 0 and skel_landmarks1.shape[0] > 0:
        lm1 = get_mesh_landmarks(mesh1, skel_landmarks1)
        lm2 = get_mesh_landmarks(mesh2, skel_landmarks2)
    else:
        lm1, lm2 = None, None

    vertices_1 = np.asarray(mesh1.vertices)
    vertices_2 = np.asarray(mesh2.vertices)

    threshold = 2 * voxel_size

    # check the projection of landmarks
    if lm1 is not None and lm2 is not None:
        distance_lm_corres = np.linalg.norm(vertices_1[T21[lm2]] - vertices_1[lm1], axis=1)
        lm_precision = np.sum(distance_lm_corres < 2 * threshold) / distance_lm_corres.shape[0]
    else:
        lm_precision = None

    # Evaluate the continuity of mapping
    sample_step = max(int(1 / sample_proportion), 1)
    tree = KDTree(vertices_2)
    NN_mesh2 = tree.query(np.asarray(vertices_2), k=2, return_distance=False)

    pass_cnt = 0
    total_cnt = 0
    for i in range(0, NN_mesh2.shape[0], sample_step):
        n1 = NN_mesh2[i, 0]
        n2 = NN_mesh2[i, 1]
        m1, m2 = T21[n1], T21[n2]
        total_cnt += 1
        if m1 == m2:
            continue
        if np.linalg.norm(vertices_1[m1] - vertices_1[m2]) < threshold:
            pass_cnt += 1
    continuity_mesure = pass_cnt / total_cnt

    if T12 is not None:
        total_cnt = 0
        pass_cnt = 0
        for i in range(T21.shape[0]):
            total_cnt += 1
            v_reflect = vertices_2[T12[T21[i]]]
            v_original = vertices_2[i, :]
            if np.linalg.norm(v_reflect - v_original) < threshold:
                pass_cnt += 1
        cycle_mesure = pass_cnt / total_cnt
    else:
        cycle_mesure = None

    res = []
    if continuity_mesure is not None:
        if verbose:
            print("evaluate: continuity -- ", continuity_mesure, end=", " )
        res.append(continuity_mesure)

    if lm_precision is not None:
        if verbose:
            print(" landmark precision -- ", lm_precision, end=", ")
        res.append(lm_precision)

    if cycle_mesure is not None:
        if verbose:
            print("cycle mesure -- ", cycle_mesure)
        res.append(cycle_mesure)

    return res