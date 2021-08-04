import os
import copy
import time
import numpy as np
from scipy.optimize import linprog
import sys
sys.path.insert(0, "../")
from segment_matching.segment_matching_utils import get_segments


def get_optimal_match_with_weights(M):
    m, n = M.shape
    M = M.reshape(-1)
    row_constraint = np.zeros((m, M.shape[0]))
    for i in range(m):
        row_constraint[i, i * n: (i + 1) * n] = 1
    col_constraint = np.zeros((n, M.shape[0]))
    for i in range(n):
        col_constraint[i, [j * n + i for j in range(m)]] = 1
    row_eq = np.ones(m)
    col_eq = np.ones(n)
    res = linprog(-M, A_ub=-row_constraint,
                  b_ub=-row_eq,
                  A_eq=col_constraint,
                  b_eq=col_eq, bounds=[(0, 1)])

    return res


def get_segment_match(organ_1, organ_2, landmark1, landmark2, istop_1, istop_2):
    """

    :param organ_1:
    :param organ_2:
    :param landmark1: landmark matched already for organ1
    :param landmark2: landmark matched already for organ2, landmark1.shape == landmark2.shape
    :return:
    """

    match = []

    segments_1 = get_segments(organ_1, istop_1)
    segments_2 = get_segments(organ_2, istop_2)

    segment_index_1 = np.array(list(segments_1.keys()))
    segment_index_2 = np.array(list(segments_2.keys()))

    match_count = np.zeros((len(segment_index_1), len(segment_index_2)))
    for seg1 in segment_index_1:
        for seg2 in segment_index_2:
            i_1 = np.where(segment_index_1 == seg1)[0][0]
            i_2 = np.where(segment_index_2 == seg2)[0][0]
            match_count[i_1, i_2] = 1 - 1 * np.abs(segments_1[seg1]["point_number_proportion"]
                                    - segments_2[seg2]["point_number_proportion"]) \
                                    - 3 * abs(segments_1[seg1]["geodesic_distance"]
                                              - segments_2[seg2]["geodesic_distance"])

    for seg1, seg2 in zip(landmark1[:, 3], landmark2[:, 3]):
        if seg1 in segment_index_1 and seg2 in segment_index_2:
            i_1 = np.where(segment_index_1 == seg1)[0][0]
            i_2 = np.where(segment_index_2 == seg2)[0][0]
            match_count[i_1, i_2] += 1

    # linprog_res = get_optimal_match_with_weights(match_count)
    if len(segment_index_1) <= len(segment_index_2):
        linprog_res = get_optimal_match_with_weights(match_count)
        if linprog_res.success:
            x = linprog_res.x.reshape(match_count.shape)
            segment_match = []
            not_visted = segment_index_2
            for i in range(x.shape[0]):
                segment_match.append([segment_index_1[i],
                                      segment_index_2[np.argmax(x[i, :])]])
            segment_match = np.vstack(segment_match)
            pcd1, pcd2, skel_lm1, skel_lm2 = [], [], [], []
            for [seg_i1, seg_i2] in segment_match:
                pcd1.append(segments_1[seg_i1]['pcd'])
                pcd2.append(segments_2[seg_i2]['pcd'])
                skel_1 = segments_1[seg_i1]['skeleton']
                skel_2 = segments_2[seg_i2]['skeleton']
                len_keep = min(skel_1.shape[0], skel_2.shape[0])
                skel_1, skel_2 = skel_1[:len_keep], skel_2[:len_keep]
                skel_lm1.append(np.vstack([skel_1[0:skel_1.shape[0]-1:5], skel_1[-1]]))
                skel_lm2.append(np.vstack([skel_2[0:skel_2.shape[0]-1:5], skel_2[-1]]))

            for seg_i2 in segment_index_2:
                if seg_i2 not in segment_match[:, 1]:
                    pcd2.append(segments_2[seg_i2]['pcd'])

        else:
            pcd1, pcd2, skel_lm1, skel_lm2 = [organ_1["pcd"]], [organ_2["pcd"]], [landmark1[:, :3]], [landmark2[:, :3]]
        return pcd1, pcd2, skel_lm1, skel_lm2

    else:
        linprog_res = get_optimal_match_with_weights(match_count.T)
        if linprog_res.success:
            x = linprog_res.x.reshape((match_count.T).shape)
            segment_match = []
            for i in range(x.shape[0]):
                segment_match.append((segment_index_1[np.argmax(x[i, :])],
                                      segment_index_2[i]))
            pcd1, pcd2, skel_lm1, skel_lm2 = [], [], [], []
            segment_match = np.vstack(segment_match)

            for [seg_i1, seg_i2] in segment_match:
                pcd1.append(segments_1[seg_i1]['pcd'])
                pcd2.append(segments_2[seg_i2]['pcd'])
                skel_1 = segments_1[seg_i1]['skeleton']
                skel_2 = segments_2[seg_i2]['skeleton']
                len_keep = min(skel_1.shape[0], skel_2.shape[0])
                skel_1, skel_2 = skel_1[:len_keep], skel_2[:len_keep]
                skel_lm1.append(np.vstack([skel_1[0:skel_1.shape[0] - 1:5], skel_1[-1]]))
                skel_lm2.append(np.vstack([skel_2[0:skel_2.shape[0] - 1:5], skel_2[-1]]))

            for seg_i1 in segment_index_1:
                if seg_i1 not in segment_match[:, 0]:
                    pcd1.append(segments_1[seg_i1]['pcd'])

            else:
                pcd1, pcd2, skel_lm1, skel_lm2 = [organ_1["pcd"]], [organ_2["pcd"]], [landmark1[:, :3]], [
                    landmark2[:, :3]]
        else:
            pcd1, pcd2, skel_lm1, skel_lm2 = [organ_1["pcd"]], [organ_2["pcd"]], [landmark1[:, :3]], [landmark2[:, :3]]
        return pcd1, pcd2, skel_lm1, skel_lm2