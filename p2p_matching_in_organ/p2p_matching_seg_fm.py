import copy
import os
import sys
import time
import json
import numpy as np
sys.path.insert(0, "../")
import open3d
from point_cloud_clean import clean_point_cloud
from p2p_matching_in_organ.landmark_utils import get_skeleton
from utils import point_cloud_to_mesh, plot_skeleton_matching
from organ_matching.organ_matching_lr import match_organ_two_days
from skeleton_matching.skeleton_match import get_skeleton_landmarks_pairs
from segment_matching.segment_matching_ml import get_segment_match
from p2p_matching_in_organ.p2p_matching_evaluation import evaluate_p2p_matching
from p2p_matching_in_organ.p2p_matching_fm import get_p2p_mapping_organ
from p2p_matching_in_organ.p2p_matching_visualization import visualize_p2p_matching


def get_p2p_mapping_organ_collection(organ_collection_1,
                                     organ_collection_2,
                                     organ_matching,
                                     day1,
                                     day2,
                                     dataset,
                                     if_match_skeleton=True,
                                     plot_mesh=False,
                                     plot_pcd=False,
                                     plot_skeleton=False,
                                     show_all=True,
                                     apply_double_direction_refinement=True,
                                     verbose=True,
                                     options=None):

    """

    :param apply_double_direction_refinement:
    :param if_match_skeleton:
    :param organ_collection_1: List of organ dictionary for the plant one
                                organ_dict : {
                                    'pcd': point cloud of the organ
                                    'stem_connect_point': the connect point on main stem if organ is a branch,
                                    'skeleton_pcd': numpy array of shape [,4], first three column are
                                                    xyz for the skeleton of the organ the fourth column is the
                                                    index of segment
                                    'segment_index': the segment indexes included in the organ
                                    'skeleton_connection': the connect relationship between the segments in the organ

                                }

    :param organ_collection_2: List of organ dictionary for the plant two
    :param organ_matching: List, organ_matching[i] == j  ==> the i-th organ in plant one and j-th organ in plant two are
                                matched-up
    :param day1: str, first day
    :param day2: str, second day
    :param dataset: str, the name of the data set

    Following three bool values are parameters controlling the visualization for debug purpose
    :param plot_mesh: if plat the meshes
    :param plot_pcd: if plot the pcd
    :param plot_skeleton: if plot the skeleton
    :param show_all: if show all segments in the visualization, no matter they are matched or not
    :param verbose:
    :param options: the other hyper parameters
    :return:
    """
    options_path = "../hyper_parameters/lyon2.json"
    if not options:
        with open(options_path, "r") as json_file:
            options = json.load(json_file)

    mesh_collection_1 = []  # store the produced matched mesh
    mesh_collection_2 = []
    no_matched_pcd_collection_1 = []  # store the segments without matching
    no_matched_pcd_collection_2 = []

    p2p_mappings = [] # store the p2p mappings for each pair of segments
    scores = []
    score_weights = []
    visited = []
    for index, org1 in enumerate(organ_collection_1):

        if organ_matching[index] < 0:
            continue

        if organ_matching[index] in visited:
            org1['pcd'] = org1['pcd'].voxel_down_sample(0.8)
            no_matched_pcd_collection_1.append(org1['pcd'])
            continue

        visited.append(organ_matching[index])
        org2 = organ_collection_2[int(organ_matching[index])]

        xyz_1, connect_1, xyz_noted_1 = get_skeleton(org1, downsample_rate=0.15)  # Load the skeleton from the organ
        xyz_2, connect_2, xyz_noted_2 = get_skeleton(org2, downsample_rate=0.15)

        if plot_skeleton:
            plot_skeleton_matching(xyz_1, xyz_2, connect_1, connect_2)
        if if_match_skeleton:
            skel_landmarks1, skel_landmarks2 = get_skeleton_landmarks_pairs(xyz_noted_1, xyz_noted_2, connect_1, connect_2,
                                                                            visualize=plot_skeleton)
            if verbose:
                print("Used: ", skel_landmarks1.shape[0], " landmarks")
        else:
            skel_landmarks1 = None
            skel_landmarks2 = None
        # skel_landmarks2[[-2, -1], :] = skel_landmarks2[[-1, -2], :]

        istop_1 = index == len(organ_collection_1) - 1 and dataset not in ["tomato", "maize"]
        istop_2 = int(organ_matching[index]) == len(organ_collection_2) - 1 and dataset not in [ "tomato", "maize"]
        isstem_1 = index == 0 and dataset not in [ "tomato", "maize"]
        isstem_2 = int(organ_matching[index]) == 0 and dataset not in [ "tomato", "maize"]
        pcd1, pcd2, skel_lm1, skel_lm2, semantic_1, semantic_2 = \
            get_segment_match(org1, org2, skel_landmarks1, skel_landmarks2, istop_1,
                              istop_2, isstem_1, isstem_2, options)
        for i in range(len(skel_lm1)):
            p1, p2, lm1, lm2 = pcd1[i], pcd2[i], skel_lm1[i], skel_lm2[i]
            process_params = {
                'n_ev': (75, 75),  # Number of eigenvalues on source and Target
                'subsample_step': 2,  # In order not to use too many descriptors
                'descr_type': 'MIX',  # WKS or HKS
            }
            mesh1, mesh2, T21, T12 = get_p2p_mapping_organ(p1, p2, lm1, lm2,
                                                           verbose=verbose,
                                                           plot_pcd=plot_pcd,
                                                           plot_mesh=plot_mesh,
                                                           apply_double_direction_refinement=
                                                           apply_double_direction_refinement,
                                                           process_params=process_params)
            score = evaluate_p2p_matching(mesh1, mesh2, lm1, lm2, T21, T12,
                                          sample_proportion=0.5)
            print("score: ", score)
            print(64 * "=")
            scores.append(score)
            score_weights.append(np.asarray(mesh1.vertices).shape[0])
            mesh_collection_1.append(mesh1)
            mesh_collection_2.append(mesh2)
            p2p_mappings.append(copy.deepcopy(T21))

        if len(pcd1) > len(skel_lm1):
            for i in range(len(skel_lm1), len(pcd1)):
                pcd1[i] = pcd1[i].voxel_down_sample(0.8)
                no_matched_pcd_collection_1.append(pcd1[i])
        if len(pcd2) > len(skel_lm2):
            for i in range(len(skel_lm2), len(pcd2)):
                pcd2[i] = pcd2[i].voxel_down_sample(0.8)
                no_matched_pcd_collection_2.append(pcd2[i])
    for i in range(len(organ_collection_2)):
        if i not in organ_matching and i > 2:
            organ_collection_2[i]['pcd'] = organ_collection_2[i]['pcd'].voxel_down_sample(0.8)
            no_matched_pcd_collection_2.append(organ_collection_2[i]['pcd'])

    save_path_format = options["p2p_save_path_segment"]
    if not os.path.exists(save_path_format.format(dataset, day1, day2)):
        os.makedirs(save_path_format.format(dataset, day1, day2))

    i = 0
    for m1, m2 in zip(mesh_collection_1, mesh_collection_2):
        np.savetxt(save_path_format.format(dataset, day1, day2) + "{}_{}.csv".format(day1, i), np.asarray(m1.vertices))
        np.savetxt(save_path_format.format(dataset, day1, day2) + "{}_{}.csv".format(day2, i), np.asarray(m2.vertices))
        np.savetxt(save_path_format.format(dataset, day1, day2) + "T21_{}.csv".format(i), p2p_mappings[i])
        i += 1

    i = 0
    for p in no_matched_pcd_collection_1:
        np.savetxt(save_path_format.format(dataset, day1, day2) + "{}_no_matched_{}.csv".format(day1, i),
                   np.asarray(p.points))
        i += 1

    i = 0
    for p in no_matched_pcd_collection_2:
        np.savetxt(save_path_format.format(dataset, day1, day2) + "{}_no_matched_{}.csv".format(day2, i),
                   np.asarray(p.points))
        i += 1

    # visualize_p2p_matching(day1, day2, dataset, save_path_format=save_path_format, show_all=True)
    print(score_weights)
    scores = np.vstack(scores)
    np.savetxt(save_path_format.format(dataset, day1, day2) + "score.csv", scores)
    return np.average(scores, axis=0, weights=score_weights)


def form_tomato_org(day1, options=None):
    dataset = "tomato"

    # Load the options if it's not given
    if options is None:
        options_path = "../hyper_parameters/tomato.json"
        with open(options_path, "r") as json_file:
            options = json.load(json_file)

    # Load the parameters from the option
    pc_path = options['pc_path']
    skel_noted_path = options['skel_noted_path']  # the xyz of skeleton nodes with the segment index labelled
    segment_connect_path = options['segment_connect_path']  # the connection relationship among the segments
    stem_node_path = options['stem_node_path']  #
    mesh_radius_factor = options['mesh_radius_factor']
    pcd_clean_option = options['pcd_clean_option']

    pcd1 = open3d.io.read_point_cloud(pc_path.format(dataset, day1))

    pcd1 = clean_point_cloud(pcd1, option=pcd_clean_option)[0]
    mesh1 = point_cloud_to_mesh(pcd1, radius_factor=mesh_radius_factor)

    if os.path.exists(skel_noted_path.format(dataset, day1, day1)) and \
            os.path.exists(segment_connect_path.format(dataset, day1, day1)):

        # If the skeleton data has been calculated, we load the data
        xyz_labeled_1 = np.loadtxt(skel_noted_path.format(dataset, day1, day1))
        branch_connect_1 = np.loadtxt(segment_connect_path.format(dataset, day1, day1))

        stem_connect_point = xyz_labeled_1[np.argmin(xyz_labeled_1[:, 2]), :3]

        segment_index = np.unique(xyz_labeled_1[:, 3])
        branch_connect_1 = np.vstack([branch_connect_1, [xyz_labeled_1[np.argmin(xyz_labeled_1[:, 2]), 3],
                                                         np.max(segment_index) + 1]])

    else:
        return None

    return {
        "pcd": pcd1,
        "stem_connect_point": stem_connect_point,
        "skeleton_pcd": xyz_labeled_1,
        "skeleton_connection": branch_connect_1,
        "segment_index": segment_index,
    }


if __name__ == "__main__":
    # day1 = "05-12_AM"
    # day2 = "05-13_AM"
    day1 = "03-22_PM"
    day2 = "03-23_PM"
    dataset = "lyon2"
    t_start = time.time()

    if dataset == "lyon2":
        matches_index, org_collection_1, org_collection_2 = \
            match_organ_two_days(day1, day2, dataset, visualize=True, verbose=True)
        t_end = time.time()
        print("Get the organs matched, used ", t_end - t_start, " s")

        t_start = time.time()
        get_p2p_mapping_organ_collection(org_collection_1[0:1] + org_collection_1[3:],
                                         org_collection_2,
                                         np.append(matches_index[0:1], matches_index[3:]),
                                         day1=day1,
                                         day2=day2,
                                         dataset=dataset,
                                         if_match_skeleton=False,
                                         plot_mesh=False,
                                         plot_pcd=False,
                                         verbose=False,
                                         show_all=False)

        # get_p2p_mapping_organ_collection(org_collection_1[0:1],
        #                                  org_collection_2,
        #                                  matches_index[0:1],
        #                                  day1=day1,
        #                                  day2=day2,
        #                                  dataset=dataset,
        #                                  if_match_skeleton=False,
        #                                  plot_mesh=True,
        #                                  plot_pcd=True,
        #                                  verbose=False,
        #                                  show_all=False)

        t_end = time.time()
        print("Total time used: ", t_end - t_start)

    if dataset == "tomato":
        # 'pcd', 'stem_connect_point', 'skeleton_pcd', 'skeleton_connection', 'segment_index'
        org_collection_1 = [form_tomato_org(day1)]
        org_collection_2 = [form_tomato_org(day2)]
        matches_index = [0]
        get_p2p_mapping_organ_collection(org_collection_1,
                                         org_collection_2,
                                         matches_index,
                                         day1=day1,
                                         day2=day2,
                                         dataset=dataset,
                                         plot_mesh=False,
                                         plot_pcd=False,
                                         verbose=False,
                                         show_all=False)
