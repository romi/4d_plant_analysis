import copy
import os
import sys
import time
import json
import numpy as np
import pandas as pd
from sklearn.neighbors import KDTree

sys.path.insert(0, "../")
import open3d

from skeletonization.point_cloud_segmentation import point_cloud_segmentation
from point_cloud_clean import clean_point_cloud
from p2p_matching_in_organ.landmark_utils import get_skeleton
from utils import point_cloud_to_mesh, plot_skeleton_matching
from organ_matching.organ_matching_lr import match_organ_two_days
from skeleton_matching.skeleton_match import get_skeleton_landmarks_pairs
from segment_matching.segment_matching_ml import get_segment_match
from p2p_matching_in_organ.p2p_matching_evaluation import evaluate_p2p_matching
from p2p_matching_in_organ.p2p_matching_fm import double_direction_refinement
from p2p_matching_in_organ.landmark_utils import get_mesh_landmarks
from p2p_matching_in_organ.p2p_matching_visualization import visualize_p2p_matching
from registration.registration_icp import my_icp


def get_triangle_angles(a, b, c):
    assert a + b >= c and a + c >= b and b + c >= a
    cos_ = [
        (b ** 2 + c ** 2 - a ** 2) / (2 * c * b + 10 ** -5),
        (c ** 2 + a ** 2 - b ** 2) / (2 * a * c + 10 ** -5),
        (a ** 2 + b ** 2 - c ** 2) / (2 * a * b + 10 ** -5)
    ]
    return np.arccos(cos_)


def get_open3d_skel_line(lm, color=[0.2, 0.9, 0.1]):
    l_lm = open3d.geometry.LineSet()
    l_lm.points = open3d.utility.Vector3dVector(lm)
    l_lm.lines = open3d.utility.Vector2iVector([[i, i + 1] for i in range(lm.shape[0] - 1)])

    p_lm = []
    for xyz in lm:
        m_lm = open3d.geometry.TriangleMesh.create_sphere(0.5)
        m_lm.translate(xyz)
        m_lm.vertex_colors = open3d.utility.Vector3dVector(color * np.ones(np.asarray(m_lm.vertices).shape))
        p_lm.append(m_lm)
    return l_lm, p_lm


def get_p2p_mapping_organ_collection(organ_collection_1,
                                     organ_collection_2,
                                     organ_matching,
                                     day1,
                                     day2,
                                     dataset,
                                     if_match_skeleton=True,
                                     plot_mesh=False,
                                     plot_pcd=False,
                                     plot_res=False,
                                     plot_skeleton=False,
                                     show_all=True,
                                     verbose=True,
                                     apply_double_direction_refinement=True,
                                     options=None):
    """

    :param plot_res:
    :param apply_double_direction_refinement: bool: if apply the double direction after the registration
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
    options_path = "../hyper_parameters/{}.json".format(dataset)
    if not options:
        with open(options_path, "r") as json_file:
            options = json.load(json_file)

    mesh_collection_1 = []  # store the produced matched mesh
    mesh_collection_2 = []
    no_matched_pcd_collection_1 = []  # store the segments without matching
    no_matched_pcd_collection_2 = []

    semantic_collection_1 = []
    semantic_collection_2 = []
    T21s = []
    T12s = []  # store the p2p mappings for each pair of segments
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

        xyz_1, connect_1, xyz_noted_1 = get_skeleton(org1, downsample_rate=0.08)  # Load the skeleton from the organ
        xyz_2, connect_2, xyz_noted_2 = get_skeleton(org2, downsample_rate=0.08)

        if plot_skeleton:
            plot_skeleton_matching(xyz_1, xyz_2, connect_1, connect_2)
        if if_match_skeleton:
            skel_landmarks1, skel_landmarks2 = get_skeleton_landmarks_pairs(xyz_noted_1, xyz_noted_2, connect_1,
                                                                            connect_2,
                                                                            visualize=plot_skeleton)
            if verbose:
                print("Used: ", skel_landmarks1.shape[0], " landmarks")
        else:
            skel_landmarks1 = None
            skel_landmarks2 = None
        # skel_landmarks2[[-2, -1], :] = skel_landmarks2[[-1, -2], :]

        istop_1 = index == len(organ_collection_1) - 1 and dataset not in ["tomato", "maize"]
        istop_2 = int(organ_matching[index]) == len(organ_collection_2) - 1 and dataset not in ["tomato", "maize"]
        isstem_1 = index == 0 and dataset not in ["tomato", "maize"]
        isstem_2 = int(organ_matching[index]) == 0 and dataset not in ["tomato", "maize"]
        pcd1, pcd2, skel_lm1, skel_lm2, semantic_1, semantic_2 = \
            get_segment_match(org1, org2, skel_landmarks1, skel_landmarks2, istop_1,
                              istop_2, isstem_1, isstem_2, options, sample_method="end")
        for i in range(len(skel_lm1)):
            p1, p2, lm1, lm2 = pcd1[i], pcd2[i], skel_lm1[i], skel_lm2[i]

            l_lm1, p_lm1 = get_open3d_skel_line(lm1)
            l_lm2, p_lm2 = get_open3d_skel_line(lm2, [0.9, 0.2, 0.1])
            translat_xyz = [0, 20, 0]
            for p in p_lm2:
                p.translate(translat_xyz)
            # p2.translate(translat_xyz)
            l_lm2.translate(translat_xyz)
            connect = open3d.geometry.LineSet()
            n = np.asarray(l_lm1.points).shape[0]
            connect.points = open3d.utility.Vector3dVector(
                np.vstack([np.asarray(l_lm1.points), np.asarray(l_lm2.points)]))
            connect.lines = open3d.utility.Vector2iVector([[i, i+n] for i in range(n)])

            sm1 = semantic_1[i]
            sm2 = semantic_2[i]
            # open3d.visualization.draw_geometries([p1, p2])
            mesh1 = point_cloud_to_mesh(p1, radius_factor=options['mesh_radius_factor'])
            mesh2 = point_cloud_to_mesh(p2, radius_factor=options['mesh_radius_factor'])
            T21 = get_p2p_mapping_from_segments_using_icp(mesh1, mesh2, lm1, lm2,
                                                          options["pcd_clean_option"]["voxel_size"] / 2)
            T12 = get_p2p_mapping_from_segments_using_icp(mesh2, mesh1, lm2, lm1,
                                                          options["pcd_clean_option"]["voxel_size"] / 2)

            lm_in_mesh_1 = get_mesh_landmarks(mesh1, lm1)
            lm_in_mesh_2 = get_mesh_landmarks(mesh2, lm2)

            if apply_double_direction_refinement:
                T21, T12 = double_direction_refinement(mesh1, mesh2, T21, T12, lm_in_mesh_1, lm_in_mesh_2)

            score = evaluate_p2p_matching(mesh1, mesh2, lm1, lm2, T21, T12, verbose=False,
                                          sample_proportion=0.5)
            # print("score: ", score)
            # print(64 * "=")
            scores.append(score)
            score_weights.append(np.asarray(mesh1.vertices).shape[0])
            mesh_collection_1.append(mesh1)
            mesh_collection_2.append(mesh2)
            T21s.append(copy.deepcopy(T21))
            T12s.append(copy.deepcopy(T12))
            semantic_collection_1.append(sm1)
            semantic_collection_2.append(sm2)

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

    pd.DataFrame(np.array(semantic_collection_2)).to_csv(
        save_path_format.format(dataset, day1, day2) + "semantic_2.csv")
    pd.DataFrame(np.array(semantic_collection_1)).to_csv(
        save_path_format.format(dataset, day1, day2) + "semantic_1.csv")

    i = 0
    for m1, m2 in zip(mesh_collection_1, mesh_collection_2):
        np.savetxt(save_path_format.format(dataset, day1, day2) + "{}_{}.csv".format(day1, i), np.asarray(m1.vertices))
        np.savetxt(save_path_format.format(dataset, day1, day2) + "{}_{}.csv".format(day2, i), np.asarray(m2.vertices))
        np.savetxt(save_path_format.format(dataset, day1, day2) + "T21_{}.csv".format(i), T21s[i])
        np.savetxt(save_path_format.format(dataset, day1, day2) + "T12_{}.csv".format(i), T12s[i])
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

    if plot_res:
        visualize_p2p_matching(day1, day2, dataset, save_path_format=save_path_format, show_all=True)
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
        gb1, bc1, bo1, bsc1, mesh1, label_mesh_1 = point_cloud_segmentation(pcd1,
                                                                            day1,
                                                                            dataset,
                                                                            path_format="../data/{}/{}_stem/",
                                                                            verbose=False,
                                                                            visualize=False,
                                                                            options=options)
        xyz_labeled_1 = np.loadtxt(skel_noted_path.format(dataset, day1, day1))
        branch_connect_1 = np.loadtxt(segment_connect_path.format(dataset, day1, day1))
        xyz_stem_node_1 = np.loadtxt(stem_node_path.format(dataset, day1))
        stem_connect_point = xyz_labeled_1[np.argmin(xyz_labeled_1[:, 2]), :3]

        segment_index = np.unique(xyz_labeled_1[:, 3])
        branch_connect_1 = np.vstack([branch_connect_1, [xyz_labeled_1[np.argmin(xyz_labeled_1[:, 2]), 3],
                                                         np.max(segment_index) + 1]])

    return {
        "pcd": pcd1,
        "stem_connect_point": stem_connect_point,
        "skeleton_pcd": xyz_labeled_1,
        "skeleton_connection": branch_connect_1,
        "segment_index": segment_index,
    }


def form_maize_org(day1, options=None):
    """
    Generate the organ of maize: since the maize plant has a simpler structure, we can consider it as a single organ
    :param day1: str: the date of the organ, used to fetch the data
    :param options: dict: hyper-parameters used for data processing
    :return: dict: a dictionary to describe an organ, with:
                    {
                        "pcd": the point cloud of the organ,
                        "stem_connect_point": the connect point to the main stem or root,
                        "skeleton_pcd": float numpy array of shape [N, 4], the nodes of the organ's skeleton, first 3 indices are the
                                        xyz values of the nodes, the 4th one is the segment index of the node
                        "skeleton_connection": integer numpy array of shape [N, 2], the connection of the skeleton's node,
                        "segment_index": all indices of the segments in this organ
                    }
    """
    dataset = "maize"

    # Load the options if it's not given
    if options is None:
        options_path = "../hyper_parameters/maize.json"
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
        gb1, bc1, bo1, bsc1, mesh1, label_mesh_1 = point_cloud_segmentation(pcd1,
                                                                            day1,
                                                                            dataset,
                                                                            path_format="../data/{}/{}_stem/",
                                                                            verbose=False,
                                                                            visualize=False,
                                                                            options=options)
        xyz_labeled_1 = np.loadtxt(skel_noted_path.format(dataset, day1, day1))
        branch_connect_1 = np.loadtxt(segment_connect_path.format(dataset, day1, day1))
        xyz_stem_node_1 = np.loadtxt(stem_node_path.format(dataset, day1))
        stem_connect_point = xyz_labeled_1[np.argmin(xyz_labeled_1[:, 2]), :3]

        segment_index = np.unique(xyz_labeled_1[:, 3])
        branch_connect_1 = np.vstack([branch_connect_1, [xyz_labeled_1[np.argmin(xyz_labeled_1[:, 2]), 3],
                                                         np.max(segment_index) + 1]])

    return {
        "pcd": pcd1,
        "stem_connect_point": stem_connect_point,
        "skeleton_pcd": xyz_labeled_1,
        "skeleton_connection": branch_connect_1,
        "segment_index": segment_index,
    }


def get_p2p_mapping_from_segments_using_icp(mesh1, mesh2, lm1, lm2, voxel_size):
    xyz1 = np.asarray(copy.deepcopy(mesh1).vertices)
    xyz2 = np.asarray(copy.deepcopy(mesh2).vertices)

    if lm1.shape[0] > 1:
        tree1 = KDTree(lm1)
        distance_to_skel_1, NN_skel_1 = tree1.query(xyz1, k=2)

        tree2 = KDTree(lm2)
        distance_to_skel_2, NN_skel_2 = tree2.query(xyz2, k=2)

        color_setting = np.random.random((lm1.shape[0], 3))
        color_1 = color_setting[NN_skel_1[:, 0]]
        color_2 = color_setting[NN_skel_2[:, 0]]
        mesh1.vertex_colors = open3d.utility.Vector3dVector(color_1)
        mesh2.vertex_colors = open3d.utility.Vector3dVector(color_2)
        p1 = open3d.geometry.PointCloud()
        p2 = open3d.geometry.PointCloud()
        p1.points = mesh1.vertices
        p2.points = mesh2.vertices
        p1.colors = mesh1.vertex_colors
        p2.colors = mesh2.vertex_colors
        p2.colors = open3d.utility.Vector3dVector([0.7, 0.7, 0.7] * np.ones(np.asarray(p2.points).shape))
        p1.colors = open3d.utility.Vector3dVector([0.7, 0.7, 0.7] * np.ones(np.asarray(p1.points).shape))

        corres = np.array([[i, i] for i in range(lm1.shape[0])])
        Rotation_12 = []
        Translation_12 = []
        # Transformation_12 = register_skeleton(lm2, lm1, corres, params)
        for i in range(lm1.shape[0]):
            xyz1_p = xyz1[NN_skel_1[:, 0] == i]
            xyz2_p = xyz2[NN_skel_2[:, 0] == i]
            if xyz2_p.shape[0] > 0 and xyz1_p.shape[0] > 0:
                T, R, t = my_icp(xyz2_p, xyz1_p)
            else:
                R = None
                t = None
            Rotation_12.append(R)
            Translation_12.append(t)
        for i in range(lm1.shape[0]):
            if Rotation_12[i] is None:
                if i > 0 and Rotation_12[i-1] is not None:
                    Rotation_12[i] = copy.deepcopy(Rotation_12[i-1])
                elif i < lm1.shape[0] - 1 and Rotation_12[i+1] is not None:
                    Rotation_12[i] = copy.deepcopy(Rotation_12[i+1])
                else:
                    Rotation_12[i] = np.eye(3)

            if Translation_12[i] is None:
                if i > 0 and Translation_12[i - 1] is not None:
                    Translation_12[i] = copy.deepcopy(Translation_12[i - 1])
                elif i < lm1.shape[0] - 1 and Translation_12[i + 1] is not None:
                    Translation_12[i] = copy.deepcopy(Translation_12[i + 1])
                else:
                    Translation_12[i] = np.mean(xyz1, axis=0) - np.mean(xyz2, axis=0)


    else:
        tree1 = KDTree(lm1)
        distance_to_skel_1, NN_skel_1 = tree1.query(xyz1, k=1)

        tree2 = KDTree(lm2)
        distance_to_skel_2, NN_skel_2 = tree2.query(xyz2, k=1)
        T, R, t = my_icp(xyz2, xyz1)
        # Transformation_12 = [icp_p2p(xyz2, xyz1, voxel_size).transformation]
        Rotation_12 = [R]
        Translation_12 = [t]


    tree_pcd_1 = KDTree(xyz1)
    xyz2_transform = copy.deepcopy(xyz2)
    for i in range(xyz2.shape[0]):
        # transform the point
        xyz_ = xyz2[i, :]
        NN_skel_ = NN_skel_2[i, :]

        if len(NN_skel_) > 1:
            R1 = Rotation_12[NN_skel_[0]]
            R2 = Rotation_12[NN_skel_[1]]
            t1 = Translation_12[NN_skel_[0]]
            t2 = Translation_12[NN_skel_[1]]

            distance_to_skel_ = distance_to_skel_2[i, :]

            distance_bw_skel = np.linalg.norm(lm2[NN_skel_[1]] - lm2[NN_skel_[0]])

            [A, B, C] = get_triangle_angles(distance_bw_skel, distance_to_skel_[0], distance_to_skel_[1])

            if C > np.pi / 2 - 10 ** -5:
                f0 = 1
                f1 = 0
            else:
                f0 = np.abs(np.cos(B)) / (np.abs(np.cos(B)) + np.abs(np.cos(C)))
                f1 = np.abs(np.cos(C)) / (np.abs(np.cos(B)) + np.abs(np.cos(C)))
        else:
            R1 = Rotation_12[NN_skel_[0]]
            R2 = Rotation_12[NN_skel_[0]]
            t1 = Translation_12[NN_skel_[0]]
            t2 = Translation_12[NN_skel_[0]]
            # print(T1)
            f0 = f1 = 0.5

        xyz_t_0 = xyz_ @ R1.T + t1.T
        xyz_t_1 = xyz_ @ R2.T + t2.T
        xyz2_transform[i, :] = (f0 * xyz_t_0 + f1 * xyz_t_1).flatten()
    T21 = tree_pcd_1.query(xyz2_transform, k=1, return_distance=False).flatten()
    return T21


if __name__ == "__main__":
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
        get_p2p_mapping_organ_collection(org_collection_1[:],
                                         org_collection_2,
                                         matches_index[:],
                                         day1=day1,
                                         day2=day2,
                                         dataset=dataset,
                                         if_match_skeleton=True,
                                         plot_mesh=True,
                                         plot_skeleton=False,
                                         plot_pcd=True,
                                         verbose=False,
                                         show_all=False)

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
                                         if_match_skeleton=True,
                                         verbose=False,
                                         show_all=False)
        t_end = time.time()
        print("Total time used: ", t_end - t_start)

    if dataset == "maize":
        org_collection_1 = [form_maize_org(day1)]
        org_collection_2 = [form_maize_org(day2)]
        matches_index = [0]
        get_p2p_mapping_organ_collection(org_collection_1,
                                         org_collection_2,
                                         matches_index,
                                         day1=day1,
                                         day2=day2,
                                         dataset=dataset,
                                         plot_mesh=False,
                                         plot_pcd=False,
                                         if_match_skeleton=True,
                                         verbose=False,
                                         show_all=False)
        t_end = time.time()
        print("Total time used: ", t_end - t_start)