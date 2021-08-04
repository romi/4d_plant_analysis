import os
import copy
import time
import json
import numpy as np
import sys
import open3d

sys.path.insert(0, "../")
from point_cloud_clean import clean_point_cloud
from sklearn.neighbors import KDTree
from p2p_matching_in_organ.landmark_utils import get_skeleton, get_mesh_landmarks
from utils import save_off, \
    point_cloud_to_mesh, \
    visualization_basis_function, \
    plot_skeleton_matching
from organ_matching.organ_matching_lr import match_organ_two_days
from skeleton_matching.skeleton_match import get_skeleton_landmarks_pairs
from p2p_matching_in_organ.p2p_matching_evaluation import evaluate_p2p_matching
from p2p_matching_in_organ.p2p_matching_visualization import visualize_p2p_matching
from functional_map_.functional import FunctionalMapping


def get_p2p_from_mesh(mesh1, mesh2, lm,
                      process_params=None, fit_params=None, refine_params=None, verbose=False):
    if not process_params:
        process_params = {
            'n_ev': (70, 70),  # Number of eigenvalues on source and Target
            'landmarks': lm,
            'subsample_step': 5,  # In order not to use too many descriptors
            'descr_type': 'MIX',  # WKS or HKS
        }

    process_params['landmarks'] = lm

    if not fit_params:
        fit_params = {
            'descr_mu': 1e0,
            'lap_mu': 1e-3,
            'descr_comm_mu': 1e-1,
            'orient_mu': 0,
        }

    if not refine_params:
        refine_params = {
            'icp_iter_num': 10,
            'zoomout_iter_num': 20
        }

    model1 = FunctionalMapping(mesh1, mesh2)
    model1.preprocess(**process_params, verbose=verbose)

    # Define parameters for optimization and fit the Functional Map

    model1.fit(**fit_params, verbose=verbose)

    if refine_params['icp_iter_num'] > 0:
        model1.icp_refine(nit=refine_params['icp_iter_num'], verbose=verbose)
    # model.change_FM_type('classic')  # Chose between 'classic', 'icp' or 'zoomout'

    if refine_params['zoomout_iter_num'] > 0:
        if verbose:
            print("max dimenstion of laplacian basis function: ", model1.max_dim)
        if model1.max_dim > model1.k1 + refine_params['zoomout_iter_num']:
            model1.zoomout_refine(nit=refine_params['zoomout_iter_num'], verbose=verbose)  # This refines the current
            # model.FM, be careful which FM type is
            # used
        else:
            if model1.max_dim > model1.k1:
                model1.zoomout_refine(nit=max(model1.max_dim - model1.k1, 1), verbose=verbose)
    p2p = model1.p2p
    return p2p


def double_direction_refinement(mesh1, mesh2, T21, T12, lm1=None, lm2=None):
    vertices1 = np.asarray(mesh1.vertices)
    vertices2 = np.asarray(mesh2.vertices)

    tree = KDTree(vertices1)
    NN_mesh1 = tree.query(np.asarray(vertices1), k=5, return_distance=False)

    # if lm1 is not None and lm2 is not None:
    #     T21[lm2] = lm1
    #     T12[lm1] = lm2

    for target_value in np.unique(T21):
        source_index = np.where(T21 == target_value)[0]
        if len(source_index) == 1:
            continue
        reflect_distance_min = 10 ** 4
        ind_keep = source_index[0]
        for ind in source_index:
            v_source = vertices2[ind]
            v_reflect = vertices2[T12[T21[ind]]]
            reflect_distance = np.linalg.norm(v_source - v_reflect)
            if reflect_distance < reflect_distance_min:
                reflect_distance_min = reflect_distance
                ind_keep = ind

        for ind in source_index:
            if ind != ind_keep:
                if ind in np.unique(T12):
                    target_candidates = np.where(T12 == ind)[0]
                else:
                    target_candidates = NN_mesh1[target_value, 1:]
                target_keep = target_candidates[0]
                for ind_t in target_candidates:
                    if ind_t not in np.unique(T21):
                        target_keep = ind_t
                T21[ind] = target_keep

    # NN_mesh1 = tree.query(np.asarray(vertices1), k=5, return_distance=False)
    for target in range(vertices1.shape[0]):
        if target not in np.unique(T21):
            for neighbor in NN_mesh1[target, 1:]:
                source_index = np.where(T21 == neighbor)[0]
                if len(source_index) < 1:
                    continue
                reflect_distance_min = 10 ** 4
                ind_keep = source_index[0]
                for ind in source_index:
                    v_source = vertices2[ind]
                    v_reflect = vertices2[T12[T21[ind]]]
                    reflect_distance = np.linalg.norm(v_source - v_reflect)
                    if reflect_distance < reflect_distance_min:
                        reflect_distance_min = reflect_distance
                        ind_keep = ind
                for ind in source_index:
                    if ind != ind_keep:
                        T21[ind] = target

    return T21, T12


def get_p2p_mapping_organ(pcd1, pcd2,
                          skel_landmarks1, skel_landmarks2,
                          verbose=False,
                          plot_pcd=False,
                          plot_mesh=False,
                          process_params=None,
                          apply_double_direction_refinement=True,
                          fit_params=None):
    options_path = "../hyper_parameters/lyon2.json"
    with open(options_path, "r") as json_file:
        options = json.load(json_file)

    # Load the parameters from the option
    pc_path = options['pc_path']
    skel_noted_path = options['skel_noted_path']  # the xyz of skeleton nodes with the segment index labelled
    segment_connect_path = options['segment_connect_path']  # the connection relationship among the segments
    stem_node_path = options['stem_node_path']  #
    mesh_radius_factor = options['mesh_radius_factor']
    pcd_clean_option = options['pcd_clean_option']
    pcd1, description_1 = clean_point_cloud(copy.deepcopy(pcd1), option=pcd_clean_option, verbose=verbose,
                                            translate=False)
    pcd2, description_2 = clean_point_cloud(copy.deepcopy(pcd2), option=pcd_clean_option, verbose=verbose,
                                            translate=False)
    if plot_pcd:
        open3d.visualization.draw_geometries([pcd1, pcd2])
    voxel_size = (description_1['voxel_size'] + description_2['voxel_size']) / 2

    mesh2 = point_cloud_to_mesh(pcd2, radius_factor=mesh_radius_factor)
    mesh1 = point_cloud_to_mesh(pcd1, radius_factor=mesh_radius_factor)
    # open3d.visualization.draw_geometries([mesh1, mesh2])
    if skel_landmarks2.shape[0] > 0 and skel_landmarks1.shape[0] > 0:
        lm1 = get_mesh_landmarks(mesh1, skel_landmarks1)
        lm2 = get_mesh_landmarks(mesh2, skel_landmarks2)
        # lm_len = min(lm1.shape[0], lm2.shape[0])
        lm = np.vstack([lm1, lm2]).T
    else:
        lm = None

    if plot_mesh:
        open3d.visualization.draw_geometries([mesh1, mesh2])

    T21 = get_p2p_from_mesh(mesh1, mesh2, lm, verbose=verbose, process_params=process_params, fit_params=fit_params)
    if lm is not None:
        lm = np.vstack([lm2, lm1]).T
    T12 = get_p2p_from_mesh(mesh2, mesh1, lm, verbose=verbose, process_params=process_params, fit_params=fit_params)
    if lm is not None and apply_double_direction_refinement == True:
        T21, T12 = double_direction_refinement(mesh1, mesh2, T21, T12, lm1, lm2)

    return mesh1, mesh2, T21, T12


def get_p2p_mapping_organ_collection(organ_collection_1,
                                     organ_collection_2,
                                     organ_matching,
                                     day1,
                                     day2,
                                     dataset,
                                     plot_mesh=False,
                                     plot_pcd=False,
                                     plot_skeleton=False,
                                     show_all=True,
                                     options=None):
    options_path = "../hyper_parameters/lyon2.json"
    if not options:
        with open(options_path, "r") as json_file:
            options = json.load(json_file)
    mesh_collection_1 = []
    mesh_collection_2 = []

    p2p_mappings = []
    scores = []
    for index, org1 in enumerate(organ_collection_1):
        if organ_matching[index] < 0:
            continue
        org2 = organ_collection_2[int(organ_matching[index])]

        # skel_landmarks1 = get_skeleton_landmarks(org1)
        # skel_landmarks2 = get_skeleton_landmarks(org2)

        xyz_1, connect_1, xyz_noted_1 = get_skeleton(org1)
        xyz_2, connect_2, xyz_noted_2 = get_skeleton(org2)
        #
        if plot_skeleton:
            plot_skeleton_matching(xyz_1, xyz_2, connect_1, connect_2)
        skel_landmarks1, skel_landmarks2 = get_skeleton_landmarks_pairs(xyz_1, xyz_2, connect_1, connect_2,
                                                                        visualize=False)
        skel_landmarks1 = skel_landmarks1[:, :3]
        skel_landmarks2 = skel_landmarks2[:, :3]
        print("Used: ", skel_landmarks1.shape[0], " landmarks")
        # skel_landmarks2[[-2, -1], :] = skel_landmarks2[[-1, -2], :]

        pcd1 = org1["pcd"]
        pcd2 = org2["pcd"]

        mesh1, mesh2, T21, T12 = get_p2p_mapping_organ(pcd1, pcd2, skel_landmarks1, skel_landmarks2,
                                                       verbose=True,
                                                       plot_pcd=plot_pcd,
                                                       plot_mesh=plot_mesh)

        score = evaluate_p2p_matching(mesh1, mesh2, skel_landmarks1, skel_landmarks2, T21, T12, sample_proportion=0.5)
        print("score: ", score)
        scores.append(score)
        print(64 * "=")
        mesh_collection_1.append(mesh1)
        mesh_collection_2.append(mesh2)
        p2p_mappings.append(copy.deepcopy(T21))

    save_path_format = options["save_path_segment"]
    i = 0
    for m1, m2 in zip(mesh_collection_1, mesh_collection_2):
        np.savetxt(save_path_format.format(dataset, day1, day2) + "{}_{}.csv".format(day1, i), np.asarray(m1.vertices))
        np.savetxt(save_path_format.format(dataset, day1, day2) + "{}_{}.csv".format(day2, i), np.asarray(m2.vertices))
        np.savetxt(save_path_format.format(dataset, day1, day2) + "match_{}.csv".format(i), p2p_mappings[i])
        i += 1

    visualize_p2p_matching(day1, day2, dataset, save_path_format=save_path_format, show_all=True)
    print(scores)
    print(np.mean(scores))
    open3d.visualization.draw_geometries(pcd1 + pcd2)


if __name__ == "__main__":
    day1 = "03-22_PM"
    day2 = "03-23_PM"
    t_start = time.time()
    matches_index, org_collection_1, org_collection_2 = \
        match_organ_two_days(day1, day2, visualize=True)
    # org_collection_1 = preprocess_pcd(org_collection_1)
    # org_collection_2 = preprocess_pcd(org_collection_2)
    t_end = time.time()
    print("Get the organs matched, used ", t_end - t_start, " s")

    t_start = time.time()
    # get_p2p_mapping_organ_collection(org_collection_1[0:1] + org_collection_1[3:],
    #                                  org_collection_2,
    #                                  np.append(matches_index[0:1], matches_index[3:]),
    #                                  day1=day1,
    #                                  day2=day2,
    #                                  plot_mesh=False,
    #                                  plot_pcd=False,
    #                                  show_all=False)
    #
    # get_p2p_mapping_organ_collection(org_collection_1[3:5],
    #                                  org_collection_2,
    #                                  matches_index[3:5],
    #                                  day1=day1,
    #                                  day2=day2,
    #                                  plot_mesh=False,
    #                                  plot_pcd=False,
    #                                  show_all=False)

    org_1 = org_collection_1[0:1] + org_collection_1[3:]
    org_2 = org_collection_2[0:1] + org_collection_2[3:]
    match = np.append(matches_index[0:1], matches_index[3:])
    pcd_1 = open3d.geometry.PointCloud()
    pcd_1.points = open3d.utility.Vector3dVector(
        np.vstack([np.asarray(org["pcd"].points) for org in org_1])
    )

    pcd_2 = open3d.geometry.PointCloud()
    pcd_2.points = open3d.utility.Vector3dVector(
        np.vstack([np.asarray(org["pcd"].points) for org in org_2])
    )

    # open3d.visualization.draw_geometries([pcd_1, pcd_2])
    # skel_landmarks1, skel_landmarks2 = [], []
    skel_landmarks1 = np.loadtxt("landmarks/landmark_{}.csv".format(day1))
    skel_landmarks2 = np.loadtxt("landmarks/landmark_{}.csv".format(day2))
    # for index, org1 in enumerate(org_1):
    #     org2 = org_collection_2[int(match[index])]
    #
    #     xyz_1, connect_1 = get_skeleton(org1)
    #     xyz_2, connect_2 = get_skeleton(org2)
    #     skel_lm1, skel_lm2 = get_skeleton_landmarks_pairs(xyz_1, xyz_2, connect_1, connect_2,
    #                                                                     visualize=False)
    #     skel_landmarks1.append(skel_lm1[::5])
    #     skel_landmarks2.append(skel_lm2[::5])
    #
    # skel_landmarks1 = np.vstack(skel_landmarks1)
    # skel_landmarks2 = np.vstack(skel_landmarks2)

    mesh1, mesh2, T21, T12 = get_p2p_mapping_organ(pcd_1, pcd_2, skel_landmarks1, skel_landmarks2,
                                                   verbose=True,
                                                   plot_pcd=True,
                                                   plot_mesh=True)
    score = evaluate_p2p_matching(mesh1, mesh2, skel_landmarks1, skel_landmarks2, T21, T12, sample_proportion=0.5)
    print("score: ", score)
    mesh_collection_1 = []
    mesh_collection_2 = []
    p2p_mappings = []

    mesh_collection_1.append(mesh1)
    mesh_collection_2.append(mesh2)
    p2p_mappings.append(copy.deepcopy(T21))

    color_bound_box = np.array([[-1000.0, -1000.0, -1000.0], [1000.0, 1000.0, 1000.0]])
    for i in range(len(mesh_collection_2)):
        m2 = mesh_collection_2[i]
        P = np.asarray(m2.vertices)
        P_min = np.min(P, axis=0)
        P_max = np.max(P, axis=0)
        color_bound_box[0, :] = np.max(np.vstack([color_bound_box[0, :], P_max]), axis=0)
        color_bound_box[1, :] = np.min(np.vstack([color_bound_box[1, :], P_min]), axis=0)

    # print(color_bound_box)

    for index, m2 in enumerate(mesh_collection_2):
        P = np.asarray(m2.vertices)
        color_value = np.zeros(P.shape)
        color_value = (P - color_bound_box[0, :]) / (color_bound_box[1, :] - color_bound_box[0, :])
        m2.vertex_colors = open3d.utility.Vector3dVector(copy.deepcopy(color_value))

        m1 = mesh_collection_1[index]
        color_value_1 = 0.8 * np.ones((np.asarray(m1.vertices).shape[0], 3))
        p2p = p2p_mappings[index]
        color_value_1[p2p] = color_value
        points_mapped = np.unique(p2p)
        tree = KDTree(np.asarray(m1.vertices)[points_mapped])
        all_mapped = tree.query(np.asarray(m1.vertices), k=1, return_distance=False).flatten()
        # color_value_1 = color_value_1[points_mapped[all_mapped]]
        m1.vertex_colors = open3d.utility.Vector3dVector(color_value_1)
        m1.translate([0, 100, 0])

    pcd1 = []
    pcd2 = []
    for m1, m2 in zip(mesh_collection_1, mesh_collection_2):
        p1 = open3d.geometry.PointCloud()
        p2 = open3d.geometry.PointCloud()
        p1.points = m1.vertices
        p1.colors = m1.vertex_colors
        p2.points = m2.vertices
        p2.colors = m2.vertex_colors
        pcd1.append(p1)
        pcd2.append(p2)
    print(score)

    open3d.visualization.draw_geometries(pcd1 + pcd2)

    t_end = time.time()
    print(t_end - t_start)
