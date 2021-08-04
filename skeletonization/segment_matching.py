import random

import numpy as np
import open3d
import networkx as nx
from organ_matching.organ_matching_lr import point_cloud_segmentation_from_skeleton
from skeletonization.organ_segmentation import organ_segmentation
from utils import point_cloud_to_mesh
from point_cloud_clean import clean_point_cloud

import time
from convert import nn_search
from functional_map_.fast_sinkhorn_filter import fast_sinkhorn_filter

day1 = '03-22_AM'
day2 = '03-22_PM'
pc_path = '../data/lyon2/processed/{}_segmented.ply'
skel_path = "../data/lyon2/{}_stem/skeleton_{}_connected.txt"
skel_noted_path = "../data/lyon2/{}_stem/skeleton_{}_noted.csv"
branch_connect_path = "../data/lyon2/{}_stem/branch_connection_{}.csv"
mesh_radius_factor = 3

pcd_clean_option = {'downsample_rate': 80,
                    'voxel_size': 0.5,
                    'crop': False,
                    'crop_low_bound': 0.0,
                    'crop_up_bound': 1}


def get_description(branch_graph,
                    branch_center,
                    branch_orientation,
                    branch_semantic_value,
                    organ_segmentation_dict,
                    root,
                    stem,
                    weight=[0.1, 3, 0.1, 5, 5, 0.01, 3]):

    branch_center = branch_center - np.array([0, 0, branch_center[root,2]])
    diag = np.linalg.norm(branch_orientation, axis=1)
    diag = diag.reshape(-1,1) + 2 ** -5
    branch_orientation = branch_orientation / diag
    geo_dist = np.zeros((branch_center.shape[0], 1))
    stem_dist = np.zeros((branch_center.shape[0], 1))
    degree = np.zeros((branch_center.shape[0], 1))
    organ_index = np.zeros((branch_center.shape[0], 1))
    branch_semantic_value = branch_semantic_value / np.sum(branch_semantic_value, axis=1).reshape(-1, 1)

    for i in range(branch_center.shape[0]):
        # Get the shortest path from root to ith node
        sp_root = nx.shortest_path(branch_graph, source=root, target=i)
        geo_dist[i, 0] = len(sp_root) - 1

        # Get the shortest path from stem to ith node
        sp_stem = nx.shortest_path(branch_graph, source=stem, target=i)
        geo_dist[i, 0] = len(sp_stem) - 1

        # Get the degree of the ith node
        degree[i, 0] = branch_graph.degree[i]

        # Get the organ index the ith node belongs to
        if i == organ_segmentation_dict["root"]:
            organ_index[i, 0] = 100
        elif i == organ_segmentation_dict["stem"]:
            organ_index[i, 0] = 200
        else:
            for j in range(len(organ_segmentation_dict["branches"])):
                if i in organ_segmentation_dict["branches"][j]:
                    organ_index[i, 0] = j

    return np.hstack([weight[0] * branch_center,
                      weight[1] * branch_orientation,
                      weight[2] * diag,
                      weight[3] * geo_dist,
                      weight[4] * stem_dist,
                      weight[5] * branch_semantic_value,
                      weight[6] * organ_index,
                      weight[7] * degree])


def get_segment_matching(desc_1, desc_2, using_fast_sinkhorn = True):

    if using_fast_sinkhorn:
        T12, T21 = fast_sinkhorn_filter(desc_1, desc_2, options={'maxiter': 400})
        matches = T21
    else:
        matches = nn_search(desc_2, desc_1)

    return matches


if __name__ == "__main__":

    day1 = "03-22_AM"
    day2 = "03-22_PM"
    print("Preprocessing: get the segments")
    print(64 * "-")
    t_start = time.time()

    xyz_labeled_1 = np.loadtxt(skel_noted_path.format(day1, day1))
    branch_connect_1 = np.loadtxt(branch_connect_path.format(day1, day1))
    pcd1 = open3d.io.read_point_cloud(pc_path.format(day1))
    pcd1, description = clean_point_cloud(pcd1, option=pcd_clean_option)
    voxel_size = description['voxel_size']
    mesh1 = point_cloud_to_mesh(pcd1, radius_factor=mesh_radius_factor)

    gb1, bc1, bo1, bsc1, mesh1, label_mesh_1 = point_cloud_segmentation_from_skeleton(
        xyz_labeled_1,
        branch_connect_1,
        mesh1,
        visualize=True,
        verbose=True)
    organ_dict_1, bi1 = organ_segmentation(gb1, bc1, mesh1, label_mesh_1, xyz_labeled_1)

    xyz_labeled_2 = np.loadtxt(skel_noted_path.format(day2, day2))
    branch_connect_2 = np.loadtxt(branch_connect_path.format(day2, day2))
    pcd2 = open3d.io.read_point_cloud(pc_path.format(day2))
    pcd2, description = clean_point_cloud(pcd2, option=pcd_clean_option)
    voxel_size = description['voxel_size']
    mesh2 = point_cloud_to_mesh(pcd2, radius_factor=mesh_radius_factor)

    gb2, bc2, bo2, bsc2, mesh2, label_mesh_2 = point_cloud_segmentation_from_skeleton(
        xyz_labeled_2,
        branch_connect_2,
        mesh2,
        visualize=False,
        verbose=True)
    organ_dict_2, bi2 = organ_segmentation(gb2, bc2, mesh2, label_mesh_2, xyz_labeled_2)

    root_1 = np.argmin(bc1[:, 2])
    root_2 = np.argmin(bc2[:, 2])
    stem_1 = max(gb1.nodes, key=lambda n: len(list(gb1.neighbors(n))))
    stem_2 = max(gb2.nodes, key=lambda n: len(list(gb2.neighbors(n))))

    mesh2.translate([0, 0, bc1[root_1,2] - bc2[root_2,2]])
    mesh2.translate([0, 100, 0])
    open3d.visualization.draw_geometries([mesh1, mesh2])

    t_end = time.time()
    print("Segments in first plant: {}".format(len(bc1)))
    print("Segments in second plant: {}".format(len(bc2)))
    print("Used {:.2f}s".format(t_end - t_start))

    root_1 = np.argmin(bc1[:, 2])
    root_2 = np.argmin(bc2[:, 2])

    bc2 = bc2 + np.array([0, 0, bc1[root_1][2] - bc2[root_2][2]])

    print("")
    print("Establishing Matching between Segments")
    print(64 * "-")

    weight = [0.5, 3, 0.1, 5, 10, 0, 1]
    # desc_1 = get_description(gb1, bc1, bo1, bsc1, root_1, stem_1, weight)
    # desc_2 = get_description(gb2, bc2, bo2, bsc2, root_2, stem_2, weight)
    # matches = get_segment_matching(desc_1, desc_2)
    # inject_rate = np.unique(matches).shape[0] / len(matches)
    # print("::{:.2f} segments get mapped".format(inject_rate))

    weight_scale = [1, 10, 0.1, 10, 30, 0.1, 1, 5]
    optimal_weight = weight_scale
    optimal_match = None
    optimal_inject_rate = 0

    for _ in range(500):
        weight = np.random.rand(8)
        weight = weight * weight_scale
        desc_1 = get_description(gb1, bc1, bo1, bsc1, organ_dict_1, root_1, stem_1, weight)
        desc_2 = get_description(gb2, bc2, bo2, bsc2, organ_dict_2, root_2, stem_2, weight)
        matches = get_segment_matching(desc_1, desc_2)
        inject_rate = np.unique(matches).shape[0] / len(matches)
        if inject_rate > optimal_inject_rate:
            optimal_inject_rate = inject_rate
            optimal_weight = weight
            optimal_match = matches
    print("::optimal weight: ", optimal_weight)
    print("::{:.2f} segments get mapped".format(optimal_inject_rate))
    matches = optimal_match

    # Visualization of Results
    v2 = np.asarray(mesh2.vertices)
    v1 = np.asarray(mesh1.vertices)

    c1 = np.asarray(mesh1.vertex_colors)
    c2 = np.asarray(mesh2.vertex_colors)

    color_value_2 = 0.95 * np.ones(np.asarray(mesh2.vertices).shape)

    mesh2.vertex_colors = open3d.utility.Vector3dVector(color_value_2)
    open3d.visualization.draw_geometries([mesh1, mesh2])

    for i in range(label_mesh_2.shape[0]):
        l2 = label_mesh_2[i]
        matched_from = np.where(matches==l2)[0]
        if matched_from.shape[0] == 0:
            continue
        if matched_from.shape[0] > 1:
            j = random.randint(0, matched_from.shape[0]-1)
        else:
            j = 0
        l1 = matched_from[j]
        color_value_2[i] = c1[label_mesh_1 == l1][0]

    mesh2.vertex_colors = open3d.utility.Vector3dVector(color_value_2)
    open3d.visualization.draw_geometries([mesh1, mesh2])
