import numpy as np
import open3d
from sklearn.neighbors import KDTree
from point_cloud_clean import clean_point_cloud
from skeletonization.Laplacian_skeletonization import laplacian_skeletonization, skeleton_extraction_by_day
from skeletonization.skeleton_segmentation import skeleton_segmentation
from utils import point_cloud_to_mesh
import networkx as nx
import time
import os
import json
import colorsys
from collections import Counter

color_setting = np.array([list(colorsys.hsv_to_rgb(i / 20, 0.82, 0.96)) for i in range(20)])


# color_setting = color_setting / 255

# pc_path = '../data/lyon2/processed/{}_segmented.ply'
# skel_path = "../data/lyon2/{}/skeleton_{}_connected.txt"
# skel_noted_path = "../data/lyon2/{}/skeleton_{}_noted.csv"
# branch_connect_path = "../data/lyon2/{}/branch_connection_{}.csv"
#
# mesh_radius_factor = 2
#
# pcd_clean_option = {'downsample_rate': 80,
#                     'voxel_size': 0.7,
#                     'crop': False,
#                     'crop_low_bound': 0.0,
#                     'crop_up_bound': 1}


def get_head_tail(xyz):
    head = np.zeros(3)
    tail = np.zeros(3)
    max_dist = 0
    for i in range(xyz.shape[0]):
        for j in range(i + 1, xyz.shape[0]):
            xyz_1 = xyz[i]
            xyz_2 = xyz[j]
            if np.linalg.norm(xyz_1 - xyz_2) > max_dist:
                max_dist = np.linalg.norm(xyz_1 - xyz_2)
                head = xyz_1
                tail = xyz_2
    head, tail = (tail, head) if tail[2] < head[2] else (head, tail)
    return head, tail


def point_cloud_segmentation_from_skeleton(xyz_labeled,
                                           branch_connect,
                                           mesh,
                                           visualize=False,
                                           verbose=False,
                                           options=None):
    """

    :param xyz_labeled:
    :param branch_connect:
    :param mesh:
    :param visualize:
    :param verbose:
    :param options:
    :return:
    """
    if options is None:
        options_path = "../hyper_parameters/lyon2.json"
        with open(options_path, "r") as json_file:
            options = json.load(json_file)

    xyz_skel, labels = xyz_labeled[:, :3], xyz_labeled[:, 3]
    xyz_pcd = np.asarray(mesh.vertices)
    label_pcd = -1 * np.ones(xyz_pcd.shape[0])

    semantic_color_pcd = np.asarray(mesh.vertex_colors)
    if len(semantic_color_pcd) == 0:
        semantic_color_pcd = [1, 0, 0] * np.ones(xyz_pcd.shape)
    semantic_color_setting = [(1, 0, 0), (0, 1, 0), (1, 1, 0), (1, 0, 1), (1, 1, 1)]
    semantic_value_pcd = -1 * np.ones(xyz_pcd.shape[0])

    t_0 = time.time()
    t_start = time.time()
    tree = KDTree(xyz_skel)
    distance, skel_matches = tree.query(xyz_pcd, k=1, return_distance=True)
    t_end = time.time()
    if verbose:
        print("Label the Point Cloud by Nearest Skeleton Point" + " :used {:.2f} s".format(t_end - t_start))

    graph_branch = nx.Graph()
    branch_label = np.unique(labels)
    graph_branch.add_nodes_from(branch_label)
    for c in branch_connect:
        graph_branch.add_edge(c[0], c[1], weight=1)
    graph_branch.add_edges_from(branch_connect)
    graph_branch = nx.minimum_spanning_tree(graph_branch)

    for i in range(xyz_pcd.shape[0]):
        label_pcd[i] = labels[skel_matches[i, 0]]
        if tuple(semantic_color_pcd[i]) in semantic_color_setting:
            semantic_value_pcd[i] = semantic_color_setting.index(tuple(semantic_color_pcd[i]))

    # Calculate the center position of each branch
    branch_center = np.zeros((branch_label.shape[0], 3))
    for i in branch_label:
        branch_center[int(i), :] = xyz_skel[labels == i].mean(axis=0)

    # Calculate the root part (soil part)
    root = np.argmin(branch_center[:, 2])

    # Calculate the direction of each branch
    branch_orientation = np.zeros((branch_label.shape[0], 3))
    for i in branch_label:
        xyz_br = xyz_skel[labels == i]
        head, tail = get_head_tail(xyz_br)
        branch_orientation[int(i)] = tail - head

    # Get the semantic meaning of each branch
    branch_semantic_value = np.zeros((len(branch_label), len(semantic_color_setting)))
    for i in branch_label:
        semantic_vote = Counter(semantic_value_pcd[label_pcd == i])
        for j in range(len(semantic_color_setting)):
            branch_semantic_value[int(i), int(j)] = semantic_vote[float(j)]

    # color the branches
    branch_color = {}
    for br in np.unique(labels):
        sp = nx.shortest_path(graph_branch, source=root, target=br)
        color_index = len(sp) - 1
        color_orientation = branch_orientation[int(br)] / (1 ** -5 + np.linalg.norm(branch_orientation[int(br)]))
        branch_color[br] = np.array(list(colorsys.hsv_to_rgb(color_index / 10,
                                                             1 ** -3 * np.sqrt((np.abs(color_orientation[0]) +
                                                                                np.abs(color_orientation[1])) / 2),
                                                             1)))
        # branch_color[br] = np.random.rand(3)
    organ_color_values = 0.5 * np.ones((xyz_pcd.shape[0], 3))
    for i in range(xyz_pcd.shape[0]):
        organ_color_values[i, :] = branch_color[labels[skel_matches[i, 0]]]

    mesh.vertex_colors = open3d.utility.Vector3dVector(organ_color_values)

    centers_np = branch_center

    if visualize:
        line_set = open3d.geometry.LineSet()
        line_set.points = open3d.utility.Vector3dVector(centers_np)
        line_set.lines = open3d.utility.Vector2iVector(np.array(graph_branch.edges))

        pcd_branches_center = open3d.geometry.PointCloud()
        pcd_branches_center.points = open3d.utility.Vector3dVector(centers_np)
        p_ = open3d.geometry.PointCloud()
        p_.points = mesh.vertices
        p_.colors = mesh.vertex_colors
        p_.estimate_normals()
        open3d.visualization.draw_geometries([p_, pcd_branches_center, line_set])

    if verbose:
        print("segmented the point cloud using the skeleton ::used {:.2f} s".format(time.time() - t_0))
    return graph_branch, branch_center, branch_orientation, branch_semantic_value, mesh, label_pcd


def point_cloud_segmentation(pcd,
                             day,
                             dataset,
                             path_format="../data/{}/{}/",
                             verbose=False,
                             visualize=True,
                             options=None):
    """
    Whole pipeline to segment a plant from point cloud
    :return:
    """

    if options is None:
        options_path = "../hyper_parameters/lyon2.json"
        with open(options_path, "r") as json_file:
            options = json.load(json_file)

    t_0 = time.time()
    pc_path = options['pc_path']
    skel_noted_path = options['skel_noted_path'] # the xyz of skeleton nodes with the segment index labelled
    skel_path = options['skel_path']
    segment_connect_path = options['segment_connect_path']  # the connection relationship among the segments
    stem_node_path = options['stem_node_path']  #
    mesh_radius_factor = options['mesh_radius_factor']
    # skel_path = path_format.format(dataset, day) + "skeleton_pcd.csv".format(day)
    if os.path.exists(skel_path):
        if_contract_skel_pcd = False
    else:
        if_contract_skel_pcd = True

    t_start = time.time()

    pcd, description = clean_point_cloud(pcd, options["pcd_clean_option"])
    voxel_size = description['voxel_size']
    t_end = time.time()
    if verbose:
        print("Clean the point Cloud : used {:.2f}s".format(t_end - t_start))
        print("")

    mesh = point_cloud_to_mesh(pcd, radius_factor=options['mesh_radius_factor'])

    if verbose:
        if not if_contract_skel_pcd:
            print(":load calculated contracted center points")
        else:
            print(":calculating contracted center points")
    t_start = time.time()

    # Apply the skeleton extraction
    xyz, connection = skeleton_extraction_by_day(day,
                                                 dataset,
                                                 load_data=if_contract_skel_pcd,
                                                 verbose=False,
                                                 visualize=False)
    t_end = time.time()
    if verbose:
        print("Skeleton extracted :used {:.2f}s".format(t_end - t_start))
        print("")

    t_start = time.time()

    if "skel_segmentation" not in options["skeleton_extract_option"]:
        options["skeleton_extract_option"][day]["skel_segmentation"] = {}

    # Apply the Skeleton Segmentation
    xyz_labeled, branch_connection = skeleton_segmentation(xyz,
                                                           connection,
                                                           day,
                                                           dataset,
                                                           visualize=visualize,
                                                           verbose=visualize,
                                                           options=options)
    t_end = time.time()
    if verbose:
        print("Skeleton segmented :used {:.2f}s".format(t_end - t_start))
        print("")

    t_start = time.time()
    graph_branch, branch_center, branch_orientation, branch_semantic_value, mesh, label_pcd = \
        point_cloud_segmentation_from_skeleton(xyz_labeled, branch_connection, mesh,
                                               visualize=False,
                                               verbose=verbose,
                                               options=options)
    t_end = time.time()
    if verbose:
        print("Point cloud segmented :used {:.2f}s".format(t_end - t_start))
        print("")

    if verbose:
        print(64 * "=")
        print("Finished Segmentation")
        print("::used {:.2f}s".format(t_end - t_0))
        print("")
    return graph_branch, branch_center, branch_orientation, branch_semantic_value, mesh, label_pcd


if __name__ == "__main__":
    day = "03-23_PM"
    dataset = "lyon2"
    # print(color_setting)
    # xyz_labeled = np.loadtxt(skel_noted_path.format(day, day))
    # branch_connect = np.loadtxt(branch_connect_path.format(day, day))
    pc_path = '../data/lyon2/processed/{}_segmented.ply'
    pcd = open3d.io.read_point_cloud(pc_path.format(day))
    pcd_clean_option = {'downsample_rate': 80,
                        'voxel_size': 0.7,
                        'crop': False,
                        'crop_low_bound': 0.0,
                        'cluster_distance_threshold': 5,
                        'crop_up_bound': 1}
    pcd, description = clean_point_cloud(pcd, option=pcd_clean_option)
    # voxel_size = description['voxel_size']
    # mesh = point_cloud_to_mesh(pcd, radius_factor=mesh_radius_factor)

    options_path = "../hyper_parameters/lyon2.json"
    with open(options_path, "r") as json_file:
        options = json.load(json_file)

    # graph_branch, branch_center, branch_orientation, branch_semantic_value, mesh, _ = point_cloud_segmentation_from_skeleton(
    #     xyz_labeled,
    #     branch_connect,
    #     mesh,
    #     visualize=True,
    #     verbose=True)

    point_cloud_segmentation(pcd,
                             day,
                             dataset,
                             path_format="../data/{}/{}/",
                             verbose=True,
                             visualize=True,
                             options=options)
