import json
import numpy as np
import open3d
import copy
import sys
import os
from point_cloud_clean import clean_point_cloud
from skeletonization.point_cloud_segmentation import point_cloud_segmentation_from_skeleton, point_cloud_segmentation
from skeletonization.organ_segmentation import organ_segmentation, get_collection_organ_pcd
from utils import point_cloud_to_mesh


sys.path.insert(0, "../")


def get_precision(match, correct_match):
    correct_count = 0
    for i in range(min(len(correct_match), len(match))):
        if correct_match[i] == match[i]:
            correct_count += 1
    return correct_count / min(len(correct_match), len(match))


def get_organ_skeletonized(org_pcd_collect_1, stem_connect_xyz_1, org_dict_1, xyz_skel_labeled_1, branch_connect_1):
    """

    Find the skeleton nodes for each organ in the organ collection, and then regroup them into a dictionary

    :param org_pcd_collect_1: List of Point cloud: the collection of point clouds of each organ to process
    :param stem_connect_xyz_1: Numpy array: the connect point of each branch on the main stem
    :param org_dict_1: dictionary: the dictionary containing the index of segments for each organ
                                    org_dict["stem"] are the segments for main stem
                                    org_dict["root"] are the segments for root
                                    org_dict["branches"][i] are the i-th branch
    :param xyz_skel_labeled_1: Numpy array of shape [N, 4]:
                                the xyz values of the skeleton nodes labeled with the segments index
    :param branch_connect_1: Numpy array of shape [M, 2]: defining the connection between segments
    :return:
    """
    org_collect_1 = []
    for i in range(len(org_pcd_collect_1)):
        org_set = {
            "pcd": org_pcd_collect_1[i],
            "stem_connect_point": stem_connect_xyz_1[i, :]
        }
        segment_index = []
        connect = np.array([])
        if i == 0:
            skeleton = xyz_skel_labeled_1[xyz_skel_labeled_1[:, 3] == org_dict_1["stem"]]
            segment_index = [org_dict_1["stem"]]
            org_set["stem_connect_point"] = stem_connect_xyz_1[3:, :]
        elif i == 1:
            skeleton = xyz_skel_labeled_1[xyz_skel_labeled_1[:, 3] == org_dict_1["root"]]
            segment_index = [org_dict_1["root"]]
            org_set["stem_connect_point"] = None

        elif i == 2:
            if len(org_dict_1["root_branch"]) > 0 and np.asarray(org_set["pcd"].points).shape[0] > 10:
                segment_index = org_dict_1["root_branch"]
                skeleton = np.vstack(
                    [xyz_skel_labeled_1[xyz_skel_labeled_1[:, 3] == seg] for seg in org_dict_1["root_branch"]])
                org_set["stem_connect_point"] = None
                connect = np.vstack(
                    [branch_connect_1[np.any(branch_connect_1 == seg, axis=1)] for seg in org_dict_1["root_branch"]])
            else:
                org_set["pcd"] =  org_pcd_collect_1[1]
                org_set["stem_connect_point"] = stem_connect_xyz_1[1, :]
                skeleton = xyz_skel_labeled_1[xyz_skel_labeled_1[:, 3] == org_dict_1["root"]]
                segment_index = [org_dict_1["root"]]
                org_set["stem_connect_point"] = None
        else:
            segment_index = org_dict_1["branches"][i - 3]
            skeleton = np.vstack(
                [xyz_skel_labeled_1[xyz_skel_labeled_1[:, 3] == seg] for seg in org_dict_1["branches"][i - 3]])
            connect = np.vstack(
                [branch_connect_1[np.any(branch_connect_1 == seg, axis=1)] for seg in org_dict_1["branches"][i - 3]])
        org_set["skeleton_pcd"] = skeleton
        org_set["skeleton_connection"] = connect
        org_set["segment_index"] = segment_index
        org_collect_1.append(copy.deepcopy(org_set))
    return org_collect_1


def visualize_matching(org_collect_1, org_collect_2, matches, not_in_one=True):
    oc1 = copy.deepcopy(org_collect_1)
    oc2 = copy.deepcopy(org_collect_2)

    points = np.zeros((2 * len(oc1), 3))
    lines = np.zeros((len(oc1), 2))
    visited = set([-1])
    for i in range(len(oc1)):
        oc1[i].translate([0, 150 * i * int(not_in_one), 0])
        # oc1[i].estimate_normals()

        if matches[i] not in visited and np.asarray(oc1[i].colors).shape[0] > 0:
            color_1 = np.asarray(oc1[i].colors)[0, :]
            color_2 = np.ones(np.asarray(oc2[int(matches[i])].points).shape)
            color_2 = color_2 * color_1

            oc2[int(matches[i])].translate([250 * i * int(not_in_one), 200, -100 * int(not_in_one)])
            oc2[int(matches[i])].colors = open3d.utility.Vector3dVector(color_2)
            # oc2[int(matches[i])].estimate_normals()
            visited.add(int(matches[i]))

            p1 = np.mean(np.asarray(oc1[i].points), axis=0)
            p2 = np.mean(np.asarray(oc2[int(matches[i])].points), axis=0)
            points[2 * i, :] = p1
            points[2 * i + 1, :] = p2
            lines[i, :] = np.array([2 * i, 2 * i + 1])
    lineset = open3d.geometry.LineSet()
    lineset.points = open3d.utility.Vector3dVector(points)
    lineset.lines = open3d.utility.Vector2iVector(lines)
    add_up = []
    if not not_in_one:
        for i in range(len(oc2)):
            if i not in visited:
                oc2[i].translate([0, 200, 0])
                color_2 = 0.95 * np.ones(np.asarray(oc2[i].points).shape)
                oc2[i].colors = open3d.utility.Vector3dVector(color_2)
                # oc2[i].estimate_normals()
                add_up.append(oc2[i])

    vis_list = oc1 + [oc2[int(matches[i])] for i in range(len(oc1))] + add_up + [lineset]
    open3d.visualization.draw_geometries(vis_list)


def preprocess_data(day1, dataset, visualize=False, verbose=False, options=None):
    """
    given the day and the data set, extract the target plant's point cloud, and apply skeleton extraction, geometric
    segmentation and organ formation on it.

    :param day1: the day time of the plant, in form "MM-DD_AM(PM)"
    :param dataset: the data set name to fetch the data, in "lyon" or "tomato"
    :param visualize:
    :param verbose:
    :param options: the parameters used
    :return:
    """

    assert dataset in ["lyon2", "tomato"]

    if options is None:
        options_path = "../hyper_parameters/lyon2.json"
        with open(options_path, "r") as json_file:
            options = json.load(json_file)

    # Load the parameters from the option
    pc_path = options['pc_path']
    skel_noted_path = options['skel_noted_path']  # the xyz of skeleton nodes with the segment index labelled
    segment_connect_path = options['segment_connect_path']  # the connection relationship among the segments
    stem_node_path = options['stem_node_path']  # the path to the xyz of main stem
    mesh_radius_factor = options['mesh_radius_factor'] # the radius for mesh creation

    if not os.path.exists(pc_path.format(dataset, day1)):
        print("point cloud document not found: " + pc_path.format(dataset, day1))

    # Load the data
    pcd1 = open3d.io.read_point_cloud(pc_path.format(dataset, day1))
    pcd1 = clean_point_cloud(pcd1, option=options["pcd_clean_option"])[0]
    mesh1 = point_cloud_to_mesh(pcd1, radius_factor=mesh_radius_factor)

    # Apply the geometric segmentation on the point cloud

    if os.path.exists(skel_noted_path.format(dataset, day1, day1)) and \
            os.path.exists(segment_connect_path.format(dataset, day1, day1)) and \
            os.path.exists(stem_node_path.format(dataset, day1)):

        # If the skeleton data has been calculated, we load the data
        xyz_labeled_1 = np.loadtxt(skel_noted_path.format(dataset, day1, day1))
        branch_connect_1 = np.loadtxt(segment_connect_path.format(dataset, day1, day1))
        xyz_stem_node_1 = np.loadtxt(stem_node_path.format(dataset, day1))

        gb1, bc1, bo1, bsc1, mesh1, label_mesh_1 = point_cloud_segmentation_from_skeleton(
            xyz_labeled_1,
            branch_connect_1,
            mesh1,
            visualize=False,
            verbose=verbose,
            options=options)
    else:
        # Calculate the skeleton and apply segmentation from scratch
        gb1, bc1, bo1, bsc1, mesh1, label_mesh_1 = point_cloud_segmentation(pcd1,
                                                                            day1,
                                                                            dataset,
                                                                            path_format="../data/{}/{}_stem/",
                                                                            verbose=verbose,
                                                                            visualize=visualize,
                                                                            options=options)
        xyz_labeled_1 = np.loadtxt(skel_noted_path.format(dataset, day1, day1))
        branch_connect_1 = np.loadtxt(segment_connect_path.format(dataset, day1, day1))
        xyz_stem_node_1 = np.loadtxt(stem_node_path.format(dataset, day1))

    os_opt = {}
    if day1 in options['organ_segment_option']:
        os_opt = options['organ_segment_option'][day1]

    # reforming the segments into organs
    org_dict_1, bi1 = organ_segmentation(gb1, bc1, mesh1, label_mesh_1, xyz_labeled_1, xyz_stem_node_1,
                                         is_include_root=dataset == "lyon2",
                                         option=os_opt)

    org_pcd_collect_1 = get_collection_organ_pcd(mesh1, bi1)

    return org_dict_1, org_pcd_collect_1, gb1, xyz_labeled_1, xyz_stem_node_1, branch_connect_1
