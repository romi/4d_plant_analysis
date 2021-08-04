import os
import copy
import time
import numpy as np
import colorsys
import networkx as nx
import open3d
from point_cloud_clean import clean_point_cloud
from sklearn.neighbors import KDTree
from p2p_matching_in_organ.landmark_utils import get_mesh_landmarks
from utils import save_off, \
    point_cloud_to_mesh, \
    visualization_basis_function
# from icp import prepare_dataset, icp_init, execute_global_registration
from organ_matching.organ_matching_lr import match_organ_two_days
from skeleton_matching.skeleton_match import get_skeleton_landmarks_pairs
from p2p_matching_in_organ.p2p_matching_evaluation import evaluate_p2p_matching
from functional_map_.functional import FunctionalMapping
from visualize_p2p import visualize_p2p

pc_path = '../data/lyon2/processed/{}_segmented.ply'
skel_path = "../data/lyon2/{}_stem/skeleton_{}_connected.txt"
skel_noted_path = "../data/lyon2/{}_stem/skeleton_{}_noted.csv"
branch_connect_path = "../data/lyon2/{}_stem/branch_connection_{}.csv"
stem_node_path = "../data/lyon2/{}_stem/stem_nodes_ordered.csv"
mesh_radius_factor = 2

descriptor_weight = [1, 1, 1, 1]  # hks, wks, fpfh, xyz
plot_basis = True
plot_mesh = True
verbose = True
saveoff = True

pcd_clean_option = {'downsample_rate': 80,
                    'voxel_size': 1.2,
                    'crop': False,
                    'crop_low_bound': 0.0,
                    'crop_up_bound': 1}

correct_matching = {
    ("03-22_AM", "03-22_PM"): [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15, 16],
    ("03-22_PM", "03-23_PM"): [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15, 14, 20],
    ("03-22_PM", "03-23_AM"): [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
    ("03-21_PM", "03-22_AM"): [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15],
    ("03-21_AM", "03-21_PM"): [0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 14],
    ("03-20_AM", "03-20_PM"): [0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 11],
    ("03-20_PM", "03-21_AM"): [0, 1, 2, 3, 4, 5, 6, 7, 8],
}


def measure_organ(organ):

    pcd = organ["pcd"]
    segment_index = organ["segment_index"]
    skeleton_connect = organ["skeleton_connection"]
    skeleton_xyz = organ["skeleton_pcd"]

    measurement = {}
    segment_length = {}

    for seg in segment_index:
        skeleton_xyz_seg = skeleton_xyz[skeleton_xyz[:, 3] == seg][:, :3]
        l = 0.0
        for i in range(skeleton_xyz_seg.shape[0] - 1):
            l += np.linalg.norm(skeleton_xyz_seg[i + 1] - skeleton_xyz_seg[i])
        segment_length[seg] = l

    measurement["segment_length"] = segment_length
    measurement["connection"] = skeleton_connect
    # print(measurement)
    pcd_skel = open3d.geometry.PointCloud()
    pcd_skel.points = open3d.utility.Vector3dVector(skeleton_xyz[:, :3])
    pcd_skel.colors = open3d.utility.Vector3dVector([0.2, 0.97, 0.15] * np.ones((skeleton_xyz.shape[0], 3)))
    # open3d.visualization.draw_geometries([pcd_skel])

    return measurement


if __name__ == "__main__":
    day1 = "03-22_PM"
    day2 = "03-23_PM"
    t_start = time.time()
    matches_index, org_collection_1, org_collection_2 = \
        match_organ_two_days(day1, day2, visualize=False)
    # org_collection_1 = preprocess_pcd(org_collection_1)
    # org_collection_2 = preprocess_pcd(org_collection_2)
    t_end = time.time()
    print("Get the organs matched, used ", t_end - t_start, " s")
    measure_organ(org_collection_2[3])
    print("_")