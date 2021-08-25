import numpy as np
import open3d
import copy

from organ_matching.organ_matching_lr import match_organ_two_days, get_precision
from p2p_matching_in_organ.p2p_matching_mesure import measure_organ
from utils import point_cloud_to_mesh, saveColors
from sklearn.neighbors import KDTree
import os
import colorsys
from visualize_p2p import get_p2p_mapping_for_two_day


def load_p2p_matching_series(days, dataset, path_format):
    if len(days) < 2:
        print("At least two days are needed")
        return

    mapping_collector = {}
    for i in range(len(days) - 1):
        xyz_1, xyz_2, T21 = get_p2p_mapping_for_two_day(days[i], days[i + 1], dataset, path_format=path_format)
        mapping_collector[(days[i], days[i + 1])] = {
            0: xyz_1,
            1: xyz_2,
            2: T21
        }

    pcd_collector = {}
    for i in range(len(days)):
        if i > 0:
            xyz_fixed = mapping_collector[(days[i - 1], days[i])][1]
        else:
            xyz_fixed = mapping_collector[(days[i], days[i + 1])][0]
        p_ = open3d.geometry.PointCloud()
        p_.points = open3d.utility.Vector3dVector(xyz_fixed)
        # p_.estimate_normals()
        pcd_collector[days[i]] = p_
    return pcd_collector, mapping_collector


def find_correspondent_point(xyz, mapping_collector, distance_threshold=10):
    if xyz.shape[0] == 0:
        return np.array([])
    xyz_1 = mapping_collector[0]
    xyz_2 = mapping_collector[1]
    T21 = mapping_collector[2]
    tree = KDTree(xyz_2)

    distance, indices = tree.query(xyz.reshape(1, -1), return_distance=True, k=1)
    distance_to_xyz2 = distance[0, 0]
    NN_in_xyz2 = indices[0, 0]
    if T21[NN_in_xyz2] < 0 or distance_to_xyz2 > distance_threshold:
        return np.array([])
    return xyz_1[int(T21[NN_in_xyz2])]


def get_skeleton_by_day(day, dataset):
    skel_noted_path = "../data/{}/{}/skeleton_{}_noted.csv"
    segment_connect_path = "../data/{}/{}/branch_connection_{}.csv"
    xyz_skeleton_labeled = np.loadtxt(skel_noted_path.format(dataset, day, day))
    segment_connect = np.loadtxt(segment_connect_path.format(dataset, day, day))
    return xyz_skeleton_labeled


def get_segment_skeleton(xyz_skeleton_labeled, segment_label):
    assert segment_label in xyz_skeleton_labeled[:, 3]
    return xyz_skeleton_labeled[xyz_skeleton_labeled[:, 3] == segment_label][:, :3]


def track_one_point(xyz, pcd_collector, mapping_collector, visualize=False):
    days = list(pcd_collector.keys())
    assert len(days) >= 2
    track_point = []

    for i in range(len(days) - 1):
        day1 = days[-i-2]
        day2 = days[-i-1]
        track_point.append(copy.deepcopy(xyz))
        xyz = find_correspondent_point(xyz,
                                       mapping_collector[day1, day2])
    track_point.append(xyz)

    if visualize:
        track_point_copy = copy.deepcopy(track_point)
        pcds = list(pcd_collector.values())
        for i, p_ in enumerate(pcds):
            p_.translate([0, (100 + 10 * i) * i, 0])
            p_.colors = open3d.utility.Vector3dVector([0.8, 0.8, 0.8] * np.ones(np.asarray(p_.points).shape))
        track_point_balls = []
        for i in range(len(track_point)-1, -1, -1):
            sph = open3d.geometry.TriangleMesh.create_sphere(radius=3.0)
            if track_point_copy[i] is not None:
                track_point_copy[i] = track_point_copy[i] + np.array([0,
                                                                      (100 + 10 * (len(track_point) - 1 - i)) *
                                                                      (len(track_point) - 1 - i), 0])
                sph.translate(track_point_copy[i])
                sph.vertex_colors = open3d.utility.Vector3dVector([0.9, 0.2, 0.1] * np.ones(np.asarray(sph.vertices).shape))
                track_point_balls.append(sph)
        track_point_copy = [p for p in track_point_copy if p is not None]
        ls = open3d.geometry.LineSet()
        ls.points = open3d.utility.Vector3dVector(np.vstack(track_point_copy))
        ls.lines = open3d.utility.Vector2iVector([i, i+1] for i in range(len(track_point_copy) - 1))
        open3d.visualization.draw_geometries(pcds + track_point_balls + [ls])
    return track_point


def track_segment(dataset, day, segment_label, pcd_collector, mapping_collector):
    xyz_skeleton = get_skeleton_by_day(day, dataset)
    xyz_segment_skeleton = get_segment_skeleton(xyz_skeleton, segment_label)
    xyz_segment_skeleton = xyz_segment_skeleton[ [0, -1], :]
    track_points_collector = []
    p_skel = open3d.geometry.PointCloud()
    p_skel.points = open3d.utility.Vector3dVector(xyz_segment_skeleton)
    open3d.visualization.draw_geometries([p_skel])
    track_one_point(xyz_segment_skeleton[0], pcd_collector, mapping_collector, visualize=True)
    for xyz in xyz_segment_skeleton:
        track_points_collector.append(track_one_point(xyz, pcd_collector, mapping_collector, visualize=False))

    segment_skel_t_collector = {}
    for i in range(len(days)):
        segment_skel_t = []
        for j in range(len(track_points_collector)):
            segment_skel_t.append(track_points_collector[j][i])
        segment_skel_t_collector[days[-i - 1]] = copy.deepcopy(segment_skel_t)

    for d in segment_skel_t_collector:
        print(d)
        if segment_skel_t_collector[d][1] is not None and segment_skel_t_collector[d][0] is not None:
            print(np.linalg.norm(segment_skel_t_collector[d][1] - segment_skel_t_collector[d][0]))
    return segment_skel_t_collector


def track_point_in_time_series(dataset, days, xyz, visualize=False):
    match_path_format = "../data/{}/registration_result/{}_to_{}/"
    pcd_collector, mapping_collector = load_p2p_matching_series(days, dataset, match_path_format)
    track = np.vstack(track_one_point(xyz, pcd_collector, mapping_collector, visualize=visualize))
    return track

if __name__ == "__main__":
    import time
    dataset = "lyon2"
    match_path_format = "../data/{}/registration_result/{}_to_{}/"
    # days = ["03-20_AM", "03-20_PM", "03-21_AM", "03-21_PM", "03-22_AM", "03-22_PM", "03-23_PM"]
    days = ["03-22_AM", "03-22_PM", "03-23_PM"]
    # days = ["05-18_AM", "05-18_PM", "05-19_AM", "05-19_PM", "05-20_AM", "05-20_PM"]
    pcd_collector, mapping_collector = load_p2p_matching_series(days, dataset, match_path_format)

    xyz = np.array([0.0, 0.0, -20.0])

    t_start = time.time()
    print(track_one_point(xyz, pcd_collector, mapping_collector, visualize=True))
    t_end = time.time()
    print(t_end - t_start)

    # open3d.visualization.draw_geometries(list(pcd_collector.values()))
