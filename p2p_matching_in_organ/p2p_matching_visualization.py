import os
import copy
import time
import numpy as np
import pandas as pd
import sys
sys.path.insert(0, "../")
import networkx as nx
import open3d
from sklearn.neighbors import KDTree


save_path_format = "../data/{}/registration_result/{}_to_{}/"


def visualize_p2p_matching(day1,
                           day2,
                           dataset,
                           save_path_format="../data/{}/registration_result/{}_to_{}/",
                           show_all=False,
                           show_lines=False,
                           show_separate=False):
    mesh_collection_1 = []
    mesh_collection_2 = []
    p2p_mappings = []
    no_matched_pcd_collection_1 = []
    no_matched_pcd_collection_2 = []

    semantic_collection_2 = list(pd.read_csv(save_path_format.format(dataset, day1, day2) + "semantic_2.csv")['0'])
    semantic_collection_1 = list(pd.read_csv(save_path_format.format(dataset, day1, day2) + "semantic_1.csv")['0'])

    i = 0
    lp = []
    lc = []
    while i < 200:
        if not os.path.exists(save_path_format.format(dataset, day1, day2) + "{}_{}.csv".format(day1, i)):
            i += 1
            continue
        mesh_collection_1.append(np.loadtxt(save_path_format.format(dataset, day1, day2) + "{}_{}.csv".format(day1, i)))
        mesh_collection_2.append(np.loadtxt(save_path_format.format(dataset, day1, day2) + "{}_{}.csv".format(day2, i)))
        if np.random.rand() > 0.01:
            lp.append(np.mean(mesh_collection_2[-1], axis=0) + [20 * (i-24) * int(show_separate) , 0, 0])
            lp.append(np.mean(mesh_collection_1[-1], axis=0) + [20 * (i-24) * int(show_separate) , 100, 0])
            lc.append([len(lp)-2, len(lp)-1])

        p2p_mappings.append(np.loadtxt(save_path_format.format(dataset, day1, day2) + "T21_{}.csv".format(i)))
        i += 1

    i = 0
    while i < 0:
        if not os.path.exists(save_path_format.format(dataset, day1, day2) + "{}_no_matched_{}.csv".format(day1, i)):
            break
        no_matched_pcd_collection_1.append(np.loadtxt(save_path_format.format(dataset, day1, day2) +
                                                      "{}_no_matched_{}.csv".format(day1, i)))
        i += 1

    i = 0
    while i < 0:
        if not os.path.exists(save_path_format.format(dataset, day1, day2) + "{}_no_matched_{}.csv".format(day2, i)):
            break
        no_matched_pcd_collection_2.append(np.loadtxt(save_path_format.format(dataset, day1, day2) +
                                                      "{}_no_matched_{}.csv".format(day2, i)))
        i += 1

    color_bound_box = 10 * np.array([[-100.0, -100.0, -100.0], [100.0, 100.0, 100.0]])
    for i in range(len(mesh_collection_2)):
        P = mesh_collection_2[i]
        P_min = np.min(P, axis=0)
        P_max = np.max(P, axis=0)
        color_bound_box[0, :] = np.max(np.vstack([color_bound_box[0, :], P_max]), axis=0)
        color_bound_box[1, :] = np.min(np.vstack([color_bound_box[1, :], P_min]), axis=0)

    # print(color_bound_box)
    pcd1 = []
    pcd2 = []

    for index, m2 in enumerate(mesh_collection_2):
        P = m2
        color_value = np.ones(P.shape)
        # print(color_bound_box)
        X = (P - color_bound_box[0, :]) / (color_bound_box[1, :] - color_bound_box[0, :])
        X = X[:, [2, 1, 0]]
        # color_value = ([1.15, 1.45, 2.9] * (np.mean(X, axis=0) * color_value)) % 1
        color_value = X
        semantic_type = semantic_collection_2[index]
        if semantic_type == "leaf":
            c = [0.23, 0.87, 0.12]
        elif semantic_type == "branch":
            c = [0.86, 0.1, 0.56]
        elif semantic_type == "top":
            c = [0.78, 0.98, 0.2]
        elif semantic_type == "main_stem":
            c = [0.78, 0.1, 0.05]
        else:
            c = [0.9, 0.7, 0.1]

        # color_value = np.vstack([np.array(c) for i in range(X.shape[0])])
        p1 = open3d.geometry.PointCloud()
        p1.points = open3d.utility.Vector3dVector(m2)
        p1.colors = open3d.utility.Vector3dVector(copy.deepcopy(color_value))
        #p1.estimate_normals()
        if show_separate:
            p1.translate([20 * index, 0, 0])
        pcd1.append(p1)

        m1 = mesh_collection_1[index]
        color_value_1 = 0.8 * np.ones((m1.shape[0], 3))
        p2p = p2p_mappings[index].astype(int)
        color_value_1[p2p] = color_value
        points_mapped = np.unique(p2p)
        tree = KDTree(m1[points_mapped])
        all_mapped = tree.query(m1, k=1, return_distance=False).flatten()
        if show_all:
            color_value_1 = color_value_1[points_mapped[all_mapped]]
        p2 = open3d.geometry.PointCloud()
        p2.points = open3d.utility.Vector3dVector(m1)
        p2.colors = open3d.utility.Vector3dVector(copy.deepcopy(color_value_1))
        #p2.estimate_normals()
        p2.translate([20 * index * int(show_separate), 100, 0])
        pcd2.append(p2)

    for m1 in no_matched_pcd_collection_1:
        # p1.translate([0, 200, 0])
        p1 = open3d.geometry.PointCloud()
        p1.points = open3d.utility.Vector3dVector(m1)
        p1.colors = open3d.utility.Vector3dVector(0.8 * np.ones((m1.shape[0], 3)))
        #p1.estimate_normals()
        p1.translate([0, 100, 0])
        pcd1.append(p1)

    for index, m2 in enumerate(no_matched_pcd_collection_2):
        if index >= len(no_matched_pcd_collection_2) :
            break
        p2 = open3d.geometry.PointCloud()
        p2.points = open3d.utility.Vector3dVector(m2)
        p2.colors = open3d.utility.Vector3dVector(0.8 * np.ones((m2.shape[0], 3)))
        #p2.estimate_normals()
        pcd2.append(p2)

    ls = open3d.geometry.LineSet()
    ls.points = open3d.utility.Vector3dVector(np.vstack(lp))
    ls.lines = open3d.utility.Vector2iVector(np.vstack(lc).astype(int))
    if show_lines:
        open3d.visualization.draw_geometries(pcd1 + pcd2 + [ls])
    else:
        open3d.visualization.draw_geometries(pcd1 + pcd2)


if __name__ == "__main__":
    day1 = "03-22_PM"
    day2 = "03-23_PM"
    # day1 = "05-06_AM"
    # day2 = "05-07_AM"
    visualize_p2p_matching(day1, day2, "lyon2", show_all=True, show_lines=True, show_separate=False)