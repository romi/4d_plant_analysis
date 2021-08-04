import os
import copy
import time
import json
import numpy as np
import sys
import imageio

sys.path.insert(0, "../")
import networkx as nx
import open3d
from sklearn.neighbors import KDTree


def load_pcd_registration(day1, day2, dataset, save_path_format):
    mesh_collection_1 = []
    mesh_collection_2 = []
    T21s = []
    T12s = []
    no_matched_pcd_collection_1 = []
    no_matched_pcd_collection_2 = []

    i = 0
    lp = []
    lc = []
    while i < 500:
        if not os.path.exists(save_path_format.format(dataset, day1, day2) + "{}_{}.csv".format(day1, i)):
            break
        mesh_collection_1.append(np.loadtxt(save_path_format.format(dataset, day1, day2) + "{}_{}.csv".format(day1, i)))
        mesh_collection_2.append(np.loadtxt(save_path_format.format(dataset, day1, day2) + "{}_{}.csv".format(day2, i)))
        if np.random.rand() > 0.44:
            lp.append(np.mean(mesh_collection_2[-1], axis=0))
            lp.append(np.mean(mesh_collection_1[-1], axis=0) + [0, 150, 0])
            lc.append([len(lp) - 2, len(lp) - 1])

        T21s.append(np.loadtxt(save_path_format.format(dataset, day1, day2) + "T21_{}.csv".format(i)))
        T12s.append(np.loadtxt(save_path_format.format(dataset, day1, day2) + "T12_{}.csv".format(i)))
        i += 1

    i = 0
    while i < 500:
        if not os.path.exists(save_path_format.format(dataset, day1, day2) + "{}_no_matched_{}.csv".format(day1, i)):
            break
        no_matched_pcd_collection_1.append(np.loadtxt(save_path_format.format(dataset, day1, day2) +
                                                      "{}_no_matched_{}.csv".format(day1, i)))
        i += 1
    i = 0
    while i < 500:
        if not os.path.exists(save_path_format.format(dataset, day1, day2) + "{}_no_matched_{}.csv".format(day2, i)):
            break
        no_matched_pcd_collection_2.append(np.loadtxt(save_path_format.format(dataset, day1, day2) +
                                                      "{}_no_matched_{}.csv".format(day2, i)))
        i += 1

    return mesh_collection_1, mesh_collection_2, T21s, T12s, no_matched_pcd_collection_1, no_matched_pcd_collection_2


def interpolation_plant_level(day1, day2, dataset, alpha):
    options_path = "../hyper_parameters/lyon2.json"
    with open(options_path, "r") as json_file:
        options = json.load(json_file)
    pcd_matching_path_format = options["p2p_save_path_segment"]
    P1_collection, P2_collection, T21_collection, T12_collection, Pn1_collection, Pn2_collection = \
        load_pcd_registration(day1, day2, dataset, pcd_matching_path_format)

    # print(len(P1_collection), len(P2_collection), len(T21_collection), len(Pn1_collection), len(Pn2_collection))

    P_inter_collection = []

    for i in range(len(P1_collection)):
        P_inter_collection.append(interpolation_segment_level(xyz1=P1_collection[i],
                                                              xyz2=P2_collection[i],
                                                              T21=T21_collection[i],
                                                              T12=T12_collection[i],
                                                              alpha=alpha))
    pcd_inter = open3d.geometry.PointCloud()

    xyz_inter = np.vstack(P_inter_collection)

    Pn = []
    if alpha < 0.3:
        for Pn1 in Pn1_collection:
            random_v = np.random.random((Pn1.shape[0], ))
            for xyz, r_v in zip(Pn1, random_v):
                if 0.3 * r_v > alpha:
                    Pn.append(xyz)
    xyz_1_collection = np.vstack(P1_collection)
    tree_xyz_inter = KDTree(xyz_1_collection)
    if alpha > 0.0:
        for Pn2 in Pn2_collection:
            xyz_NN = np.zeros(3)
            distance_NN = 1000
            for xyz_ in Pn2:
                distance, indices = tree_xyz_inter.query(xyz_.reshape(1, -1))
                if distance[0, 0] < distance_NN:
                    NN_index = indices[0, 0]
                    xyz_NN = xyz_1_collection[NN_index]

            random_v = np.random.random((Pn2.shape[0],))
            for xyz, r_v in zip(Pn2, random_v):
                xyz_n_inter = 1 * ((alpha - 0.0) * xyz + (1 - alpha) * xyz_NN)
                Pn.append(xyz_n_inter)

    if len(Pn) > 0:
        xyz_n = np.vstack(Pn)
        xyz_inter = np.vstack([xyz_inter, xyz_n])

    pcd_inter.points = open3d.utility.Vector3dVector(xyz_inter)
    pcd_inter = pcd_inter.voxel_down_sample(1.0)
    xyz_inter = np.asarray(pcd_inter.points)
    color = (np.max(xyz_inter, axis=0) - xyz_inter) / (np.max(xyz_inter, axis=0) - np.min(xyz_inter, axis=0))
    color = [0.3, 0.9, 0.1] * np.ones(xyz_inter.shape)
    # color = (1.4 * color ) % 1
    # color = color[:, [2, 2, 2]]
    pcd_inter.points = open3d.utility.Vector3dVector(xyz_inter)
    pcd_inter.colors = open3d.utility.Vector3dVector(color)
    pcd_inter.estimate_normals()
    return pcd_inter


def interpolation_segment_level(xyz1, xyz2, T21, T12, alpha):
    pcd1 = open3d.geometry.PointCloud()
    pcd1.points = open3d.utility.Vector3dVector(xyz1)
    pcd2 = open3d.geometry.PointCloud()
    pcd2.points = open3d.utility.Vector3dVector(xyz2)

    xyz2_origin = xyz1[T21.astype(int)]
    xyz_inter_0 = (1 - alpha) * xyz2_origin + alpha * xyz2

    xyz1_future = xyz2[T12.astype(int)]
    xyz_inter_1 = (1 - alpha) * xyz1 + alpha * xyz1_future

    xyz_inter = np.vstack([xyz_inter_0, xyz_inter_1])
    return xyz_inter


def produce_video(path):
    filenames = []

    for root, dirs, files in os.walk(path):
        files.sort()
        print(files)
        for file in files:
            # append the file name to the list
            filenames.append(os.path.join(root, file))

    with imageio.get_writer('./result/interpolation_lyon_may_2.mp4', mode='I') as writer:
        for filename in filenames:
            image = imageio.imread(filename)
            writer.append_data(image)
    return


if __name__ == "__main__":
    # day1 = "03-22_PM"
    # day2 = "03-23_PM"
    # dataset = "lyon2"

    day1 = "03-19_AM"
    day2 = "03-20_AM"
    dataset = "maize_1"

    X = []
    i = 1

    for alpha in [0.5 * j for j in range(1, 3)]:
        pcd_inter = interpolation_plant_level(day1, day2, dataset, alpha)
        open3d.visualization.draw_geometries([pcd_inter])
        i += 1
