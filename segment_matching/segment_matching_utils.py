import sys
import copy
import numpy as np
import open3d
from sklearn.neighbors import KDTree
import networkx as nx


def get_segments(organ, istop=False):

    pcd = organ["pcd"]
    segment_index = organ["segment_index"]
    xyz_skeleton = organ["skeleton_pcd"]
    xyz_center = np.mean(xyz_skeleton[:, :3], axis=0)
    segment_connection = organ['skeleton_connection']
    graph = nx.Graph()
    graph.add_edges_from(segment_connection)

    if istop:
        segment_collection = {}
        j = 1000
        seg = {"pcd": pcd, "skeleton": xyz_skeleton, "point_number_proportion": 1, 'geodesic_distance': 0, 'degree': 1,
               "organ_center_offset": xyz_center, "semantic_meaning": "top"}
        seg["skeleton"] = seg["skeleton"][:, :3]
        segment_collection[j] = seg
        return segment_collection

    if organ["stem_connect_point"] is None:
        # root based
        segment_collection = {}

        if len(segment_index) == 1:
            j = segment_index[0]
            seg = {"pcd": pcd, "skeleton": xyz_center.reshape(1, -1), "point_number_proportion": 1, 'geodesic_distance': 0,
                   'degree': 0, "organ_center_offset": xyz_center, "semantic_meaning": "root"}
            segment_collection[j] = seg
            return segment_collection
        else:
            stem = segment_index[0]
            if np.unique(segment_connection).shape[0] == 1:
                stem = np.unique(segment_connection)[0]
            else:
                for n in np.unique(segment_connection):
                    if n not in segment_index:
                        stem = n

            seg_skeleton = xyz_skeleton[:, 3]
            xyz_skeleton = xyz_skeleton[:, :3]

            tree = KDTree(xyz_skeleton)
            xyz_pcd = np.asarray(pcd.points)

            nn = tree.query(xyz_pcd, k=1, return_distance=False).reshape(-1)
            seg_pcd = seg_skeleton[nn]

            segment_collection = {}

            graph = nx.Graph()
            graph.add_edges_from(organ['skeleton_connection'])

            for seg_i in segment_index:
                seg = {}
                p_ = open3d.geometry.PointCloud()
                p_.points = open3d.utility.Vector3dVector(xyz_pcd[seg_pcd == seg_i])
                seg["pcd"] = p_
                seg["skeleton"] = xyz_skeleton[seg_skeleton == seg_i]
                seg["point_number_proportion"] = np.sum(seg_pcd == seg_i) / len(seg_pcd)
                seg['geodesic_distance'] = nx.shortest_path_length(graph, source=stem, target=seg_i)
                seg['degree'] = graph.degree[seg_i]
                seg["semantic_meaning"] = "branch" if seg['degree'] > 1 else "leaf"
                seg["organ_center_offset"] = xyz_center
                if seg["skeleton"].shape[0] >= 2 and np.asarray(seg["pcd"].points).shape[0] > 50:
                    segment_collection[seg_i] = copy.deepcopy(seg)
            return segment_collection

    if len(segment_connection) == 0:
        # The main stem
        xyz_skeleton = xyz_skeleton[xyz_skeleton[:, 2].argsort()]
        xyz_pcd = np.asarray(organ["pcd"].points)
        segment_collection = {}

        stem_connect_point = organ["stem_connect_point"]
        j = -2

        seg = {}
        p_ = open3d.geometry.PointCloud()
        p_.points = open3d.utility.Vector3dVector(xyz_pcd[xyz_pcd[:, 2] < stem_connect_point[0, 2]])
        seg["pcd"] = p_
        seg["skeleton"] = xyz_skeleton[xyz_skeleton[:, 2] < stem_connect_point[0, 2]]
        seg["skeleton"] = seg["skeleton"][:, :3]
        seg["organ_center_offset"] = xyz_center
        seg["semantic_meaning"] = "main_stem"

        if seg["skeleton"].shape[0] >= 1 and np.asarray(seg["pcd"].points).shape[0] > 30:
            seg["point_number_proportion"] = len(np.asarray(p_.points)) / xyz_pcd.shape[0]
            seg['geodesic_distance'] = j
            seg['degree'] = 2
            segment_collection[j] = seg
            j -= 1

        for i in range(len(stem_connect_point) - 1):

            seg = {}
            p_ = open3d.geometry.PointCloud()
            p_.points = open3d.utility.Vector3dVector( xyz_pcd[np.logical_and(stem_connect_point[i, 2] <= xyz_pcd[:, 2],
                                                          xyz_pcd[:, 2] < stem_connect_point[i+1, 2])])
            seg["pcd"] = p_
            seg["skeleton"] = xyz_skeleton[np.logical_and(stem_connect_point[i, 2] <= xyz_skeleton[:, 2],
                                                          xyz_skeleton[:, 2] < stem_connect_point[i+1, 2])]
            seg["skeleton"] = seg["skeleton"][:, :3]
            seg["organ_center_offset"] = xyz_center
            seg["semantic_meaning"] = "main_stem"
            if seg["skeleton"].shape[0] >= 1 and np.asarray(seg["pcd"].points).shape[0] > 30:
                seg["point_number_proportion"] = len(np.asarray(p_.points)) / xyz_pcd.shape[0]
                seg['geodesic_distance'] = j
                seg['degree'] = 2
                segment_collection[j] = seg
                j -= 1
        return segment_collection

    # Find the stem index
    stem = segment_index[0]
    if np.unique(segment_connection).shape[0] == 1:
        stem = np.unique(segment_connection)[0]
    else:
        for n in np.unique(segment_connection):
            if n not in segment_index:
                stem = n

    seg_skeleton = xyz_skeleton[:, 3]
    xyz_skeleton = xyz_skeleton[:, :3]

    tree = KDTree(xyz_skeleton)
    xyz_pcd = np.asarray(pcd.points)

    nn = tree.query(xyz_pcd, k=1, return_distance=False).reshape(-1)
    seg_pcd = seg_skeleton[nn]

    segment_collection = {}

    graph = nx.Graph()
    graph.add_edges_from(organ['skeleton_connection'])

    for seg_i in segment_index:
        seg = {}
        p_ = open3d.geometry.PointCloud()
        p_.points = open3d.utility.Vector3dVector( xyz_pcd[seg_pcd == seg_i] )
        seg["pcd"] = p_
        seg["skeleton"] = xyz_skeleton[seg_skeleton == seg_i]
        seg["point_number_proportion"] = np.sum(seg_pcd == seg_i) / len(seg_pcd)
        seg['geodesic_distance'] = nx.shortest_path_length(graph, source=stem, target=seg_i)
        seg['degree'] = graph.degree[seg_i]
        seg["semantic_meaning"] = "branch" if seg['degree'] > 1 else "leaf"
        seg["organ_center_offset"] = xyz_center
        if seg["skeleton"].shape[0] >= 1 and np.asarray(seg["pcd"].points).shape[0] > 30:
            segment_collection[seg_i] = copy.deepcopy(seg)
    return segment_collection