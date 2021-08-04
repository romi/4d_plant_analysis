import numpy as np
import open3d
from utils import point_cloud_to_mesh, saveColors
from sklearn.neighbors import KDTree
import os
import colorsys

pc_path = './data/lyon2/processed/{}_segmented.ply'

pcd_clean_option = {'downsample_rate': 80,
                    'voxel_size': 0.7,
                    'crop': False,
                    'crop_low_bound': 0.0,
                    'crop_up_bound': 1}


def visualize_p2p(p2p, mesh1, mesh2, color_value_2, translate_distance=1.0, show_all=True, saveoff=False):
    color_value_1 = 0.8 * np.ones((np.asarray(mesh1.vertices).shape[0], 3))
    # for i in range(color_value_1.shape[0]):
    #     color_value_1[i, :] = color_value_2[p2p[i], :]
    color_value_1[p2p] = color_value_2
    points_mapped = np.unique(p2p)
    tree = KDTree(np.asarray(mesh1.vertices)[points_mapped])
    all_mapped = tree.query(np.asarray(mesh1.vertices), k=1, return_distance=False).flatten()
    if show_all:
        color_value_1 = color_value_1[points_mapped[all_mapped]]
    mesh1.vertex_colors = open3d.utility.Vector3dVector(color_value_1)
    open3d.visualization.draw_geometries([mesh1, mesh2.translate((0.0, translate_distance, 0.0))])
    if saveoff:
        saveColors("data/match_to.off", mesh1.vertlist, color_value_1[:, :3], mesh1.trilist)
        saveColors("data/match_from.off", mesh2.vertlist, color_value_2[:, :3], mesh2.trilist)


def get_p2p_mapping_for_two_day(day1, day2, dataset, path_format):
    mesh_collection_1 = []
    mesh_collection_2 = []
    p2p_mappings = []
    no_matched_pcd_collection_1 = []
    no_matched_pcd_collection_2 = []

    i = 0
    N = 0
    while i < 200:
        if not os.path.exists(path_format.format(dataset, day1, day2) + "{}_{}.csv".format(day1, i)):
            i += 1
            continue
        mesh_collection_1.append(np.loadtxt(path_format.format(dataset, day1, day2) + "{}_{}.csv".format(day1, i)))
        mesh_collection_2.append(np.loadtxt(path_format.format(dataset, day1, day2) + "{}_{}.csv".format(day2, i)))

        p2p_map = np.loadtxt(path_format.format(dataset, day1, day2) + "T21_{}.csv".format(i))
        p2p_mappings.append(p2p_map + N)
        N += mesh_collection_1[-1].shape[0]
        i += 1

    i = 0
    while i < 500:
        if not os.path.exists(path_format.format(dataset, day1, day2) + "{}_no_matched_{}.csv".format(day1, i)):
            break
        m1 = np.loadtxt(path_format.format(dataset, day1, day2) + "{}_no_matched_{}.csv".format(day1, i))
        mesh_collection_1.append(m1)
        i += 1
    i = 0
    while i < 500:
        if not os.path.exists(path_format.format(dataset, day1, day2) + "{}_no_matched_{}.csv".format(day2, i)):
            break
        m2 = np.loadtxt(path_format.format(dataset, day1, day2) + "{}_no_matched_{}.csv".format(day2, i))
        mesh_collection_2.append(m2)
        p2p_mappings.append(-1 * np.ones(mesh_collection_2[-1].shape[0]))
        i += 1

    xyz_1 = np.vstack(mesh_collection_1)
    xyz_2 = np.vstack(mesh_collection_2)
    mapping_21 = np.hstack(p2p_mappings)
    return xyz_1, xyz_2, mapping_21


def color_using_matching(pcd1, pcd2, p2p_mapping):
    color_value_2 = np.asarray(pcd2.colors)
    color_value_1 = 0.8 * np.ones((np.asarray(pcd1.points).shape[0], 3))

    xyz_2 = np.asarray(pcd2.points)
    tree = KDTree(p2p_mapping[1])
    distance, indices = tree.query(xyz_2, return_distance=True, k=1)
    T21 = p2p_mapping[2][indices.reshape(-1)].astype(int)

    T21_ = T21[T21 > 0]

    xyz_1 = np.asarray(pcd1.points)
    tree = KDTree(xyz_1)
    distance, indices = tree.query(p2p_mapping[0], return_distance=True, k=3)

    T21_ = indices[:, 0][T21_]

    color_value_1[T21_.astype(int)] = color_value_2[T21 > 0]

    points_mapped = np.unique(T21_)
    tree = KDTree(np.asarray(pcd1.points)[points_mapped])
    distance, all_mapped = tree.query(np.asarray(pcd1.points), k=5, return_distance=True)

    for i in range(color_value_1.shape[0]):
        if distance[i, 0] <= 10 ** -5:
            color_value_1[i, :] = color_value_1[points_mapped[all_mapped[i, 0]]]
            continue
        weights = 1 / distance[i]
        color_value_1[i, :] = np.average(color_value_1[points_mapped[all_mapped[i]]], weights=weights, axis=0)
    pcd1.colors = open3d.utility.Vector3dVector(color_value_1)

    return


def visualize_pcd_registration_series(days, dataset, path_format):
    if len(days) < 2:
        print("At least two days are needed")
        return

    pcd = {}

    mapping_two_day_collector = {}
    for i in range(len(days) - 1):
        xyz_1, xyz_2, T21 = get_p2p_mapping_for_two_day(days[i], days[i + 1], dataset, path_format=path_format)
        mapping_two_day_collector[(days[i], days[i + 1])] = {
            0: xyz_1,
            1: xyz_2,
            2: T21
        }

    for i in range(len(days)):
        if i > 0:
            xyz_fixed = mapping_two_day_collector[(days[i - 1], days[i])][1]
        else:
            xyz_fixed = mapping_two_day_collector[(days[i], days[i + 1])][0]
        p_ = open3d.geometry.PointCloud()
        p_.points = open3d.utility.Vector3dVector(xyz_fixed)
        # p_.estimate_normals()
        pcd[days[i]] = p_

    last_day = days[-1]
    xyz = np.asarray(pcd[last_day].points)
    color_bound_box = np.array([[57.99211873, -2.02318142, 8.03139783],
                                [-0.86733103, -59.91601056, -45.19114869]])
    color = (np.max(xyz, axis=0) - xyz) / (np.max(xyz, axis=0) - np.min(xyz, axis=0))
    # color = (xyz - color_bound_box[0, :]) / (color_bound_box[1, :] - color_bound_box[0, :])
    # color = [0.3, 0.9, 0.1] * color
    color = (1.2 * color)

    # color = np.mean(xyz, axis=0) % 1
    color = np.abs((color - 1)%2- 1)
    color = color[:, [2, 1, 0]]
    pcd[last_day].colors = open3d.utility.Vector3dVector(color)

    for i in range(len(days) - 1, 0, -1):
        pcd2 = pcd[days[i]]
        pcd1 = pcd[days[i - 1]]
        color_using_matching(pcd1, pcd2, mapping_two_day_collector[(days[i - 1], days[i])])

    # for d in days:
    #     open3d.visualization.draw_geometries([pcd[d]])

    translate_distance = 0
    ls = open3d.geometry.LineSet()

    ls_p = []
    ls_connect = []
    ls_color = []

    color_set = [np.array(list(colorsys.hsv_to_rgb(color_index / 5,
                                                   0.9, 1))) for color_index in range(len(days) - 1)]

    for i in range(len(days) - 1):
        pcd1 = pcd[days[i]]
        pcd2 = pcd[days[i + 1]]
        translate_distance += 120 + (10 * i)
        pcd2.translate([0, translate_distance, 0])

        for j in range(22):
            xyz_2 = mapping_two_day_collector[(days[i], days[i + 1])][1]
            xyz_1 = mapping_two_day_collector[(days[i], days[i + 1])][0]
            T21 = mapping_two_day_collector[(days[i], days[i + 1])][2]

            random_index = int((j+1) * xyz_2.shape[0] / 23)
            if T21[random_index] < 0:
                continue
            ls_p.append(xyz_2[random_index] + [0, translate_distance, 0])
            ls_p.append(xyz_1[int(T21[random_index])] + [0, translate_distance - (120 + (10 * i)), 0])
            ls_connect.append([len(ls_p) - 2, len(ls_p) - 1])
            # ls_color.append(color_set[i])
            ls_color.append( np.asarray(pcd2.colors)[random_index] )

    ls.points = open3d.utility.Vector3dVector(np.vstack(ls_p))
    ls.lines = open3d.utility.Vector2iVector(ls_connect)
    ls.colors = open3d.utility.Vector3dVector(ls_color)
    open3d.visualization.draw_geometries(list(pcd.values()) + [ls])

    return None


if __name__ == "__main__":
    # days = ["03-20_AM", "03-20_PM", "03-21_AM", "03-21_PM", "03-22_AM", "03-22_PM", "03-23_PM"]
    days = ["03-22_PM", "03-23_PM"]
    # days = ["03-22_PM", "03-23_PM"]
    # days = ["05-06_AM", "05-07_AM", "05-08_AM", "05-09_AM", "05-11_AM", "05-12_AM", "05-13_AM" ]
    # days = ["03-15_AM", "03-16_AM", "03-17_AM", "03-18_AM", "03-19_AM", "03-20_AM", "03-21_AM"]
    # days = ["03-05_AM", "03-06_AM", "03-07_AM", "03-08_AM", "03-09_AM", "03-10_AM", "03-11_AM"]
    match_path_format = "data/{}/registration_result/{}_to_{}/"
    visualize_pcd_registration_series(days, "lyon2", path_format=match_path_format)
