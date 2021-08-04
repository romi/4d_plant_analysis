import colorsys
import copy
import numpy as np
import open3d
from skeletonization.point_cloud_segmentation import point_cloud_segmentation_from_skeleton
from utils import point_cloud_to_mesh
from point_cloud_clean import clean_point_cloud
from sklearn.neighbors import KDTree

import time


def organ_segmentation(branch_graph,
                       branch_center,
                       mesh,
                       label_mesh,
                       xyz_skel,
                       xyz_stem_node,
                       is_include_root=True,
                       option={},
                       visualize=False):
    """
    Cluster the segments of the plants into different organs: root, branch, main stem
    Each organ is made up with one or several segments defined in branch graph
    Then re-color the plant mesh according to the organ labels.
    :param is_include_root:
    :param xyz_stem_node:
    :param branch_graph: networkx Graph, define the connection between different segments of plant
    :param branch_center: numpy array of shape [N, 3], define the center positions of each segments
    :param mesh: open3d.geometry.trianglemesh, the original mesh of the plant, containing M points
    :param label_mesh: numpy array of shape [M, ], define the segment index each point in mesh belongs to
    :param xyz_skel: numpy array of shape [S, 3], define the position and segment index of the points in skeleton
    :param option: dictionary, fix the parameter used , default {}
    :param visualize: bool, if plot the result mesh with organ labels, default false

    :return: organ_dict: dictionary, define the segments' index belonging to each organ
                {
                    "root": ...,
                    "stem": ...,
                    "branches": [...],
                    ...
                }
            organ index mesh: numpy array of shape [M, ], define the index of organ each point in mesh belonging to
                0: stem, 1: root, 2: root branch, i+3: ith stem branch

    """
    organ_dict = {}
    root = np.argmin(branch_center[:, 2])
    if is_include_root:
        organ_dict["root"] = root
    else:
        organ_dict["root"] = -10
    stem = max(branch_graph.nodes, key=lambda n: len(list(branch_graph.neighbors(n))))
    organ_dict["stem"] = stem
    organ_dict["root_branch"] = []
    organs = []

    # Apply dfs to find the branches connected to main stem
    not_visited = set(list(branch_graph.nodes))
    # not_visited.remove(root)
    not_visited.remove(stem)

    branch_head = list(branch_graph.neighbors(stem))
    branch_head.sort(key=lambda n: branch_center[int(n), 2])
    for b in branch_head:
        if b == root:
            # find the branch connect to the root
            root_branch = []
            q = [int(root)]
            while len(q) > 0:
                current = q.pop()
                root_branch.append(int(current))
                not_visited.remove(current)
                for neigh in list(branch_graph.neighbors(current)):
                    if neigh in not_visited:
                        q.append(int(neigh))
            organ_dict["root_branch"] = root_branch
            continue
        head = b
        branch_ = []
        q = [int(head)]
        while len(q) > 0:
            current = q.pop()
            branch_.append(int(current))
            not_visited.remove(current)
            for neigh in list(branch_graph.neighbors(current)):
                if neigh in not_visited:
                    q.append(int(neigh))
        organs.append(copy.deepcopy(branch_))

    # if len(organ_dict["root_branch"]) > 0:
    #     organs.append(copy.deepcopy(organ_dict["root_branch"]))
    organs.sort(key=lambda l: np.mean(branch_center[l, 2]))  # sort the branches with its average center height

    xyz_mesh = np.asarray(mesh.vertices)

    # Merge lowest branches to root branch and remove the branches with too few skeleton points

    if "root_height_threshold" not in option:
        option["root_height_threshold"] = -1
    if "skel_point_threshold" not in option:
        option["skel_point_threshold"] = 5

    i = 0
    while i < len(organs):
        low_bound = 1000
        skel_point_count = 0
        br = organs[i]
        for seg in br:
            # Calculate the lower bound and skeleton point of each branch
            if seg not in label_mesh:
                continue
            low_bound = min( np.mean(xyz_mesh[label_mesh == seg][:, 2]), low_bound)
            skel_point_count += xyz_skel[xyz_skel[:, 3] ==seg].shape[0]

        if is_include_root and low_bound < branch_center[root, 2] + option["root_height_threshold"]:
            organ_dict["root_branch"] += br
            del organs[i]
            continue

        elif skel_point_count < option["skel_point_threshold"]:
            del organs[i]
            continue
        else:
            i += 1

    # Merge the branches with close connection points
    if "merge_dh_threshold" not in option:
        option["merge_dh_threshold"] = 8
    if "merge_height_limit" not in option:
        option["merge_height_limit"] = 0.8
    height_limit = np.min(branch_center[:, 2]) + \
                   option["merge_height_limit"] * (np.max(branch_center[:, 2])-np.min(branch_center[:, 2]))

    connect_height = list(np.zeros(len(organs)))
    organ_center_position = np.zeros((len(organs), 3))
    for i in range(len(organs)):
        organ_center_position[i, :] = np.mean(branch_center[organs[i]], axis=0)
        connect_seg_index = organs[i][0]
        connect_seg = xyz_skel[xyz_skel[:, 3] == connect_seg_index]
        if connect_seg.shape[0] > 0:
            connect_point = connect_seg[0, :3]
            connect_height[i] = connect_point[2]
    organ_center_position = organ_center_position - branch_center[int(stem)]

    i = 0
    while i < len(organs):
        if connect_height[i] > height_limit:
            break
        j = i + 1
        while j < len(organs):
            if np.abs(connect_height[i] - connect_height[j]) < option["merge_dh_threshold"] and \
                    np.sum(organ_center_position[i, :2] * organ_center_position[j, :2]) > 0.5:
                if len(organs[i]) <= len(organs[j]):
                    organs[i] = organs[j] + organs[i]
                else:
                    organs[i] = organs[i] + organs[j]
                del organs[j]
                del connect_height[j]
                organ_center_position = np.delete(organ_center_position, [j], axis=0)
            else:
                j += 1
        i += 1
    # print(organ_center_position)

    tree = KDTree(xyz_stem_node)
    connect_points = np.zeros((len(organs), 3))
    for i, br in enumerate(organs):
        seg = br[0]
        connect_point = xyz_skel[xyz_skel[:, 3] == seg][0, :3]
        connect_points[i, :] = connect_point
    matches = tree.query(connect_points, k=1, return_distance=False).flatten()

    organs = [x for _, x in sorted(zip(matches, organs))]

    i = 0
    while i < len(organs):
        if len(organs[i]) == 1 and connect_height[i] < height_limit:
            seg = xyz_skel[xyz_skel[:, 3] == organs[i][0]]
            direction = (seg[-1, :3] - seg[0, :3]) / np.linalg.norm(seg[-1, :3] - seg[0, :3])
            if abs(direction[2]) > 0.8:
                label_mesh[label_mesh == organs[i][0]] = stem
                del organs[i]
                del connect_height[i]
                organ_center_position = np.delete(organ_center_position, [i], axis=0)
            else:
                i += 1
                continue
        else:
            i += 1

    organ_dict["branches"] = copy.deepcopy(organs)
    # root branch height:
    h = -np.infty
    for br in organ_dict["root_branch"]:
        seg = xyz_skel[xyz_skel[:, 3] == br]
        for j in range(seg.shape[0]):
            if np.linalg.norm(seg[j, :2]) < 16:
                h = max(h, np.max(xyz_mesh[label_mesh == br][:, 2]))

    # for i in range(xyz_mesh.shape[0]):
    #     if xyz_mesh[i, 2] < h and label_mesh[i] == stem:
    #         label_mesh[i] = root

    # color the branches and roots
    branch_color = {}
    for br in range(branch_center.shape[0]):
        if int(stem) == br:
            branch_color[br] = np.array([0.79, 0.96, 0.48])
            continue
        if br == root:
            branch_color[br] = np.array([0.50, 0.23, 0.14])
            continue
        if br in organ_dict["root_branch"]:
            branch_color[br] = np.array([0.85, 0.43, 0.14])
            continue

        branch_count = len(organ_dict["branches"])
        branch_index = -1
        for i in range(branch_count):
            if br in organ_dict["branches"][i]:
                branch_color[br] = np.array(list(colorsys.hsv_to_rgb(0.0 + 0.1 * i,
                                                                     0.87 - i / 1000,
                                                                     0.96 - i / 1000)))
                # decided by the height, the colors of branch varies from yellow to green to blue to purple

                # branch_color[br] = np.random.rand(3)
                branch_index = i
                break
        if branch_index >= 0:
            continue
        else:
            # un colored branches
            branch_color[br] = np.array([0.95, 0.97, 0.95])

    organ_index_mesh = -1 * np.ones(xyz_mesh.shape[0])
    organ_color_values = 0.95 * np.ones((xyz_mesh.shape[0], 3))

    for i in range(xyz_mesh.shape[0]):
        organ_color_values[i, :] = branch_color[int(label_mesh[i])]
        l = int(label_mesh[i])
        if l == stem:
            organ_index_mesh[i] = 0
            continue
        if l == root:
            organ_index_mesh[i] = 1
            continue
        if l in organ_dict["root_branch"]:
            organ_index_mesh[i] = 2
            continue
        for j in range(len(organ_dict["branches"])):
            if l in organ_dict["branches"][j]:
                organ_index_mesh[i] = j + 3

    mesh.vertex_colors = open3d.utility.Vector3dVector(organ_color_values)

    if visualize:
        open3d.visualization.draw_geometries([mesh])

    return organ_dict, organ_index_mesh


def get_organ_pcd(mesh, branch_index, target_index, visualize=False):
    pcd = open3d.geometry.PointCloud()
    points = np.asarray(mesh.vertices)
    points_selected = points[branch_index == target_index]
    colors = np.asarray(mesh.vertex_colors)
    colors_selected = colors[branch_index == target_index]
    pcd.points = open3d.utility.Vector3dVector(points_selected)
    pcd.colors = open3d.utility.Vector3dVector(colors_selected)
    if visualize:
        open3d.visualization.draw_geometries([pcd])
    return pcd


def get_collection_organ_pcd(mesh, branch_index):
    organ_collection = []
    for i in range(int(np.max(branch_index)) + 1):
        organ_collection.append(get_organ_pcd(mesh, branch_index, i))
    return organ_collection


if __name__ == "__main__":
    days = ['03-23_PM']
    # days = ["03-20_AM", "03-20_PM", "03-21_AM", "03-21_PM", "03-22_AM", "03-22_PM", "03-23_AM",
    #         "03-23_PM"]

    pc_path = '../data/lyon2/processed/{}_segmented.ply'
    skel_path = "../data/lyon2/{}_stem/skeleton_{}_connected.txt"
    skel_noted_path = "../data/lyon2/{}_stem/skeleton_{}_noted.csv"
    branch_connect_path = "../data/lyon2/{}_stem/branch_connection_{}.csv"
    stem_node_path = "../data/lyon2/{}_stem/stem_nodes_ordered.csv"
    mesh_radius_factor = 3

    pcd_clean_option = {'downsample_rate': 80,
                        'voxel_size': 0.5,
                        'crop': False,
                        'crop_low_bound': 0.0,
                        'crop_up_bound': 1}
    options = {
        "03-20_AM": {
                "skel_point_threshold": 3
            },
        "03-20_PM": {
                "skel_point_threshold": 3
            },
        "03-21_AM": {
                "skel_point_threshold": 3
            },
        "03-21_PM": {
                "skel_point_threshold": 10
            },
        "03-22_AM": {
                "skel_point_threshold": 10
            },
        "03-22_PM": {
                "skel_point_threshold": 10
            },
        "03-23_AM": {
                "skel_point_threshold": 10
            },
        "03-23_PM": {
                "skel_point_threshold": 10,
                "merge_height_limit": 0.65,
            }
    }

    for i, day in enumerate(days):
        xyz_labeled = np.loadtxt(skel_noted_path.format(day, day))
        branch_connect = np.loadtxt(branch_connect_path.format(day, day))
        xyz_stem_node = np.loadtxt(stem_node_path.format(day))
        pcd = open3d.io.read_point_cloud(pc_path.format(day))
        pcd, description = clean_point_cloud(pcd, option=pcd_clean_option)
        voxel_size = description['voxel_size']
        mesh = point_cloud_to_mesh(pcd, radius_factor=mesh_radius_factor)

        gb, bc, bo, bsc, mesh, label_mesh = point_cloud_segmentation_from_skeleton(
            xyz_labeled,
            branch_connect,
            mesh,
            visualize=False,
            verbose=True)
        xyz_skel = xyz_labeled[:, :]

        option = options[day]

        organ_dict, b_i = organ_segmentation(gb, bc, mesh, label_mesh, xyz_skel, xyz_stem_node, option=option, visualize=True)
        print(organ_dict)

        for i in range(int(np.max(b_i)) + 1):
            p_organ = get_organ_pcd(mesh, b_i, i, visualize=False)
            # open3d.visualization.draw_geometries([p_organ])
