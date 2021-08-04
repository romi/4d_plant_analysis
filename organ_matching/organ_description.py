import copy
import numpy as np
import open3d
from sklearn.neighbors import KDTree
from point_cloud_clean import clean_point_cloud
from skeletonization.point_cloud_segmentation import point_cloud_segmentation_from_skeleton
from skeletonization.organ_segmentation import organ_segmentation, get_collection_organ_pcd
from utils import point_cloud_to_mesh

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


def get_organ_description(organ_dict, organ_collection, branch_graph, xyz_skel, xyz_stem_node, weight=None):
    if weight is None:
        weight = [1, 1, 1, 1, 1, 1, 1, 1, 0.01, 0.01, 0.01, 1, 1, 1]

    is_stem = np.zeros((len(organ_collection), 1))
    is_stem[0, 0] = 1

    is_root = np.zeros((len(organ_collection), 1))
    is_root[1, 0] = 1

    is_root_branch = np.zeros((len(organ_collection), 1))
    is_root_branch[2, 0] = 1

    is_top = np.zeros((len(organ_collection), 1))
    is_top[-1, 0] = 1

    height_order = np.zeros((len(organ_collection), 1))
    for i in range(len(organ_collection)):
        height_order[i, 0] = i / 10

    point_number = np.zeros((len(organ_collection), 1))
    for i in range(len(organ_collection)):
        point_number[i, 0] = np.asarray(organ_collection[i].points).shape[0] / 10 ** 4

    seg_number = np.zeros((len(organ_collection), 1))
    seg_number[0, 0] = 1  # stem
    seg_number[1, 0] = 1  # root
    seg_number[2, 0] = len(organ_dict["root_branch"])

    for i in range(3, len(organ_collection)):
        seg_number[i, 0] = len(organ_dict["branches"][i - 3])

    connect_stem_order = np.zeros((len(organ_collection), 1))
    connect_stem_order[0, 0] = 0
    connect_stem_order[1, 0] = 0
    connect_stem_order[2, 0] = 0

    tree = KDTree(xyz_stem_node)
    connect_points = np.zeros((len(organ_dict["branches"]), 3))
    for i, br in enumerate(organ_dict["branches"]):
        seg = br[0]
        connect_point = xyz_skel[xyz_skel[:, 3] == seg][0, :3]
        connect_points[i, :] = connect_point
    matches = tree.query(connect_points, k=1, return_distance=False).flatten()

    for i in range(len(matches)):
        connect_stem_order[i + 3, 0] = matches[i]/xyz_stem_node.shape[0]


    xyz_center = np.zeros((len(organ_collection), 3))
    for i in range(len(organ_collection)):
        xyz_center[i, :] = np.mean(np.asarray(organ_collection[i].points), axis=0)

    orientation_relative_stem = np.zeros((len(organ_collection), 3))
    orientation_relative_stem[0, :] = np.array([0, 0, 1])
    orientation_relative_stem[1, :] = np.array([0, 0, 1])
    orientation_relative_stem[2, :] = np.array([0, 0, 1])

    pcds = []
    for i, br in enumerate(organ_dict["branches"]):
        xyz_skel_br = []
        for seg in br:
            xyz_seg = xyz_skel[xyz_skel[:, 3] == seg][:, :3]
            xyz_skel_br.append(xyz_seg)
        xyz_skel_br = np.vstack(xyz_skel_br)
        # pcd = open3d.geometry.PointCloud()
        # pcd.points = open3d.utility.Vector3dVector(xyz_skel_br)
        # pcds.append(pcd)

        xyz_connect_point = xyz_stem_node[matches[i] - 5: matches[i] + 5, :]
        farthest_point = xyz_skel_br[0, :]
        for xyz_ in xyz_skel_br:
            if np.linalg.norm(xyz_ - xyz_connect_point) > np.linalg.norm(farthest_point - xyz_connect_point):
                farthest_point = xyz_

        local_stem_orientation = np.zeros(3)
        for j in range(5):
            if matches[i] + j < xyz_stem_node.shape[0]:
                local_stem_orientation += (xyz_stem_node[matches[i] + j] - xyz_stem_node[matches[i] - j])
        if np.linalg.norm(local_stem_orientation) < 10 ** -3:
            orientation_relative_stem[i + 3, :] = np.array([0, 0, 1.0])
            continue

        # Get the axis orientation of stem
        z_axis_ = local_stem_orientation / np.linalg.norm(local_stem_orientation)
        x_axis_ = np.array([z_axis_[2], 0, -z_axis_[0]])
        x_axis_ = x_axis_ / np.linalg.norm(x_axis_)
        y_axis_ = np.array([-z_axis_[0] * z_axis_[1], z_axis_[0] ** 2 + z_axis_[2] ** 2, -z_axis_[2] * z_axis_[1]])

        x_project = np.sum((farthest_point - xyz_stem_node[matches[i]]) * x_axis_)
        y_project = np.sum((farthest_point - xyz_stem_node[matches[i]]) * y_axis_)
        z_project = np.sum((farthest_point - xyz_stem_node[matches[i]]) * z_axis_)
        orientation_relative_stem[i + 3, 0] = x_project / np.linalg.norm([x_project, y_project, z_project])
        orientation_relative_stem[i + 3, 1] = y_project / np.linalg.norm([x_project, y_project, z_project])
        orientation_relative_stem[i + 3, 2] = z_project / np.linalg.norm([x_project, y_project, z_project])

    stem_connect = np.zeros((len(organ_collection), 3))
    for i in range(3, len(organ_collection)):
        stem_connect[i, :] = xyz_stem_node[matches[i-3]]

    res = np.hstack([is_stem,
                     is_root,
                     is_root_branch,
                     is_top,
                     height_order,
                     point_number,
                     seg_number,
                     connect_stem_order,
                     xyz_center,
                     orientation_relative_stem])
    return res * weight, stem_connect


if __name__ == "__main__":
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

    day = "03-22_PM"

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
        verbose=False)
    xyz_skel = xyz_labeled[:, :]

    option = options[day]

    organ_dict, b_i = organ_segmentation(gb, bc, mesh, label_mesh, xyz_skel, xyz_stem_node, option=option,
                                         visualize=True)
    organ_collect = get_collection_organ_pcd(mesh, b_i)

    print(get_organ_description(organ_dict, organ_collect, gb, xyz_labeled, xyz_stem_node))
