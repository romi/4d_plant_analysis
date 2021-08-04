import numpy as np
import open3d
from sklearn.cluster import DBSCAN
from collections import Counter


def clean_point_cloud(pcd, option={}, verbose=False, translate=True):
    """

    :param pcd: point cloud to operate
    :param option: the set of parameters of the cleaning
    :param verbose: boolean, if to print some information or not
    :return: the point cloud cleaned
    """
    if 'downsample_rate' not in option:  # We keep pcd / down sample rate points
        option['downsample_rate'] = 80
    if 'crop' not in option:  # If we crop the point cloud or not
        option['crop'] = False
    else:
        if 'crop_low_bound' not in option:  # the proportion of crop
            option['crop_low_bound'] = 0.0  # defined as the height proportion
        if 'crop_up_bound' not in option:
            option['crop_up_bound'] = 1.0

    description = get_description_point_cloud(pcd)
    if translate:
        pcd.translate(-description['coord_mean'])

    if 'voxel_size' not in option:  # if the voxel size is not given, we calculate it according to down sample rate
        voxel_size = np.min(description['coord_maxmin']) / option['downsample_rate']
    else:
        voxel_size = option['voxel_size']

    if 'cluster_distance_threshold' not in option:
        cluster_distance_threshold = 2
    else:
        cluster_distance_threshold = option['cluster_distance_threshold']

    if verbose:
        print("::Down sampling point cloud with voxel_size: {:2}".format(voxel_size))
    pcd_down = pcd.voxel_down_sample(voxel_size)

    # pcd_down = remove_small_clusters(pcd_down, cluster_distance_threshold * voxel_size)

    if option['crop'] is True:
        pcd_down = crop_point_cloud(pcd_down, option['crop_up_bound'], option['crop_low_bound'])
        if verbose:
            print("::crop point cloud from the {:.2f} to {:.2f}".format(option['crop_up_bound'],
                                                                        option['crop_low_bound']))

        pcd_down = remove_small_clusters(pcd_down, cluster_distance_threshold * voxel_size)

    description = get_description_point_cloud(pcd_down)
    description['voxel_size'] = voxel_size  # collect the description of point cloud cleaned
    pcd.translate(-description['coord_mean'])

    return pcd_down, description


def get_description_point_cloud(pcd):
    """
    Get some general description of the pcd
    :param pcd: open3d point cloud
    :return: a set of descriptor of the point cloud: the mean value of coordinates, the range of coordinates ...
    """
    description = {}
    points = np.asarray(pcd.points)
    description['point_number'] = points.shape[0]
    description['coord_maxmin'] = np.max(points, axis=0) - np.min(points, axis=0)
    description['coord_mean'] = np.mean(points, axis=0)
    return description


def remove_small_clusters(pcd, distance_threshold):
    """
    Remove the isolated clusters in the point cloud
    :param pcd: open3d point cloud
    :param distance_threshold: the distance threshold used in DBScan in order to form clusters
    :return: point cloud keeping only the principal cluster
    """
    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors)
    dbscan = DBSCAN(min_samples=2, eps= distance_threshold, p=0).fit(points)
    labels = dbscan.labels_
    label_counter = Counter(labels)

    principal_cluster_count = 0
    principal_cluster = -1
    for cl in np.unique(labels):
        points_cl = points[labels == cl]
        if cl == -1:
            continue
        if label_counter[cl] > principal_cluster_count:
            principal_cluster = cl
            principal_cluster_count = label_counter[cl]
    points = points[labels == principal_cluster]
    if colors.shape[0] > 0:
        colors = colors[labels == principal_cluster]
    pcd.points = open3d.utility.Vector3dVector(points)
    pcd.colors = open3d.utility.Vector3dVector(colors)
    return pcd


def crop_point_cloud(pcd, up_bound_proportion, low_bound_proportion):
    """
    :param pcd: open3d point cloud
    :param up_bound_proportion: the up proportion of the crop bound
    :param low_bound_proportion: the low proportion of the crop bound
    :return: pcd cropped
    """
    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors)
    z_min = np.min(points[:, 2])
    z_max = np.max(points[:, 2])
    low_bound = z_min + low_bound_proportion * (z_max - z_min)
    up_bound = z_min + up_bound_proportion * (z_max - z_min)
    point_index_keep = np.logical_and(low_bound < points[:, 2], points[:, 2] < up_bound)
    points = points[point_index_keep]
    # colors = colors[point_index_keep]
    pcd.points = open3d.utility.Vector3dVector(points)
    # pcd.colors = open3d.utility.Vector3dVector(colors)
    return pcd


day = '03-22'
hour = "PM"
# pc_path = './data/lyon2_segmented/2021-{}_{}.ply'
# out_path = './data/lyon2/processed/{}_{}_segmented.ply'
pc_path = "./data/lyon3/original/arabidoA_2021-{}_{}.ply"
out_path = "./data/lyon3/processed/{}_{}.ply"

crop_low_bound_dict = {
    # ('03-18', 'AM'): 0.53,
    # ('03-18', 'PM'): 0.495,
    # ('03-19', 'AM'): 0.40,
    # ('03-19', 'PM'): 0.37,
    # ('03-20', 'AM'): 0.28,
    # ('03-20', 'PM'): 0.27,
    # ('03-21', 'AM'): 0.22,
    # ('03-21', 'PM'): 0.21,
    # ('03-22', 'AM'): 0.19,
    # ('03-22', 'PM'): 0.18,
    # ('03-23', 'AM'): 0.19,
    # ('03-23', 'PM'): 0.165,
}

if __name__ == "__main__":
    days = ['05-19']
    hours = ['AM']
    for day in days:
        for hour in hours:
            pcd1 = open3d.io.read_point_cloud(pc_path.format(day, hour))
            get_description_point_cloud(pcd1)
            open3d.visualization.draw_geometries([pcd1])
            pcd_down, des = clean_point_cloud(pcd1,
                                              option={'downsample_rate': 50,
                                                      'voxel_size': 0.2,
                                                      'crop': True,
                                                      'crop_low_bound': 0.42,
                                                      'crop_up_bound': 0.9,
                                                      'cluster_distance_threshold': 50},
                                              verbose=True)

            open3d.visualization.draw_geometries([pcd_down])
            open3d.io.write_point_cloud(out_path.format(day, hour), pcd_down)
            print(des)
