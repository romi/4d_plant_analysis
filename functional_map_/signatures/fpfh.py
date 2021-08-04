import numpy as np
from tqdm import tqdm
from utils import binning3, get_neighbors
import open3d
import time
from point_cloud_clean import clean_point_cloud


def get_fpfh(xyz, voxel_size, normalize = True):
    pcd = open3d.geometry.PointCloud()
    pcd.points = open3d.utility.Vector3dVector(xyz)
    radius_normal = voxel_size * 2
    print(":: Estimate normal with search radius %.3f." % radius_normal)
    pcd.estimate_normals(
        open3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))
    radius_feature = voxel_size * 5
    print(":: Compute FPFH feature with search radius %.3f." % radius_feature)
    pcd_fpfh = open3d.pipelines.registration.compute_fpfh_feature(
        pcd, open3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
    fpfh_value = pcd_fpfh.data.T
    if normalize == True:
        fpfh_value = fpfh_value / fpfh_value.max(axis = 0).mean()
    return fpfh_value


day1 = '03-22_PM'
day2 = '03-22_AM'
pc_path = '../data/lyon2/processed/{}.ply'
pcd_clean_option = {'downsample_rate': 80,
                    'voxel_size': 1.1,
                    'crop': False,
                    'crop_low_bound': 0.0,
                    'crop_up_bound': 1}
verbose = True

if __name__ == "__main__":
    pcd1 = open3d.io.read_point_cloud(pc_path.format(day1))

    print("Clean Point Cloud")
    print(64 * "=")
    t_start = time.time()
    pcd1, description_1 = clean_point_cloud(pcd1, option=pcd_clean_option, verbose=verbose)
    t_end = time.time()
    print(64 * "-")
    print("point cloud 1: {} points ".format(description_1['point_number']))
    print("used {:.3f}s".format(t_end - t_start))
    voxel_size = description_1['voxel_size']

    xyz = np.asarray(pcd1.points)
    fpfh_value = get_fpfh(xyz, voxel_size)
    print(np.asarray(fpfh_value).shape)
    print("")

    for i in range(fpfh_value.shape[1]):
        if i % 5 == 0:
            color_value = 0.0 * np.ones( (fpfh_value.shape[0], 3) )
            color_value[:, 0] = fpfh_value[:, i]
            pcd1.colors = open3d.utility.Vector3dVector(color_value)
            open3d.visualization.draw_geometries([pcd1])
