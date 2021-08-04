import numpy as np
from tqdm import tqdm
from utils import binning3, get_neighbors
import open3d
import math
import numpy.linalg as LA
from collections import defaultdict
import itertools
import time
from point_cloud_clean import clean_point_cloud


def shot_calculator(point_cloud_o3d, neighbors_dict, R, soft=False):
    # soft voting: not implemented
    shot = np.zeros((len(point_cloud_o3d.points), 352))

    print("calculating SHOT...")
    for i1 in tqdm(range(len(point_cloud_o3d.points))):
        p1 = point_cloud_o3d.points[i1]
        n1 = point_cloud_o3d.normals[i1]

        """
        build LRF
        """
        denom = 0
        M = np.zeros((3, 3))
        for i2 in neighbors_dict[i1]:
            p2 = point_cloud_o3d.points[i2]
            denom += (R - LA.norm(p1 - p2))
            M += (R - LA.norm(p1 - p2)) * np.outer(p2 - p1, (p2 - p1).T)
        assert denom != 0, "neighbor cnt: {}, denom: {}".format(
            len(neighbors_dict[i1]), denom)
        M /= denom
        u, s, vh = np.linalg.svd(M, full_matrices=True)
        eigenvalues = s
        eigenvectors = u
        order = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[order]
        eigenvectors = eigenvectors[:, order]

        x_pos, y_pos, z_pos = eigenvectors
        Sx_pos, Sz_pos = 0, 0

        for i2 in neighbors_dict[i1]:
            p2 = point_cloud_o3d.points[i2]
            if np.dot(p2 - p1, x_pos) >= 0:
                Sx_pos += 1
            if np.dot(p2 - p1, z_pos) >= 0:
                Sz_pos += 1

        # if there are more points whose x >= 0, then x will be x_pos
        x = x_pos * pow(-1, Sx_pos < len(neighbors_dict) / 2)
        z = z_pos * pow(-1, Sz_pos < len(neighbors_dict) / 2)
        y = np.cross(z, x)

        # normalize the  axis!
        x = x / LA.norm(x)
        y = y / LA.norm(y)
        z = z / LA.norm(z)

        """
        divide the neighbors of p1 into 32 volumes
        """
        # azimuths = np.arange(0, 2*np.pi, 2*np.pi/8) # range: [0, 2*pi)
        # elavations = [-np.pi, 0] # range: [-pi, pi]
        # radials = [0, R/2] # range: [0, R]
        volumes = defaultdict(lambda: [])
        for i2 in neighbors_dict[i1]:
            p2 = point_cloud_o3d.points[i2]
            d = LA.norm(p2 - p1)
            azi = np.arccos(np.dot(x, (p2 - p1) / d))  # np.arccos: [0, pi]
            # check y coordinate
            if p2[1] < p1[1]: azi = 2 * np.pi - azi
            azi = math.floor(azi / (2 * np.pi / 8))
            # compare z coordinate, if less, ela will be 0, o.w. it's 1
            ela = int(p2[2] >= p1[2])
            # compare distance with R/2, if less rad will be 0, o.w. it's 1
            rad = int(d >= R / 2)
            volumes[(azi, ela, rad)].append(i2)
        # print(volumes.keys())

        """
        for each volume, build a histogram of cos(theta_i)
        """
        for idx, (azi, ela, rad) in enumerate(itertools.product(range(0, 8),
                                                                [0, 1], [0, 1])):
            cosines = []
            for i2 in volumes[(azi, ela, rad)]:
                cosines.append(
                    np.dot(n1, point_cloud_o3d.normals[i2]))

            hist, _ = np.histogram(cosines, bins=11, range=(-1, 1))
            shot[i1][idx * 11:(idx + 1) * 11] = hist
        shot[i1] /= LA.norm(shot[i1])

    return shot


day1 = '05-29'
day2 = 'colB_2'
pc_path = './data/{}.ply'

maxBasis = 50
mesh_radius_factor = 4
descriptor_weight = [1, 1, 0.01]
plot_basis = True
plot_mesh = True
verbose = True
saveoff = True
pcd_clean_option = {'downsample_rate': 100,
                      'voxel_size': 1.2,
                      'crop': False,
                      'crop_low_bound': 0.0,
                      'crop_up_bound': 1}
if_train = False

if __name__ == "__main__":
    pcd1 = open3d.io.read_point_cloud(pc_path.format(day2))
    print("Clean Point Cloud")
    print(64 * "=")
    t_start = time.time()
    pcd1, description_1 = clean_point_cloud(pcd1, option=pcd_clean_option, verbose=verbose)
    t_end = time.time()
    print(64 * "-")
    print("point cloud 1: {} points ".format(description_1['point_number']))
    print("used {:.3f}s".format(t_end - t_start))
    voxel_size = description_1['voxel_size']

    open3d.visualization.draw_geometries([pcd1])

    radius_shot_lrf = 10 * voxel_size
    # radius_shot_lrf = 0.2
    if if_train:
        neighbors_dict_shot_lrf = get_neighbors(pcd1, radius_shot_lrf)

        shot = shot_calculator(pcd1,
                               neighbors_dict_shot_lrf, radius_shot_lrf)

        # shot_soft = shot_calculator(pcd1,
        #                                  neighbors_dict_shot_lrf, radius_shot_lrf)

        # np.savetxt(ofname+"_spfh.txt", spfh)
        # np.savetxt(ofname+"_fpfh.txt", fpfh)
        np.savetxt("./data/shot_{}.txt".format(radius_shot_lrf), shot)

    shot_value = np.loadtxt("./data/shot_{}.txt".format(radius_shot_lrf))
    print("shot descriptor shape: ", shot_value.shape)

    for i in range(5):
        color_value = 0.0 * np.ones( (shot_value.shape[0], 3) )

        color_value[:, 0] = ( shot_value[:, 11 * i + 1] - np.min(shot_value[:, 11 * i + 1])) \
                            / (np.max(shot_value[:, 11 * i + 1]) - np.min(shot_value[:, 11 * i + 1]))
        color_value[:, 1] = ( shot_value[:, 11 * i + 5] - np.min(shot_value[:, 11 * i + 5])) \
                            / (np.max(shot_value[:, 11 * i + 1]) - np.min(shot_value[:, 11 * i + 1]))
        color_value[:, 2] = ( shot_value[:, 11 * i + 10] - np.min(shot_value[:, 11 * i + 10])) \
                            / (np.max(shot_value[:, 11 * i + 1]) - np.min(shot_value[:, 11 * i + 1]))
        pcd1.colors = open3d.utility.Vector3dVector(color_value)
        open3d.visualization.draw_geometries([pcd1])