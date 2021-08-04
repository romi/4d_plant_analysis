
"""
Created on Fri Dec 18 09:29:18 2020
@author: mimif
"""

import os
from glob import glob
from pyntcloud import PyntCloud
import open3d as o3d
import numpy as np
import numpy.linalg as LA
import pandas as pd
from collections import defaultdict
import math
from tqdm import tqdm
import itertools
import bisect


def get_neighbors(cloud, radius):
    tree = o3d.geometry.KDTreeFlann(cloud)
    neighbors = []

    for pi, point in enumerate(cloud.points):
        cnt, idxs, dists = tree.search_radius_vector_3d(point, radius)
        # one can't be oneself's neighbor!
        idxs.remove(pi)
        neighbors.append(idxs)

    return neighbors


def binning_cos(a):
    # cosine value's range: [-1, 1]
    hist, _ = np.histogram(a, bins=11, range=[-1, 1])
    return hist


def binning_arctan(a):
    # arctan value's range: [-PI, PI]
    hist, _ = np.histogram(a, bins=11, range=[-np.pi, np.pi])
    return hist


def binning3(alphas, phis, thetas):
    # binning and then concat
    return np.concatenate([binning_cos(alphas),
                           binning_cos(phis),
                           binning_arctan(thetas)])


def spfh_calculator(point_cloud_o3d, neighbors_dict):
    spfh = np.empty((len(point_cloud_o3d.points), 33))

    print("calculating SPFH...")
    for i1 in tqdm(range(len(point_cloud_o3d.points))):
        p1 = point_cloud_o3d.points[i1]
        u = point_cloud_o3d.normals[i1]
        alphas, phis, thetas = [], [], []
        for i2 in neighbors_dict[i1]:
            p2 = point_cloud_o3d.points[i2]
            n2 = point_cloud_o3d.normals[i2]
            """
            for each pair, calculate d, alpha, phi, theta
            """
            d = LA.norm(p2 - p1)
            # assert not np.any(np.isnan((p2-p1)/d)), "{}: {} and {}: {}".format(i1, p1, i2, p2)
            v = np.cross(u, (p2 - p1) / d)
            w = np.cross(u, v)
            alphas.append(np.dot(v, n2))
            phis.append(np.dot(u, (p2 - p1) / d))
            thetas.append(np.arctan2(np.dot(n2, w), np.dot(n2, u)))
        # print(np.min(alphas), np.max(alphas))
        # print(np.min(phis), np.max(phis))
        # print(np.min(thetas), np.max(thetas))
        spfh[i1] = binning3(alphas, phis, thetas)

    return spfh


def fpfh_calculator(point_cloud_o3d,
                    neighbors_dict=None, radius=None, spfh=None):
    if spfh is None:
        if neighbors_dict is None:
            assert radius is not None, \
                "if neighbors_dict is empty, radius should be set!"
            neighbors_dict = get_neighbors(point_cloud_o3d, radius)
        spfh = spfh_calculator(point_cloud_o3d, neighbors_dict)

    fpfh = np.empty((len(point_cloud_o3d.points), 33))

    print("calculating FPFH...")
    for i1 in tqdm(range(len(point_cloud_o3d.points))):
        p1 = point_cloud_o3d.points[i1]
        fpfh[i1] = spfh[i1]
        k = len(neighbors_dict[i1])
        for i2 in neighbors_dict[i1]:
            p2 = point_cloud_o3d.points[i2]
            # weight of i2 is 1/LA.norm(p1-p2)
            fpfh[i1] += spfh[i2] / LA.norm(p1 - p2) / k
    return fpfh


def lrf_builder(point_cloud_o3d, neighbors_indices, R, center):
    """
    build LRF
    """
    denom = 0
    M = np.zeros((3, 3))
    for i2 in neighbors_indices:
        p2 = point_cloud_o3d.points[i2]
        denom += (R - LA.norm(center - p2))
        M += (R - LA.norm(center - p2)) * np.outer(p2 - center, (p2 - center).T)
    assert denom != 0, "neighbor cnt: {}, denom: {}".format(
        len(neighbors_indices), denom)
    M /= denom
    u, s, vh = np.linalg.svd(M, full_matrices=True)
    eigenvalues = s
    eigenvectors = u
    order = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]

    x_pos, y_pos, z_pos = eigenvectors
    Sx_pos, Sz_pos = 0, 0

    for i2 in neighbors_indices:
        p2 = point_cloud_o3d.points[i2]
        if np.dot(p2 - center, x_pos) >= 0:
            Sx_pos += 1
        if np.dot(p2 - center, z_pos) >= 0:
            Sz_pos += 1

    # if there are more points whose x >= 0, then x will be x_pos
    x = x_pos * pow(-1, Sx_pos < len(neighbors_indices) / 2)
    z = z_pos * pow(-1, Sz_pos < len(neighbors_indices) / 2)
    y = np.cross(z, x)

    # normalize the  axis!
    x = x / LA.norm(x)
    y = y / LA.norm(y)
    z = z / LA.norm(z)

    return x, y, z


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


def shot_soft_calculator(point_cloud_o3d, neighbors_dict, R):
    # soft voting: not implemented
    shot = np.zeros((len(point_cloud_o3d.points), 352))

    print("calculating SHOT SOFT...")
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
        azi_res = 2 * np.pi / 8
        ela_res = np.pi / 2
        rad_res = R / 2
        cosine_res = 2 / 11

        azi_bins = np.arange(0, 2 * np.pi, azi_res)  # range: [0, 2*pi)
        ela_bins = np.arange(-np.pi / 2, np.pi / 2, ela_res)  # range: [-pi/2, pi/2]
        rad_bins = np.arange(0, R, rad_res)  # range: [0, R]
        cosine_bins = np.arange(-1, 1, cosine_res)  # range: [-1, 1]

        # print("azi_bins: ", azi_bins)
        # print("ela_bins: ", ela_bins)
        # print("rad_bins: ", rad_bins)
        # print("cosine_bins: ", cosine_bins)

        # add dummy cell for ela, rad, cosine dimension
        bins = np.zeros((8, 3, 3, 12))  # np.zeros((8,2,2,11))
        for i2 in neighbors_dict[i1]:
            p2 = point_cloud_o3d.points[i2]
            n2 = point_cloud_o3d.normals[i2]
            d = LA.norm(p2 - p1)
            azi = np.arccos(np.dot(x, (p2 - p1) / d))  # np.arccos: [0, pi]
            # check y coordinate
            if p2[1] < p1[1]: azi = 2 * np.pi - azi
            ela = np.arccos(np.dot(z, (p2 - p1) / d)) - np.pi / 2
            rad = d
            cosine = np.dot(n1, n2)

            """
            l = [0, 1]
            bisect.bisect_right(l, 0)   #1
            bisect.bisect_right(l, 0.3) #1
            bisect.bisect_right(l, 1)   #2
            """
            azi_idx_hi = bisect.bisect_right(azi_bins, azi)
            ela_idx_hi = bisect.bisect_right(ela_bins, ela)
            rad_idx_hi = bisect.bisect_right(rad_bins, rad)
            cosine_idx_hi = bisect.bisect_right(cosine_bins, cosine)

            azi_idx_lo = azi_idx_hi - 1
            ela_idx_lo = ela_idx_hi - 1
            rad_idx_lo = rad_idx_hi - 1
            cosine_idx_lo = cosine_idx_hi - 1

            # print("azi {} is in [{}, {}]".format(azi, azi_idx_lo, azi_idx_hi))
            # print("ela {} is in [{}, {}]".format(ela, ela_idx_lo, ela_idx_hi))
            # print("rad {} is in [{}, {}]".format(rad, rad_idx_lo, rad_idx_hi))
            # print("cosine {} is in [{}, {}]".format(cosine, cosine_idx_lo, cosine_idx_hi))

            # if azi_idx_hi is too large, then vote to the first cell
            if azi_idx_hi == len(azi_bins): azi_idx_hi = 0
            # if ela_idx_hi, rad_idx_hi, cosine_idx_hi equal to the length of
            # their corresponding bins, then ignore them by voting to dummy cells

            azi_w_lo = 1 - (azi - azi_idx_lo * azi_res) / azi_res
            ela_w_lo = 1 - (ela - ela_idx_lo * ela_res) / ela_res
            rad_w_lo = 1 - (rad - rad_idx_lo * rad_res) / rad_res
            cosine_w_lo = 1 - (cosine - cosine_idx_lo * cosine_res) / cosine_res

            azi_w_hi = 1 - azi_w_lo
            ela_w_hi = 1 - ela_w_lo
            rad_w_hi = 1 - rad_w_lo
            cosine_w_hi = 1 - cosine_w_lo

            bins[azi_idx_lo][ela_idx_lo][rad_idx_lo][cosine_idx_lo] += \
                azi_w_lo * ela_w_lo * rad_w_lo * cosine_w_lo
            bins[azi_idx_lo][ela_idx_lo][rad_idx_lo][cosine_idx_hi] += \
                azi_w_lo * ela_w_lo * rad_w_lo * cosine_w_hi
            bins[azi_idx_lo][ela_idx_lo][rad_idx_hi][cosine_idx_lo] += \
                azi_w_lo * ela_w_lo * rad_w_hi * cosine_w_lo
            bins[azi_idx_lo][ela_idx_lo][rad_idx_hi][cosine_idx_hi] += \
                azi_w_lo * ela_w_lo * rad_w_hi * cosine_w_hi
            bins[azi_idx_lo][ela_idx_hi][rad_idx_lo][cosine_idx_lo] += \
                azi_w_lo * ela_w_hi * rad_w_lo * cosine_w_lo
            bins[azi_idx_lo][ela_idx_hi][rad_idx_lo][cosine_idx_hi] += \
                azi_w_lo * ela_w_hi * rad_w_lo * cosine_w_hi
            bins[azi_idx_lo][ela_idx_hi][rad_idx_hi][cosine_idx_lo] += \
                azi_w_lo * ela_w_hi * rad_w_hi * cosine_w_lo
            bins[azi_idx_lo][ela_idx_hi][rad_idx_hi][cosine_idx_hi] += \
                azi_w_lo * ela_w_hi * rad_w_hi * cosine_w_hi
            bins[azi_idx_hi][ela_idx_lo][rad_idx_lo][cosine_idx_lo] += \
                azi_w_hi * ela_w_lo * rad_w_lo * cosine_w_lo
            bins[azi_idx_hi][ela_idx_lo][rad_idx_lo][cosine_idx_hi] += \
                azi_w_hi * ela_w_lo * rad_w_lo * cosine_w_hi
            bins[azi_idx_hi][ela_idx_lo][rad_idx_hi][cosine_idx_lo] += \
                azi_w_hi * ela_w_lo * rad_w_hi * cosine_w_lo
            bins[azi_idx_hi][ela_idx_lo][rad_idx_hi][cosine_idx_hi] += \
                azi_w_hi * ela_w_lo * rad_w_hi * cosine_w_hi
            bins[azi_idx_hi][ela_idx_hi][rad_idx_lo][cosine_idx_lo] += \
                azi_w_hi * ela_w_hi * rad_w_lo * cosine_w_lo
            bins[azi_idx_hi][ela_idx_hi][rad_idx_lo][cosine_idx_hi] += \
                azi_w_hi * ela_w_hi * rad_w_lo * cosine_w_hi
            bins[azi_idx_hi][ela_idx_hi][rad_idx_hi][cosine_idx_lo] += \
                azi_w_hi * ela_w_hi * rad_w_hi * cosine_w_lo
            bins[azi_idx_hi][ela_idx_hi][rad_idx_hi][cosine_idx_hi] += \
                azi_w_hi * ela_w_hi * rad_w_hi * cosine_w_hi

        shot[i1] = bins[:, :2, :2, :11].flatten()
        shot[i1] /= LA.norm(shot[i1])

    return shot


if __name__ == "__main__":
    modelnet40_dir = "D:\modelnet40_normal_resampled"

    fnames = [
        os.path.join(modelnet40_dir, "airplane", "airplane_0001.txt"),
        os.path.join(modelnet40_dir, "bottle", "bottle_0012.txt"),
        os.path.join(modelnet40_dir, "desk", "desk_0045.txt"),
        os.path.join(modelnet40_dir, "guitar", "guitar_0010.txt")
    ]

    calculate = True
    if calculate:
        """
        Calculate feature
        """
        for fname in fnames:
            content = np.loadtxt(fname, delimiter=',')
            points = content[:, :3]
            normals = content[:, 3:]
            # change the norm of normals into 1!!
            normals /= LA.norm(normals, axis=-1, keepdims=True)
            point_cloud_pynt = PyntCloud(
                pd.DataFrame(points, columns=['x', 'y', 'z']))
            # point_cloud_pynt.normals = normals

            point_cloud_o3d = point_cloud_pynt.to_instance("open3d", mesh=False)
            point_cloud_o3d.normals = o3d.cpu.pybind.utility.Vector3dVector(normals)
            assert (np.allclose(1, LA.norm(point_cloud_o3d.normals, axis=-1)))

            radius = 0.03
            neighbors_dict = get_neighbors(point_cloud_o3d, radius)

            """
            Compute SPFH
            """
            spfh = spfh_calculator(point_cloud_o3d, neighbors_dict)

            """
            Compute FPFH
            """
            fpfh = fpfh_calculator(point_cloud_o3d, neighbors_dict, spfh)

            """
            Compute SHOT
            """
            radius_shot_lrf = 0.07
            # radius_shot_lrf = 0.2
            neighbors_dict_shot_lrf = get_neighbors(point_cloud_o3d, radius_shot_lrf)

            shot = shot_calculator(point_cloud_o3d,
                                   neighbors_dict_shot_lrf, radius_shot_lrf)

            shot_soft = shot_soft_calculator(point_cloud_o3d,
                                             neighbors_dict_shot_lrf, radius_shot_lrf)

            ofname = os.path.splitext(os.path.basename(fname))[0]
            # np.savetxt(ofname+"_spfh.txt", spfh)
            # np.savetxt(ofname+"_fpfh.txt", fpfh)
            np.savetxt(ofname + "_shot_{}.txt".format(radius_shot_lrf), shot)
            np.savetxt(ofname + "_shot_soft_{}.txt".format(radius_shot_lrf), shot_soft)

    """
    Test
    """
    selected_indices = [
        [11, 8622, 3205],
        [415, 685, 5510],
        [1686, 6793, 815],
        [1859, 7847, 1180]
    ]

    test_python = True
    if test_python:
        for fname, (i1, i2, i3) in zip(fnames, selected_indices):
            ofname = os.path.splitext(os.path.basename(fname))[0]
            print(ofname)
            spfh = np.loadtxt(ofname + "_spfh.txt")
            fpfh = np.loadtxt(ofname + "_fpfh.txt")
            shot_007 = np.loadtxt(ofname + "_shot_0.07.txt")
            shot_soft_007 = np.loadtxt(ofname + "_shot_soft_0.07.txt")

            print("spfh, fpfh, shot_007, shot_soft_007: ")
            cosine_similarities = []
            cosine_similarities_dissimilar = []
            for feature in [spfh, fpfh, shot_007, shot_soft_007]:
                f1 = feature[i1] / LA.norm(feature[i1])
                f2 = feature[i2] / LA.norm(feature[i2])
                f3 = feature[i3] / LA.norm(feature[i3])
                cosine_similarities.append(np.dot(f1, f2))
                cosine_similarities_dissimilar.append(np.dot(f1, f3))
            print(cosine_similarities)
            print(cosine_similarities_dissimilar)

    test_pcl = False
    if test_pcl:
        for fname, (i1, i2, i3) in zip(fnames, selected_indices):
            ofname = os.path.splitext(os.path.basename(fname))[0]
            print(ofname)
            fpfh = np.genfromtxt(ofname + "_pcl_fpfh.txt", delimiter=",")
            shot = np.genfromtxt(ofname + "_pcl_shot.txt", delimiter=",")

            print("pcl fpfh, pcl shot: ")
            cosine_similarities = []
            cosine_similarities_dissimilar = []
            for feature in [fpfh, shot]:
                f1 = feature[i1] / LA.norm(feature[i1])
                f2 = feature[i2] / LA.norm(feature[i2])
                f3 = feature[i3] / LA.norm(feature[i3])
                cosine_similarities.append(np.dot(f1, f2))
                cosine_similarities_dissimilar.append(np.dot(f1, f3))
            print(cosine_similarities)
            print(cosine_similarities_dissimilar)

