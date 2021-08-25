import numpy as np
import open3d
import copy

from sklearn.neighbors import KDTree
from track_point import load_p2p_matching_series, track_one_point


def downsample_pcd(pcd, down_sample_rate=0.003):
    # voxel_size = np.mean(pcd.compute_nearest_neighbor_distance())
    return np.asarray(pcd.points)[::int(1/down_sample_rate)]


def get_sample_points_track(xyz, pcd_collector, mapping_collector):
    """

    :param xyz: numpy array of shape N*3, the xyz coordinates of the points
    :return: the track of the input points in the time series
    """

    track_collection = []
    for  xyz_ in xyz:
        track = track_one_point(xyz_, pcd_collector, mapping_collector)
        complete = True
        for t in track:
            if t.shape[0] == 0:
                complete = False
        if complete:
            track_collection.append(track)
    return track_collection


def predict_position_from_track(track, dt=1):
    v = track[0] - track[1]
    a = (track[0] + track[2] - 2 * track[1]) * 2
    return track[0] + dt * v


def extrapolate(pcd, track_collection, predict_position_collection):
    current_position = np.vstack([t[0] for t in track_collection])
    next_position = np.vstack(predict_position_collection)

    translation = next_position - current_position

    xyz = np.asarray(pcd.points)
    tree = KDTree(current_position)

    distance, indices = tree.query(xyz, return_distance=True, k=3)

    xyz_extrapolated = copy.deepcopy(xyz)
    for i, xyz_ in enumerate(xyz_extrapolated):
        weights = 1 / (0.002 + distance[i])
        weights = weights / np.sum(weights)
        d_xyz = (weights * translation[indices[i]].T).sum(axis=1)
        xyz_ += d_xyz
    pcd_extrapolate = open3d.geometry.PointCloud()
    pcd_extrapolate.points = open3d.utility.Vector3dVector(xyz_extrapolated)
    return pcd_extrapolate


if __name__ == "__main__":
    dataset = "lyon2"
    match_path_format = "../data/{}/registration_result/{}_to_{}/"
    days = ["03-21_PM", "03-22_AM", "03-22_PM"]

    pcd_collector, mapping_collector = load_p2p_matching_series(days, dataset, match_path_format)
    last_pcd = pcd_collector[days[-1]]
    xyz_sampled = downsample_pcd(
        last_pcd, down_sample_rate=0.02
    )
    track_collection = get_sample_points_track(xyz_sampled, pcd_collector, mapping_collector)

    pos_predict = []
    for track in track_collection:
        pos_predict.append(predict_position_from_track(track))
    pcd_extrapolate = extrapolate(last_pcd, track_collection, predict_position_collection=pos_predict)

    last_pcd = last_pcd.translate([0, 200, 0])
    last_2_pcd = pcd_collector[days[-2]].translate([0, 400, 0])
    open3d.visualization.draw_geometries([pcd_extrapolate, last_pcd, last_2_pcd])
    # xyz_ori = xyz_sampled
    # xyz_predict = np.vstack(pos_predict)
    # pcd_predict = open3d.geometry.PointCloud()
    #
    # pcd_predict.points = open3d.utility.Vector3dVector(xyz_ori)
    # open3d.visualization.draw_geometries([pcd_predict])
    #
    # pcd_predict.points = open3d.utility.Vector3dVector(xyz_predict)
    # open3d.visualization.draw_geometries([pcd_predict])
    print("")
