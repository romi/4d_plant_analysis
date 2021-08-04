from sklearn.neighbors import KDTree
import numpy as np


def get_mesh_landmarks(mesh, skeleton_landmarks):
    pcd_xyz = np.asarray(mesh.vertices)
    tree = KDTree(pcd_xyz)
    matches = tree.query(skeleton_landmarks, k=1, return_distance=False).flatten()
    return matches


def get_skeleton(organ, downsample_rate=0.15):
    skeleton_connect = organ["skeleton_connection"]
    segment_index = organ["segment_index"]
    skeleton_xyz = organ["skeleton_pcd"]

    stem = -1
    if skeleton_connect.shape[0] > 0:
        for k in np.unique(skeleton_connect):
            if k not in segment_index:
                stem = k
        if stem == -1:
            print("error stem not found")
            return
    else:
        stem = segment_index[0]
        skeleton_xyz = skeleton_xyz[skeleton_xyz[:, 2].argsort()]
        downsample_rate = 0.5 * downsample_rate

    if downsample_rate < 1:
        xyz_skel_downsampled = []
        if len(segment_index) <= 1:
            for seg_ind in segment_index:
                xyz_seg = skeleton_xyz[skeleton_xyz[:, 3] == seg_ind]
                if xyz_seg.shape[0] > int(3 / downsample_rate):
                    xyz_seg = xyz_seg[:xyz_seg.shape[0] - 1:int(0.5 / downsample_rate), :]
                elif xyz_seg.shape[0] > int(1 / downsample_rate):
                    xyz_seg = xyz_seg[:xyz_seg.shape[0] - 1:int(0.4 / downsample_rate), :]
                else:
                    xyz_seg = xyz_seg[:xyz_seg.shape[0] - 1:int(0.2 / downsample_rate), :]

            xyz_seg = np.vstack([xyz_seg, xyz_seg[-1, :]])
            xyz_skel_downsampled.append(xyz_seg)
        else:
            for seg_ind in segment_index:
                xyz_seg = skeleton_xyz[skeleton_xyz[:, 3] == seg_ind]
                if xyz_seg.shape[0] <= 5:
                    xyz_skel_downsampled.append(xyz_seg)
                xyz_seg_new = xyz_seg[:xyz_seg.shape[0] - 1:int(1 / downsample_rate), :]
                xyz_seg_new = np.vstack([xyz_seg_new, xyz_seg[-1, :]])
                xyz_skel_downsampled.append(xyz_seg_new)
        xyz_skel_downsampled = np.vstack(xyz_skel_downsampled)
        skeleton_xyz = xyz_skel_downsampled

    if skeleton_connect.shape[0] == 0:  # find the stem
        xyz = skeleton_xyz[:, :3]
        connect = [[i, i + 1] for i in range(skeleton_xyz.shape[0] - 1)]
        return xyz, connect, skeleton_xyz

    stem_connect_point = organ["stem_connect_point"]
    if stem_connect_point is None:
        stem_connect_point = skeleton_xyz[np.argmin( skeleton_xyz[:, 2] )][:3]
        stem_connect_point[2] = stem_connect_point[2] - 1
    stem_connect_point = np.append(stem_connect_point, stem)
    skeleton_xyz = np.vstack([skeleton_xyz, stem_connect_point])
    xyz = skeleton_xyz[:, :3]
    xyz = np.vstack([xyz, ])

    connect = []
    for i in range(skeleton_xyz.shape[0] - 1):
        if skeleton_xyz[i, 3] == skeleton_xyz[i + 1, 3]:
            connect.append([i, i + 1])
    for seg_connect in skeleton_connect:

        seg_1 = np.where(skeleton_xyz[:, 3] == seg_connect[0])[0]
        seg_2 = np.where(skeleton_xyz[:, 3] == seg_connect[1])[0]

        res = [-1, -1]
        min_dist = 10 ** 3

        for i in seg_1:
            for j in seg_2:
                d = np.linalg.norm(skeleton_xyz[i, :3] - skeleton_xyz[j, :3])
                if d < min_dist:
                    min_dist = d
                    res = [i, j]
        connect.append(res)

    return xyz, connect, skeleton_xyz