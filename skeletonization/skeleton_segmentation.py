import time
import json
import os
from utils import read_graph, \
    plot_skeleton
import numpy as np
import networkx as nx
import copy
import open3d


class Branch(object):

    def __init__(self, xyz, color=None):
        if color is None:
            color = [0.6, 0.2, 0.7]
        self.color = color
        self.xyz = xyz
        self.update()

    def merge(self, branch):
        self.xyz = np.vstack([self.xyz, branch.xyz])
        self.update()

    def get_head_tail(self):
        d = 10 ** -5
        head, tail = -1, -1
        for i in range(self.xyz.shape[0]):
            for j in range(i + 1, self.xyz.shape[0]):
                d12 = np.linalg.norm(self.xyz[i, :] - self.xyz[j, :])
                if d12 > d:
                    head = i
                    tail = j
                    d = d12
        return self.xyz[head], self.xyz[tail]

    def update(self):
        xyz = self.xyz
        self.center = xyz.mean(axis=0)
        if xyz.shape[0] > 1:
            xyz_head, xyz_tail = self.get_head_tail()
            self.direction = (xyz_tail - xyz_head) / (10 ** -5 + np.linalg.norm(xyz_tail - xyz_head))
        else:
            self.direction = np.array([0.0, 0.0, 0.0])

    def __len__(self):
        return self.xyz.shape[0]


def get_sub_tree_count(graph, pre, cand):
    res = 0

    q = [cand]
    visited = [pre]
    while len(q) > 0:
        current = q.pop()
        visited.append(current)
        res += 1
        neighbors = list(graph.neighbors(current))
        for neigh in neighbors:
            if neigh not in visited:
                q.append(neigh)
    return res


def get_branches(graph, xyz, root, height_threshold=5):
    if not nx.is_connected(graph):
        print("InputGraph need to be connected")
        return

    graph_nodes = list(graph.nodes)
    top = max(graph_nodes, key=lambda n: xyz[n, 2])
    stem_nodes = nx.shortest_path(graph, source=root, target=top)
    stem_nodes = stem_nodes[0: int(0.8 * len(stem_nodes))]

    # Continue the search
    visited = copy.deepcopy(stem_nodes)
    current = stem_nodes[-1]
    q = [current]

    while len(q) > 0:
        current = q.pop()
        neighbors = list(graph.neighbors(current))

        next_candidates = []
        for neigh in neighbors:
            if not neigh in visited:
                next_candidates.append(neigh)

        if len(next_candidates) == 0:
            break

        if len(next_candidates) == 1:
            stem_nodes.append(next_candidates[0])
            q.append(next_candidates[0])
            visited.append(next_candidates[0])
            continue

        next_sub_tree_count = 0
        next_node = -1
        for cand in next_candidates:
            cand_sub_tree_count = get_sub_tree_count(graph, current, cand)
            if cand_sub_tree_count > next_sub_tree_count:
                next_node = cand
                next_sub_tree_count = cand_sub_tree_count
        stem_nodes.append(next_node)
        q.append(next_node)
        visited.append(next_node)

    stem_nodes = stem_nodes[:int(0.92 * len(stem_nodes))]
    stem_nodes = list(filter(lambda n: xyz[n, 2] > xyz[root, 2] + height_threshold, stem_nodes))

    branches = []
    not_visited = list(graph.nodes)
    not_visited.sort(key=lambda n: xyz[n, 2])
    head_candidates = [root]
    while len(not_visited) > 0:
        if len(head_candidates) <= 0:
            head_candidates.append(not_visited.pop())
        head = head_candidates.pop()

        branch = [head]
        not_visited.remove(head)
        current = head
        while len(not_visited) > 0:
            neighbors = list(graph.neighbors(current))
            next_candidates = []
            for nn in neighbors:
                if nn in not_visited:
                    next_candidates.append(nn)

            if len(next_candidates) == 0:
                xyz_branch = xyz[copy.copy(branch)]
                branches.append(Branch(xyz_branch))
                break

            if len(next_candidates) == 1:
                next = next_candidates[0]
                branch.append(next)
                not_visited.remove(next)
                if len(not_visited) <= 0:
                    xyz_branch = xyz[copy.copy(branch)]
                    branches.append(Branch(xyz_branch))
                current = next
                continue

            for nn in next_candidates:
                head_candidates.append(nn)
            xyz_branch = xyz[copy.copy(branch)]
            branches.append(Branch(xyz_branch))
            break

    return branches, xyz[stem_nodes]


def plot_branches(branches):
    point_count = sum([len(br) for br in branches])
    xyz = np.zeros((point_count, 3))
    color_value = 0.85 * np.ones(xyz.shape)
    i = 0
    for br in branches:
        xyz[i: i + len(br)] = br.xyz
        i += len(br)

    # color = (np.max(xyz, axis=0) - xyz) / (np.max(xyz, axis=0) - np.min(xyz, axis=0))
    # color = (1.4 * color) % 1
    # color = color[:, [2, 1, 0]]
    i = 0
    for br in branches:
        color_value[i: i + len(br)] = np.random.rand(3)
        i += len(br)
    pcd = open3d.geometry.PointCloud()
    pcd.points = open3d.utility.Vector3dVector(xyz)
    pcd.colors = open3d.utility.Vector3dVector(color_value)
    open3d.visualization.draw_geometries([pcd])


def merge_small_branches(branches, distance_threshold=20, branch_number_threshold=2):
    """
    Merge the branches with smaller sizes to larger one near by
    :param branch_number_threshold: int, the threshold to decide whether one branch is small or large
    :param distance_threshold: float, distance threshold for merging
    :param branches:
    :return: merged branches
    """
    merged_branches = [br for br in branches if len(br) > branch_number_threshold]
    for i, br in enumerate(branches):
        if len(br) <= branch_number_threshold:
            distances = [np.linalg.norm(br.center - br_m.center) for br_m in merged_branches]
            if min(distances) < distance_threshold:
                j = distances.index(min(distances))
                merged_branches[j].merge(br)
            else:
                merged_branches.append(br)
    return merged_branches


def merge_similar_branches(branches, orientation_threshold=0.7, distance_threshold=8, branch_number_threshold=10):
    """
    Merge the branches with similar orientation and close postion
    :param branch_number_threshold:
    :param branches:
    :param orientation_threshold:
    :param distance_threshold:
    :return: merged branches
    """
    branches.sort(key=lambda x: -len(x))
    i = 0
    while i < len(branches) and len(branches[i]) > branch_number_threshold:
        i += 1

    while i < len(branches):
        j = i + 1
        while j < len(branches):
            br_1 = branches[i]
            br_2 = branches[j]
            if np.linalg.norm(br_1.center - br_2.center) < distance_threshold and \
                    sum(br_1.direction * br_2.direction) > orientation_threshold:
                br_1.merge(br_2)
                del branches[j]
            else:
                j += 1
        i += 1
    return branches


def merge_main_stem(branches, stem_nodes):
    i = 0
    main_stem_branches = []

    while i < len(branches):
        br = branches[i]
        stem_node_count = 0
        new_xyz = []
        for j, node in enumerate(br.xyz):
            match_index = np.where((stem_nodes == tuple(node)).all(axis=1))[0]
            if len(match_index) > 0:
                if len(main_stem_branches) == 0:
                    main_stem_branches.append(Branch(node.reshape(-1,3)))
                else:
                    main_stem_branches[0].xyz = np.vstack([main_stem_branches[0].xyz, node])
                    main_stem_branches[0].update()
            else:
                new_xyz.append(copy.deepcopy(node))
            stem_node_count += len(match_index)
        if len(new_xyz) > 0:
            br.xyz = np.vstack(new_xyz)
            br.update()
            i += 1
        else:
            del branches[i]

    for i in range(1, len(main_stem_branches)):
        main_stem_branches[0].merge(main_stem_branches[i])

    if len(main_stem_branches) > 0:
        branches.append(main_stem_branches[0])
    return branches


def merge_soil(branches, height_threshold):
    i = 0
    soil_branches = []
    while i < len(branches):
        br = branches[i]
        if br.center[2] < height_threshold and np.linalg.norm(br.center[:2]) < 10:
            soil_branches.append(copy.deepcopy(br))
            del branches[i]
        else:
            i += 1

    for i in range(1, len(soil_branches)):
        soil_branches[0].merge(soil_branches[i])

    if len(soil_branches) > 0:
        branches.append(soil_branches[0])
    return branches


def skeleton_segmentation(xyz_ori,
                          connection,
                          day,
                          dataset,
                          visualize=True,
                          verbose=True,
                          options=None):
    xyz_ori = np.array(xyz_ori)
    connection = np.array(connection)

    if options is None:
        options_path = "../hyper_parameters/lyon2.json"
        with open(options_path, "r") as json_file:
            options = json.load(json_file)

    path_format = options['root_path']
    if day not in options["skeleton_extract_option"]:
        options["skeleton_extract_option"][day] = {}

    if "skel_segmentation" not in options["skeleton_extract_option"][day]:
        options["skeleton_extract_option"][day]["skel_segmentation"] = {}

    option = options["skeleton_extract_option"][day]["skel_segmentation"]

    t_0 = time.time()
    if visualize:
        plot_skeleton(xyz_ori, connection, plot_point=False)

    graph = nx.Graph()
    for c in connection:
        c_v = xyz_ori[c[0]] - xyz_ori[c[1]]
        c_v[0] = 10 * c_v[0]
        c_v[1] = 10 * c_v[1]
        graph.add_edge(c[0], c[1], weight=np.linalg.norm(c_v))
    graph_nodes = list(graph.nodes)

    root = min(graph_nodes, key=lambda n: xyz_ori[n, 2])  # Get the root, the node with smallest z value

    if "height_threshold" not in option:
        option["height_threshold"] = 3
    branches, stem_nodes = get_branches(graph, xyz_ori, root, height_threshold=option["height_threshold"])


    if "branch_number_lim" not in option:
        option["branch_number_lim"] = 2
    merged_branches = merge_small_branches(branches, branch_number_threshold=option["branch_number_lim"])
    merged_branches = merge_similar_branches(merged_branches, branch_number_threshold=int(1.5 * option["branch_number_lim"]))


    if "distance_threshold" not in option:
        option["distance_threshold"] = 3
    merged_branches = merge_soil(merged_branches, xyz_ori[root,2] + option["distance_threshold"])

    if "direction_threshold" not in option:
        option["direction_threshold"] = 0.7

    if "diversion_threshold" not in option:
        option["diversion_threshold"] = 8

    if dataset not in ['tomato', "maize"]:
        merged_branches = merge_main_stem(merged_branches,
                                          stem_nodes)

    if visualize:
        plot_branches(merged_branches)
    # output the result
    point_count = sum([len(br) for br in merged_branches])
    xyz = np.zeros((point_count, 3))
    branch_index = np.zeros((point_count, 1))
    i = 0
    for j, br in enumerate(merged_branches):
        xyz[i: i + len(br)] = br.xyz
        branch_index[i: i + len(br), :] = j
        i += len(br)

    branch_connection = []
    for c in connection:
        xyz_1 = xyz_ori[c[0]]
        xyz_2 = xyz_ori[c[1]]
        index_1 = np.where(np.all(xyz == xyz_1, axis=1))[0]
        index_2 = np.where(np.all(xyz == xyz_2, axis=1))[0]
        if len(index_1) > 0 and len(index_2) > 0:
            br_1 = branch_index[index_1[0]][0]
            br_2 = branch_index[index_2[0]][0]
            if br_1 != br_2 and (min(br_1, br_2), max(br_1, br_2)) not in branch_connection:
                branch_connection.append((min(br_1, br_2), max(br_1, br_2)))

    branch_connection = [list(c) for c in branch_connection]
    xyz = np.hstack([xyz, branch_index])
    np.savetxt(path_format.format(dataset, day)+"skeleton_{}_noted.csv".format(day), xyz)
    np.savetxt(path_format.format(dataset, day)+"stem_nodes_ordered.csv".format(day), stem_nodes)
    np.savetxt(path_format.format(dataset, day)+"branch_connection_{}.csv".format(day), np.array(branch_connection))

    if verbose:
        print(64 * "-")
        print("Skeleton Segmentation Finished")
        print("::used {:.2f}s".format(time.time() - t_0))
        print("")

    return xyz, branch_connection


skel_path = "../data/{}/{}/skeleton_{}_connected.txt"
skel_noted_path = "../data/{}/{}/skeleton_{}_noted.csv"
branch_connect_path = "../data/{}/{}/branch_connection_{}.csv"
# days = ["03-19_AM", "03-19_PM", "03-20_AM", "03-20_PM", "03-21_AM", "03-21_PM", "03-22_AM", "03-22_PM", "03-23_AM", "03-23_PM"]
# days = ["05-11_AM", "05-12_AM", "05-13_AM"]
# days = ["03-15_AM", "03-16_AM", "03-17_AM", "03-18_AM", "03-19_AM", "03-20_AM", "03-21_AM"]
# days = ["03-05_AM", "03-06_AM", "03-07_AM", "03-08_AM", "03-09_AM", "03-10_AM", "03-11_AM"]
days = ["03-20_AM"]
# days = ["03-13_AM", "03-14_AM", "03-15_AM", "03-16_AM", "03-17_AM", "03-18_AM", "03-20_AM", "03-21_AM"]
debug = False
iter = True


if __name__ == "__main__":

    dataset = "lyon2"

    for day in days:
        distance_threshold = 3
        branch_number_threshold = 1

        xyz_ori, connection = read_graph(skel_path.format(dataset, day, day))
        xyz_ori = np.array(xyz_ori)
        connection = np.array(connection)

        path_format = "../data/{}/{}/"
        if not os.path.exists(path_format.format(dataset, day)):
            os.makedirs(path_format.format(dataset, day))
        xyz_labeled, branch_connection = skeleton_segmentation(xyz_ori,
                                                               connection,
                                                               day,
                                                               dataset,
                                                               verbose=True,
                                                               visualize=True)
