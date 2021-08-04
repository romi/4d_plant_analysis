import copy
from utils import plot_skeleton
import numpy as np
from dijkstar import Graph, find_path
import open3d
from matplotlib import pyplot as plt
from queue import PriorityQueue, Queue
from collections import deque
import networkx as nx


def get_sigma(center, centers, h):
    r = centers - center
    r2 = np.einsum('ij,ij->i', r, r)
    thetas = np.exp((-r2) / ((h / 2) ** 2))
    cov = np.einsum('j,jk,jl->kl', thetas, r, r)
    values, vectors = np.linalg.eig(cov)
    if np.iscomplex(values).any():
        values = np.real(values)
        vectors = np.real(vectors)
        vectors_norm = np.sqrt(np.einsum('ij,ij->i', vectors, vectors))
        vectors = vectors / vectors_norm
    sorted_indices = np.argsort(-values)
    values_sorted = values[sorted_indices]
    vectors_sorted = vectors[:, sorted_indices]
    sigma = values_sorted[0] / np.sum(values_sorted)
    return sigma, vectors_sorted


def get_point_label(xyz, NN_indices, NN_distance, number_NN=5, sigma_threshold=0.9, visualize_label=False):
    h = NN_distance[:, 1: 1 + number_NN].mean()
    sigmas = np.zeros(xyz.shape[0])
    vectors = np.zeros((xyz.shape[0], 3, 3))
    for i in range(xyz.shape[0]):
        center = xyz[i]
        neighbors = xyz[NN_indices[i, 1:1 + number_NN]]
        sigmas[i], vectors[i, :, :] = get_sigma(center, neighbors, h)
    if visualize_label:
        color_value = np.zeros(xyz.shape)
        color_value[:, 0] = 0.9 * (sigmas > sigma_threshold)
        color_value[:, 1] = 0.7 * (sigmas <= sigma_threshold)
        pcd = open3d.geometry.PointCloud()
        pcd.points = open3d.utility.Vector3dVector(xyz)
        pcd.colors = open3d.utility.Vector3dVector(color_value)

        balls = []
        for xyz_, s in zip(xyz, sigmas):
            if s <= sigma_threshold:
                b = open3d.geometry.TriangleMesh.create_sphere(radius=2)
                b.translate(xyz_)
                b.vertex_colors = open3d.utility.Vector3dVector([0.2, 0.8, 0.1] * np.ones(np.asarray(b.vertices).shape))
                balls.append(b)

        open3d.visualization.draw_geometries([pcd] + balls)
    labels = sigmas > sigma_threshold
    return labels, vectors


def connect_pcd(xyz, threshold, NN_indices, NN_distance, visualize_cluster=False):
    connection_total = []
    nodes = list(range(xyz.shape[0]))
    nodes.sort(key=lambda n: xyz[n, 2])
    not_visited = set(nodes)
    clusters = []

    while len(not_visited) > 0:
        # Pick a random node
        start = not_visited.pop()
        current = start
        cl = [start]
        cl_connect = []
        while True:
            nn_list = NN_indices[current, 1:]
            nn_distance_list = NN_distance[current, 1:]
            next = -1  # set the next node to -1

            next_dist = 10 ** 5
            for nn, nn_dist in zip(nn_list, nn_distance_list):
                if nn in not_visited:
                    next = nn
                    next_dist = nn_dist
                    break

            if next < 0:
                connection_total.append(cl_connect)
                clusters.append(cl)
                break

            if next_dist < threshold:
                cl_connect.append([min(current, next), max(current, next)])
                not_visited.remove(next)
                to_visit = len(not_visited)
                cl.append(next)
                current = next
            else:
                connection_total.append(cl_connect)
                clusters.append(cl)
                break
    if visualize_cluster:
        color_value = np.zeros(xyz.shape)
        for i, cl in enumerate(clusters):
            c = i % 3
            color_value[cl, c] = (i // 3) / 5
        pcd = open3d.geometry.PointCloud()
        pcd.points = open3d.utility.Vector3dVector(xyz)
        pcd.colors = open3d.utility.Vector3dVector(color_value)
        open3d.visualization.draw_geometries([pcd])
    return connection_total, clusters


def connect_pcd_branch(xyz, labels, vectors, threshold, NN_indices, NN_distance, number_NN=2, plot_branch=False):
    nodes = np.array(list(range(xyz.shape[0])))
    connections = set([])
    clusters = []
    nodes_branches = set(nodes[labels])
    adjency = {}
    for n in nodes:
        neighbors = NN_indices[n, 1:1 + number_NN]
        for i, nn in enumerate(neighbors):
            if nn in nodes_branches and NN_distance[n, 1 + i] < threshold:
                connection = (min(n, nn), max(nn, n))
                if connection not in connections:
                    connections.add(connection)
                    if n not in adjency:
                        adjency[n] = [nn]
                    else:
                        adjency[n].append(nn)

    graph_nodes = list(adjency.keys())
    graph_nodes.sort(key=lambda n: -xyz[n, 2])

    graph = nx.Graph()
    graph.add_nodes_from(graph_nodes)
    graph.add_edges_from(list(connections))
    clusters = list(nx.connected_components(graph))
    clusters_connection = []
    for cl_node in clusters:
        cl = graph.subgraph(cl_node)
        cl_connection = list(cl.edges)
        cl_connection = [list(c) for c in cl_connection]
        if plot_branch:
            plot_skeleton(xyz, np.array(cl_connection), plot_point=False)
        clusters_connection.append(cl_connection)
    return clusters, clusters_connection


def try_extend_branch(next, orientation_vec, xyz, not_visited, branch, labels, threshold, NN_indices, NN_distance,
                      connection, number_NN=10):
    while len(not_visited) > 0:
        not_visited.remove(next)
        if not labels[next]:
            break
        branch.append(next)
        current = next
        current_xyz = xyz[current]
        neighbors_xyz = xyz[NN_indices[current, 1:1 + number_NN]]
        neighbors_dist = NN_distance[current, 1:1 + number_NN]
        r = neighbors_xyz - current_xyz
        orientation_correspondence = r.dot(orientation_vec.T) / neighbors_dist
        next_candidate = []
        for i, oc in enumerate(orientation_correspondence):
            if 0.7 < oc < 1.0 and NN_indices[current, 1 + i] in not_visited and NN_distance[current, 1 + i] < threshold:
                next_candidate.append(NN_indices[current, 1 + i])
        if len(next_candidate) > 0:
            next = next_candidate[0]
            connection.append([current, next])
            orientation_vec = (xyz[next] - current_xyz) / np.linalg.norm(xyz[next] - current_xyz)
        else:
            break
    if next not in branch:
        plot_skeleton(xyz, np.array(connection), plot_point=False)
        branch.append(next)
    return


def get_head_tail(xyz, cl):
    d = 10 ** -5
    head, tail = -1, -1
    for i in range(len(cl)):
        n1 = cl[i]
        for j in range(i + 1, len(cl)):
            n2 = cl[j]
            d12 = np.linalg.norm(xyz[n1, :] - xyz[n2, :])
            if d12 > d:
                head = n1
                tail = n2
                d = d12
    return head, tail


def clean_cluster(xyz, cluster, threshold):
    cleaned_connection = []
    cleaned_cluster = []

    for cl in cluster:
        if len(cl) < 2:
            cleaned_connection.append([])
            continue

        # build a graph with the nodes in the cluster. The edges are established if the distance < max(connection)
        G = Graph(undirected=True)
        head, tail = get_head_tail(xyz, cl)

        for i in range(len(cl)):
            for j in range(i + 1, len(cl)):
                if np.linalg.norm(xyz[cl[i], :] - xyz[cl[j], :]) < threshold + 10 ** -5:
                    G.add_edge(cl[i], cl[j], np.linalg.norm(xyz[cl[i], :] - xyz[cl[j], :]))
        path = find_path(G, head, tail)
        cleaned_cl = list(path.nodes)
        cleaned_cl_connect = [[cleaned_cl[i], cleaned_cl[i + 1]] for i in range(len(cleaned_cl) - 1)]

        cleaned_connection.append(cleaned_cl_connect)
        cleaned_cluster.append(cleaned_cl)

    return cleaned_connection, cleaned_cluster


def get_distance_cluster(xyz, cl1, cl2):
    d = 10 ** 5
    n1_opt, n2_opt = -1, -1
    for n1 in cl1:
        for n2 in cl2:
            d12 = np.linalg.norm(xyz[n1, :] - xyz[n2, :])
            if d12 < d:
                d = d12
                n1_opt = n1
                n2_opt = n2
    return d, n1_opt, n2_opt


def merge_small_clusters(xyz, cluster, threshold):
    large_cluster = []
    small_cluster = []
    merged_cluster = []
    merged_connection = []

    mean = sum([len(cl) for cl in cluster]) / len(cluster)

    for i in range(len(cluster)):
        if len(cluster[i]) <= 5:
            small_cluster.append(i)
        else:
            large_cluster.append(i)

    for i in small_cluster:
        shortest_dist = 10 ** 5
        n1_opt, n2_opt = -1, -1
        j_opt = -1
        for j in range(len(cluster)):
            if i == j:
                continue
            d, n1, n2 = get_distance_cluster(xyz, cluster[i], cluster[j])
            if d < shortest_dist:
                shortest_dist = d
                j_opt = j
        if shortest_dist < threshold:
            cluster[j_opt] += cluster[i]
        cluster[i] = []

    return [cl for cl in cluster if len(cl) > 0]


def merge_all_clsuter(xyz, cluster_connection, cluster):
    while len(cluster) > 1:
        shortest_dist = 10 ** 5
        n1_opt, n2_opt, i_opt, j_opt = -1, -1, -1, -1
        for i in range(len(cluster)):
            for j in range(i + 1, len(cluster)):
                d, n1, n2 = get_distance_cluster(xyz, cluster[i], cluster[j])
                if d < shortest_dist:
                    shortest_dist = d
                    n1_opt = n1
                    n2_opt = n2
                    i_opt = i
                    j_opt = j
        # print(shortest_dist, len(cluster))
        if i_opt >= 0 and j_opt >= 0:
            cluster[i_opt] = cluster[i_opt] + cluster[j_opt]
            cluster_connection.append([n1_opt, n2_opt])
            cluster.pop(j_opt)
        else:
            print("ERROR")
    return cluster_connection, cluster


def remove_cycle(xyz, connection, clusters):
    graph = nx.Graph()
    for c in connection:
        graph.add_edge(c[0], c[1], weight=np.linalg.norm(xyz[c[0]] - xyz[c[1]]))
    graph_nodes = np.unique(connection)
    for nn in graph_nodes:
        graph.nodes[nn]['pos'] = tuple(xyz[nn])
    shortest_spam_tree = nx.minimum_spanning_tree(graph)
    connection = [list(c) for c in list(shortest_spam_tree.edges)]
    return connection
