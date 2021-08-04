import copy
import time
import numpy as np
import networkx as nx
import open3d
from sklearn.neighbors import KDTree


# def get_skeleton_landmarks(organ):
#
#     skeleton_connect = organ["skeleton_connection"]
#     segment_index = organ["segment_index"]
#     skeleton_xyz = organ["skeleton_pcd"]
#
#     graph = nx.Graph()
#     graph.add_edges_from(skeleton_connect)
#     degree = dict(graph.degree)
#     stem = -1
#     for k in degree.keys():
#         if k not in segment_index:
#             stem = k
#     if stem == -1:
#         print("error stem not found")
#         return
#
#     leaves_seg = [k for k in degree.keys() if degree[k] == 1 and k in segment_index]
#     leaves_seg.sort(key=lambda x: len(nx.shortest_path(graph, stem, x)))
#     skeleton_landmarks = []
#
#     for seg in leaves_seg:
#         xyz_seg = skeleton_xyz[skeleton_xyz[:, 3] == seg]
#         skeleton_landmarks.append(xyz_seg[-1, :3])
#     return np.vstack(skeleton_landmarks)

def get_skeleton_landmarks_pairs(xyz_1, xyz_2, connect_1, connect_2, visualize=False, params=None, verbose=False):
    if not params:
        params = {}

    if 'weight_e' not in params:
        params['weight_e'] = 10.0

    # use semantic labels
    if 'use_labels' not in params:
        params['use_labels'] = False

    # apply label penalty or not
    if 'label_penalty' not in params:
        params['label_penalty'] = False

    # show debug msgs/vis
    if 'debug' not in params:
        params['debug'] = False

    if 'match_ends_to_ends' not in params:
        params['match_ends_to_ends'] = True

    S1 = {"xyz": copy.deepcopy(xyz_1[:, :3]), "connect": connect_1}
    S2 = {"xyz": copy.deepcopy(xyz_2[:, :3]), "connect": connect_2}

    S2["xyz"] = S2["xyz"] + (S1["xyz"][-1] - S2["xyz"][-1])

    # print("HMM: Computing Transition and Emission probabilities")
    if verbose:
        print("Start defining the skeleton HMM setting")
        start_time = time.time()
    T1, E1, statenames1 = define_skeleton_matching_hmm(S1, S2, params, verbose=verbose)
    if verbose:
        end_time = time.time()
        print("Finished defining the skeleton HMM setting")
        print("Used : {} s".format(end_time - start_time))
        print(64 * "-")

    # Transform T and E to probability
    to_prob = lambda x: 1 / (x + 10 ** -5)
    T1 = to_prob(T1)
    E1 = to_prob(E1)

    # compute correspondence pairs using viterbi
    V = np.array(get_sequence(S1))
    best_seq = viterbi(V, T1, E1, statenames1)
    if verbose:
        print("Finished viterbi")
    corres = get_correspondences_from_seq(best_seq)

    # remove all matchings to virtual 'nothing' node
    ind_remove = np.where(corres[:, 1] == -1)
    corres = np.delete(corres, ind_remove, 0)

    # post process
    corres = remove_double_matches_in_skeleton_pair(S1, S2, corres)

    lm1 = xyz_1[corres[:, 0]]
    lm2 = xyz_2[corres[:, 1]]

    if visualize:
        pcd = open3d.geometry.PointCloud()
        pcd.points = open3d.utility.Vector3dVector(np.vstack([lm1[:, :3], lm2[:, :3]]))

        lineset = open3d.geometry.LineSet()
        lineset.points = open3d.utility.Vector3dVector(np.vstack([lm1[:, :3], lm2[:, :3]]))
        connect = [[i, i + len(lm1)] for i in range(len(lm1))]
        lineset.lines = open3d.utility.Vector2iVector(connect)
        open3d.visualization.draw_geometries([pcd, lineset])
    return lm1, lm2


def define_skeleton_matching_hmm(S1, S2, params, verbose=False):
    # define statenames
    statenames = define_statenames(S1, S2)

    # Precompute geodesic distances
    GD1, NBR1 = compute_geodesic_distance_on_skeleton(S1)
    GD2, NBR2 = compute_geodesic_distance_on_skeleton(S2)

    # Precompute euclidean distances between all pairs
    ED = compute_euclidean_distance_between_skeletons(S1, S2)

    # compute transition matrix
    T = compute_transition_matrix(S1, S2, GD1, NBR1, GD2, NBR2, ED, verbose=verbose)

    # compute emission matrix
    E = compute_emission_matrix(S1, S2, ED, params)

    return T, E, statenames


def define_statenames(S1, S2):
    # number of nodes in each skeleton
    N = S1["xyz"].shape[0]
    M = S2["xyz"].shape[0]

    statenames = [[-2, -2]]
    # starting state
    for n1 in range(N):
        for m1 in range(M):
            statenames.append([n1, m1])
    for n1 in range(N):
        statenames.append([n1, -1])

    return statenames


def compute_geodesic_distance_on_skeleton(S):
    G = nx.Graph(undirected=True)
    edges = S["connect"]
    for e in edges:
        edge_length = np.linalg.norm(S["xyz"][e[0], :] - S["xyz"][e[1], :])
        G.add_edge(e[0], e[1], weight=edge_length)

    N = S["xyz"].shape[0]
    GD = np.zeros([N, N])
    NBR = np.zeros([N, N])
    for i in range(N):
        for j in range(i + 1, N):
            path = nx.shortest_path(G, i, j, weight="weight")
            nbr = len(path)
            gdist = nx.shortest_path_length(G, i, j, weight="weight")
            GD[i, j] = gdist
            GD[j, i] = gdist
            NBR[i, j] = nbr
            NBR[j, i] = nbr

    return GD, NBR


def compute_euclidean_distance_between_skeletons(S1, S2):
    N = S1["xyz"].shape[0]
    M = S2["xyz"].shape[0]

    D = np.zeros([N, M])
    for n1 in range(N):
        for m1 in range(M):
            D[n1, m1] = np.linalg.norm(S1["xyz"][n1, :] - S2["xyz"][m1, :])
    return D


def compute_transition_matrix(S1, S2, GD1, NBR1, GD2, NBR2, ED, verbose=False):
    N = S1["xyz"].shape[0]
    M = S2["xyz"].shape[0]
    T = 1e6 * np.ones([N * M + N, N * M + N], dtype=np.float64)

    G1 = nx.Graph(undirected=True)
    edges = S1["connect"]
    for e in edges:
        edge_length = np.linalg.norm(S1["xyz"][e[0], :] - S1["xyz"][e[1], :])
        G1.add_edge(e[0], e[1], weight=edge_length)

    G2 = nx.Graph(undirected=True)
    edges = S2["connect"]
    for e in edges:
        edge_length = np.linalg.norm(S2["xyz"][e[0], :] - S2["xyz"][e[1], :])
        G2.add_edge(e[0], e[1], weight=edge_length)

    mean_dist_S1 = mean_distance_direct_neighbors(S1, GD1)
    max_cost_normal_pairs = 0

    # normal pair to normal pair:
    # (n1,m1) first pair, (n2,m2) second pair
    max_cost_normal_pairs = 0
    for n1 in range(N):
        if verbose:
            print("{}/{}".format(n1, N))
        for m1 in range(M):
            if verbose and m1 % 10 == 0:
                print("--{}/{}".format(m1, M))
            for n2 in range(N):
                for m2 in range(M):
                    # Avoid going in different directions on the skeleton. Then the
                    # geodesic difference can be small, but actually one would assume a
                    # large cost
                    v_inS1 = S1["xyz"][n2, :] - S1["xyz"][n1, :]
                    v_inS2 = S2["xyz"][m2, :] - S2["xyz"][m1, :]

                    # angle between vectors smaller 90 degrees
                    if np.dot(v_inS1, v_inS2) >= 0:
                        # geodesic distance and difference in number of branches along the way
                        g1 = GD1[n1, n2]
                        g2 = GD2[m1, m2]
                        br1 = NBR1[n1, n2]
                        br2 = NBR2[m1, m2]
                        d_degree1 = G1.degree[n2] - G1.degree[n1]
                        d_degree2 = G2.degree[m2] - G2.degree[m1]
                        v = 2 * np.abs(br1 - br2) * np.max(GD1) + 2 * np.abs(g1 - g2) + mean_dist_S1 + \
                            4 * np.abs(d_degree1 - d_degree2)
                        T[n1 * M + m1, n2 * M + m2] = v

                        if v > max_cost_normal_pairs:
                            max_cost_normal_pairs = v

    # Main diagonal should be large
    for i in range(N * M):
        T[i, i] = max_cost_normal_pairs

    # Normal pair -> not present
    for n1 in range(N):
        for m1 in range(M):
            for n2 in range(N):
                if n2 != n1:
                    T[n1 * M + m1, N * M + n2] = 1 / 8 * np.max(GD1)

    # Not present -> normal pair
    for n2 in range(N):
        for m1 in range(M):
            for n1 in range(N):
                T[N * M + n2, n1 * M + m1] = max_cost_normal_pairs

    # Not present -> not present
    S1_seq = get_sequence(S1)
    for n1 in range(N):
        for n2 in range(N):
            pos_n1 = S1_seq.index(n1) if n1 in S1_seq else -1
            pos_n2 = S1_seq.index(n2) if n2 in S1_seq else -1
            if pos_n2 > pos_n1:
                v = GD1[n1, n2]
                T[N * M + n1, N * M + n2] = v + mean_dist_S1 + np.min(ED[n1, :])

    # Add starting state (which will not be reported in hmmviterbi)
    sizeT = N * M + N;
    T1 = np.hstack((1e6 * np.ones((1, 1)), np.ones((1, N * M)), 1e6 * np.ones((1, N))))
    T2 = np.hstack((1e6 * np.ones((sizeT, 1)), T))
    T = np.vstack((T1, T2))

    return T


def mean_distance_direct_neighbors(S, GD):
    N = S["xyz"].shape[0]
    sumd = 0
    no = 0
    for i in range(N):
        for j in range(N):
            if [i, j] in S["connect"] or [j, i] in S["connect"]:  # A means adjacency matrix
                sumd = sumd + GD[i, j]
                no = no + 1

    mean_dist = sumd / no
    return mean_dist


def compute_emission_matrix(S1, S2, ED, params):
    # emmision matrix (state to sequence)
    # Here: degree difference + euclidean difference inside a pair
    N = S1["xyz"].shape[0]
    M = S2["xyz"].shape[0]
    E = 1e05 * np.ones((N * M + N, N))

    G1 = nx.Graph(undirected=True)
    edges = S1["connect"]
    for e in edges:
        edge_length = np.linalg.norm(S1["xyz"][e[0], :] - S1["xyz"][e[1], :])
        G1.add_edge(e[0], e[1], weight=edge_length)

    G2 = nx.Graph(undirected=True)
    edges = S2["connect"]
    for e in edges:
        edge_length = np.linalg.norm(S2["xyz"][e[0], :] - S2["xyz"][e[1], :])
        G2.add_edge(e[0], e[1], weight=edge_length)

    for n1 in range(N):
        for m1 in range(M):

            degree1 = G1.degree[n1]
            degree2 = G2.degree[m1]

            # Do not penalize end node against middle node
            if not params['match_ends_to_ends'] and (
                    (degree1 == 1 and degree2 == 2) or (degree1 == 2 and degree2 == 1)):
                E[n1 * M + m1, n1] = params['weight_e'] * (ED[n1, m1] + 10e-10)
            else:
                E[n1 * M + m1, n1] = np.abs(degree1 - degree2) + params['weight_e'] * (ED[n1, m1] + 10e-10)

    # No match
    for n1 in range(N):
        # Take the  best
        I = np.argsort(E[0:N * M, n1])
        E[N * M + n1, n1] = E[I[0], n1]

    # Add starting state (which will not be reported in hmmviterbi)
    E = np.vstack((1e10 * np.ones((1, N)), E))

    return E


def get_sequence(S):
    G = nx.Graph()
    edges = S["connect"]
    for e in edges:
        edge_length = np.linalg.norm(S["xyz"][e[0], :] - S["xyz"][e[1], :])
        G.add_edge(e[0], e[1], weight=edge_length)

    dfs_edges = list(nx.dfs_edges(G, source=S["xyz"].shape[0] - 1))
    sequence = [dfs_edges[0][0]]
    for i in range(len(dfs_edges)):
        sequence.append(dfs_edges[i][1])
    return sequence


def viterbi(V, T, E, StateNames):
    """
    This function computes a sequence given observations, transition and emission prob. using the Viterbi algorithm.


    Parameters
    ----------
    V : numpy array (Mx1)
        Observations
    T : numpy array (NxN)
        Transition probabilities
    E : numpy array (NxM)
        Emission probabilities
    StateNames : list
                  ames for each state used in the HMM

    Returns
    -------
    S :  list
        Best sequence of hidden states

    """
    M = V.shape[0]
    N = T.shape[0]

    omega = np.zeros((M, N))
    omega[0, :] = np.log(E[:, V[0]])

    prev = np.zeros((M - 1, N))

    for t in range(1, M):
        for j in range(N):
            # Same as Forward Probability
            probability = omega[t - 1] + np.log(T[:, j]) + np.log(E[j, V[t]])

            # This is our most probable state given previous state at time t (1)
            prev[t - 1, j] = np.argmax(probability)

            # This is the probability of the most probable state (2)
            omega[t, j] = np.max(probability)

        # Path Array
        S_ = np.zeros(M)

        # Find the most probable last hidden state
        last_state = np.argmax(omega[M - 1, :])

        S_[0] = last_state

    backtrack_index = 1
    for i in range(M - 2, -1, -1):
        S_[backtrack_index] = prev[i, int(last_state)]
        last_state = prev[i, int(last_state)]
        backtrack_index += 1

    # Flip the path array since we were backtracking
    S_ = np.flip(S_, axis=0)

    # Convert numeric values to actual hidden states
    S = []
    for s in S_:
        S.append(StateNames[int(s)])
    return S


def get_correspondences_from_seq(seq):
    N = len(seq)
    corres = np.zeros((N, 2), dtype=np.int64)
    for i in range(N):
        corres[i, 0] = seq[i][0]
        corres[i, 1] = seq[i][1]

    return corres


def remove_double_matches_in_skeleton_pair(S1, S2, corres):
    """
    Remove double matches of two skeletons and keeps only the one with the smallest distance.

    Parameters
    ----------
    S1, S2 : Skeleton Class
             Two skeletons for which we compute the correspondences
    corres : numpy array (Mx2)
             correspondence between two skeleton nodes pontetially with one-to-many matches

    Returns
    -------
    corres: numpy array
            one-to-one correspondences between the skeleton nodes.

    """
    num_corres = corres.shape[0]
    distances = np.zeros((num_corres, 1))
    for i in range(num_corres):
        distances[i] = np.linalg.norm(S1["xyz"][corres[i, 0], :] - S2["xyz"][corres[i, 1], :])

    # remove repeated corres 1 -> 2
    corres12, counts12 = np.unique(corres[:, 0], return_counts=True)
    ind_remove12 = []
    for i in range(len(corres12)):
        if counts12[i] > 1:
            ind_repeat = np.argwhere(corres[:, 0] == corres12[i])
            dist_repeat = distances[ind_repeat]
            ind_ = np.argsort(dist_repeat)[1:]
            ind_remove12.append(ind_repeat[ind_])
    corres = np.delete(corres, ind_remove12, axis=0)
    distances = np.delete(distances, ind_remove12, axis=0)

    # remove repeated corres 2 -> 1
    corres21, counts21 = np.unique(corres[:, 1], return_counts=True)
    ind_remove21 = []
    for i in range(len(corres21)):
        if counts21[i] > 1:
            ind_repeat = np.argwhere(corres[:, 1] == corres21[i]).flatten()
            dist_repeat = distances[ind_repeat].flatten()
            ind_ = np.argsort(dist_repeat)[1:]
            ind_r = ind_repeat[ind_]
            for i_r in ind_r:
                ind_remove21.append(i_r)
    corres = np.delete(corres, ind_remove21, axis=0)
    distances = np.delete(distances, ind_remove21, axis=0)

    return corres
