import numpy as np
import open3d
import random
import matplotlib.pyplot as plt
from dijkstar import Graph, find_path
from scipy import sparse
import sys


def angles_of_triangles(V, T):
    L1 = np.linalg.norm(V[T[:, 1]] - V[T[:, 2]], axis=1)
    L2 = np.linalg.norm(V[T[:, 0]] - V[T[:, 2]], axis=1)
    L3 = np.linalg.norm(V[T[:, 0]] - V[T[:, 1]], axis=1)
    A1 = ((L2**2 + L3**2 - L1**2)/(2*L2*L3)).reshape( (-1,1) )
    A2 = ((L1**2 + L3**2 - L2**2)/(2*L1*L3)).reshape( (-1,1) )
    A3 = ((L1**2 + L2**2 - L3**2)/(2*L2*L1)).reshape( (-1,1) )
    A = np.hstack([A1, A2, A3])
    A = np.arccos(A)
    return A


def fem_area_mat(vertices, faces, faces_areas=None):
    """
    Compute the area matrix for mesh functional_map_ using finite elements method.

    Entry (i,i) is 1/6 of the sum of the area of surrounding triangles
    Entry (i,j) is 1/12 of the sum of the area of triangles using edge (i,j)

    Parameters
    -----------------------------
    vertices   : (n,3) array of vertices coordinates
    faces      : (m,3) array of vertex indices defining faces
    faces_area : (m,) - Optional, array of per-face area

    Output
    -----------------------------
    A : (n,n) sparse area matrix
    """
    N = vertices.shape[0]

    # Compute face area
    if faces_areas is None:
        v1 = vertices[faces[:,0]]  # (m,3)
        v2 = vertices[faces[:,1]]  # (m,3)
        v3 = vertices[faces[:,2]]  # (m,3)
        faces_areas = 0.5 * np.linalg.norm(np.cross(v2-v1, v3-v1), axis=1)  # (m,)

    # Use similar construction as cotangent weights
    I = np.concatenate([faces[:,0], faces[:,1], faces[:,2]])  # (3m,)
    J = np.concatenate([faces[:,1], faces[:,2], faces[:,0]])  # (3m,)
    S = np.concatenate([faces_areas,faces_areas,faces_areas])  # (3m,)

    In = np.concatenate([I, J, I])  # (9m,)
    Jn = np.concatenate([J, I, I])  # (9m,)
    Sn = 1/12 * np.concatenate([S, S, 2*S])  # (9m,)

    A = sparse.coo_matrix((Sn, (In, Jn)), shape=(N, N)).tocsc()
    return A


def connectivity(T):
    nf = T.shape[0]
    nv = np.max(T)
    E2V = []
    for t in T:
        E2V.append([t[0], t[1]])
        E2V.append([t[1], t[2]])
        E2V.append([t[2], t[0]])
    E2V = np.array(E2V)
    E2V = np.sort(E2V, axis=1)
    E2V = np.unique(E2V, axis=0)
    print(E2V)

    T2E = []
    for t in T:
        e1 = np.sort([t[0], t[1]])
        e2 = np.sort([t[2], t[1]])
        e3 = np.sort([t[0], t[2]])
        ind1 = np.where(np.all(E2V == e1, axis=1))[0][0]
        ind2 = np.where(np.all(E2V == e2, axis=1))[0][0]
        ind3 = np.where(np.all(E2V == e3, axis=1))[0][0]
        T2E.append([ ind1, ind2, ind3 ])
    T2E = np.array(T2E)
    #print(T2E)

    E2T = []
    for i in range(E2V.shape[0]):
        row, col = np.where( T2E == i)
        if len(row) > 0:
            E2T.append( row )
    #print(E2T)

    T2T = []
    for index, t in enumerate(T):
        res = []
        for e in T2E[index]:
            if len(E2T[e]) > 1:
                for t_ in E2T[e]:
                    if t_ != index:
                        res.append(t_)
        T2T.append(np.array(res))
    #print(T2T)
    return E2V, T2E, E2T, T2T


def geodesic_distance_no_branches(G, i, j):
    path = find_path(G, i, j)
    gdist = path.total_cost
    return gdist


def dijkstra_fps(shape, k):
    G = Graph(undirected=True)
    E2V, T2E, E2T, T2T = connectivity(shape.T)
    for e in E2V:
        edge_length = np.linalg.norm(shape.V[e[0], :] - shape.V[e[1], :])
        G.add_edge(e[0], e[1], edge_length)

    return


def plot_semantic_pointcloud(fh, array_list):
    """
    :param array_list: a list of numpy arrays, each array represents a different class
    """
    ax = fh.gca(projection='3d')

    for P in array_list:
        P = np.asarray(P)
        ax.scatter3D(P[:, 0], P[:, 1], P[:, 2], '.')


def point_cloud_to_mesh(pcd, radius_factor=1.0):
    points = np.asarray(pcd.points)
    # if len(points) > maxPoints:
    #     random_indices = random.sample(range(0, len(points)), maxPoints)
    #     points = points[random_indices, :]

    pcd.points = open3d.utility.Vector3dVector(points)
    pcd.estimate_normals()
    # pcd.orient_normals_consistent_tangent_plane(10)
    distances = pcd.compute_nearest_neighbor_distance()
    avg_dist = np.mean(distances)
    radius = radius_factor * avg_dist

    tetra_mesh, pt_map = open3d.geometry.TetraMesh.create_from_point_cloud(pcd)
    alpha = radius
    mesh = open3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(
        pcd, alpha, tetra_mesh, pt_map)
    # radii = [radius, 1.5 * radius, 2 * radius]
    # mesh = open3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
    #     pcd, open3d.utility.DoubleVector(radii))
    mesh = mesh.remove_unreferenced_vertices()
    # mesh.orient_triangles()
    return mesh


def plot_laplacian_basis(basis_functions, mesh):
    figsb1, axes1 = plt.subplots(nrows=5, ncols=3, figsize=(8, 12),
                                 subplot_kw={'projection': '3d'})
    trilist = mesh.trilist
    vertlist = mesh.vertlist
    for i in range(np.size(axes1)):
        colors = np.mean(basis_functions[trilist, i + 0], axis=1)
        colors = 1 * colors
        ax = axes1.flat[i]
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        ax.view_init(elev=70., azim=15.)
        ax.set_aspect('auto')
        trisurfplot = ax.plot_trisurf(vertlist[:, 0], vertlist[:, 1],
                                      vertlist[:, 2], triangles=trilist,
                                      cmap=plt.cm.bwr,
                                      edgecolor='white', linewidth=0.)
        trisurfplot.set_array(colors)
        trisurfplot.set_clim(-1, 1)

    cbar = figsb1.colorbar(trisurfplot, ax=axes1.ravel().tolist(), shrink=0.75,
                           orientation='horizontal', fraction=0.05, pad=0.05,
                           anchor=(0.5, -4.0))

    plt.subplots_adjust(left=0.0, right=1.0, bottom=0.08, top=1.0)
    plt.show()


def get_functional_representation(target_function, basis_function, area, dim):
    """
    :param target_function: (n * n) the descriptor function to be projected
    :param basis_function: (n * n) Laplacian basis functions
    :param area: (n * n) the fem area for each pair of vertices
    :param dim: int
    :return:
    """
    A = basis_function[:, :dim]
    # b = target_function
    residual = [0,0]
    # x, residual, rank, s = np.linalg.lstsq(A, b, rcond=None)

    x = A.T @ area @ target_function

    return x, residual[0]/A.shape[0]


def get_transition_matrix(value_rep_1, value_rep_2, natural_frequencies_1, natural_frequencies_2, alpha, dim):

    C_init = np.random.rand(dim, dim)

    P1 = value_rep_1
    P2 = value_rep_2
    L1 = natural_frequencies_1[:dim]
    L2 = natural_frequencies_2[:dim]

    C = C_init
    A_fixed = P1.dot(P1.T)
    B = P1.dot(P2.T)

    # if np.linalg.cond(A_fixed) < 1 / sys.float_info.epsilon:
    #     C = (np.linalg.inv(A_fixed).dot(B)).T
    # else:
    #     C = (np.linalg.pinv(A_fixed).dot(B)).T

    for i in range(dim):
        A = np.diag( alpha * (L1 -L2[i])**2 ) + A_fixed
        C[i, :] = np.linalg.inv(A).dot(B[:, i])
    residual = np.linalg.norm(C.dot(P1) - P2) + alpha * np.linalg.norm(C.dot(np.diag(L1)) - np.diag(L2).dot(C))
    print("::fitting residual: ", residual/(dim * dim))
    return C


def get_gaussian_index_function( vertlist, n, h):
    res = np.zeros(vertlist.shape[0])
    target = vertlist[n]

    r = vertlist - target
    r2 = np.einsum('ij,ij->i', r, r)
    res = np.exp(-r2 / ((h / 2) ** 2))
    res[res < 1**-5] = 0
    return res


def save_off(filename, VPos, VColors, ITris):
    """
    Save a .off file
    Parameters
    ----------
    filename: string
        Path to which to write .off file
    VPos : ndarray (N, 3)
        Array of points in 3D
    VColors : ndarray(N, 3)
        Array of RGB colors
    ITris : ndarray (M, 3)
        Array of triangles connecting points, pointing to vertex indices
    """
    nV = VPos.shape[0]
    nF = ITris.shape[0]
    fout = open(filename, "w")
    if VColors.size == 0:
        fout.write("OFF\n%i %i %i\n"%(nV, nF, 0))
    else:
        fout.write("COFF\n%i %i %i\n"%(nV, nF, 0))
    for i in range(nV):
        fout.write("%g %g %g"%tuple(VPos[i, :]))
        if VColors.size > 0:
            fout.write(" %g %g %g"%tuple(VColors[i, :3]))
        fout.write("\n")
    for i in range(nF):
        fout.write("3 %i %i %i\n"%tuple(ITris[i, :]))
    fout.close()


def saveColors(filename, VPos, hks, ITris, cmap = 'gray'):
    """
    Save the mesh as a .coff file using a divergent colormap, where
    negative curvature is one one side and positive curvature is on the other
    """
    c = plt.get_cmap(cmap)
    x = (hks - np.min(hks, axis=0))
    x /= np.max(x, axis=0)
    x = np.array(np.round(x*255.0), dtype=np.int32)
    C = x
    C = C[:, 0:3]
    save_off(filename, VPos, C, ITris)


def pairwise_distance(a, b):
    """
    :param a: array
    :param b: array
    :return: out: numpy.ndarray, where out[i, j] = || a[i] - b[j] ||**2
    """
    ab = a[:, None, :] - b
    out = np.einsum('ijk,ijk->ij', ab, ab)
    return out


def visualization_basis_function(basis_functions, mesh):
    trilist = mesh.trilist
    vertlist = mesh.vertlist
    figsb1, axes1 = plt.subplots(nrows=2, ncols=3, figsize=(20, 15),
                                 subplot_kw={'projection': '3d'})
    for i in range(np.size(axes1)):
        colors = np.mean(basis_functions[trilist, i + 0], axis=1)
        colors = 1 * colors
        ax = axes1.flat[i]
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        ax.view_init(elev=70., azim=15.)
        ax.set_aspect('auto')
        trisurfplot = ax.plot_trisurf(vertlist[:, 0], vertlist[:, 1],
                                      vertlist[:, 2], triangles=trilist,
                                      cmap=plt.cm.bwr,
                                      edgecolor='white', linewidth=0.)
        trisurfplot.set_array(colors)
        trisurfplot.set_clim(-1, 1)

    cbar = figsb1.colorbar(trisurfplot, ax=axes1.ravel().tolist(), shrink=0.75,
                           orientation='horizontal', fraction=0.05, pad=0.05,
                           anchor=(0.5, -4.0))

    plt.subplots_adjust(left=0.0, right=1.0, bottom=0.08, top=1.0)
    plt.show()


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

def get_neighbors(cloud, radius):
    """

    :param cloud: open3d point cloud
    :param radius: the radius defining neighbors
    :return: the index of neighbors
    """
    tree = open3d.geometry.KDTreeFlann(cloud)
    neighbors = []
    for pi, point in enumerate(cloud.points):
        cnt, idxs, dists = tree.search_radius_vector_3d(point, radius)
        idxs.remove(pi) # remove the point itself
        neighbors.append(idxs)
    return neighbors


def plot_skeleton(V, E, plot_point=False):
    v_skel = V[np.unique(E)]
    pcd = open3d.geometry.PointCloud()
    pcd.points = open3d.utility.Vector3dVector(v_skel)
    pcd.colors = open3d.utility.Vector3dVector([0.1, 0.9, 0.1] * np.ones(v_skel.shape))

    line_set = open3d.geometry.LineSet()
    line_set.points = open3d.utility.Vector3dVector(V)
    line_set.lines = open3d.utility.Vector2iVector(E)
    if not plot_point:
        open3d.visualization.draw_geometries([line_set])
    else:
        pcd.points = open3d.utility.Vector3dVector(V[::, :])
        pcd.colors = open3d.utility.Vector3dVector([0.3, 0.8, 0.2] * np.ones(V[::, :].shape))
        open3d.visualization.draw_geometries([line_set, pcd])


def plot_skeleton_matching(xyz_1, xyz_2, connect_1, connect_2):
    p1 = open3d.geometry.PointCloud()
    p1.points = open3d.utility.Vector3dVector(xyz_1)
    l1 = open3d.geometry.LineSet()
    l1.points = open3d.utility.Vector3dVector(xyz_1)
    l1.lines = open3d.utility.Vector2iVector(connect_1)

    p2 = open3d.geometry.PointCloud()
    p2.points = open3d.utility.Vector3dVector(xyz_2)
    l2 = open3d.geometry.LineSet()
    l2.points = open3d.utility.Vector3dVector(xyz_2)
    l2.lines = open3d.utility.Vector2iVector(connect_2)

    open3d.visualization.draw_geometries([p1, p2, l1, l2])


def save_graph(filename, V, E, matlab_type=False):
    with open(filename, "w") as file:
          for i in range(V.shape[0]):
                file.write("v " + " ".join( [str(x) for x in list(V[i])]) + "\n")
          for i in range(len(E)):
                file.write("e " + " ".join([ str(x) for x in E[i] ]) + "\n")


def read_graph(filename):
    """ Read a graph from a text file as a skeleton
    """
    # read all vertices and edges
    with open(filename, "r") as file:
        vertices = []
        edges = []
        labels = []
        for line in file:
            data = line.split()
            if data[0] == 'v':
                v = np.array([float(data[1]), float(data[2]), float(data[3])])
                vertices.append(v)
                if len(data) > 4:
                    l = int(float(data[4]))
                    labels.append(l)
            if data[0] == 'e':
                e = np.array([int(data[1]), int(data[2])], dtype=np.uint32)
                edges.append(e)

    XYZ = np.stack(vertices)
    return XYZ, edges

if __name__ == "__main__":
    V = np.array([[0,0,0], [0,1,0], [1,0,0], [1,1,1], [1,1.5,2]])
    T = np.array([[0,1,2], [0,1,3], [1,2,4]])
    connectivity(T)
 #   print( angles_of_triangles(V, T) )
