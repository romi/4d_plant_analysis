import time
import scipy as sp
from scipy.sparse.linalg import lsqr
import numpy as np
from tqdm import tqdm
import open3d
from point_cloud_clean import clean_point_cloud
from utils import point_cloud_to_mesh
from functional_map_.laplacian import cotangent_weights, get_faces_area, get_vertex_area
import robust_laplacian


def averageFaceArea(mesh):
    vertices = np.asarray(mesh['vertices'])
    faces = np.asarray(mesh['faces'])
    return np.mean(get_faces_area(vertices, faces))


def getOneRingAreas(mesh):
    # Collect areas of faces adjacent to each vertex
    vertices = np.asarray(mesh['vertices'])
    faces = np.asarray(mesh['faces'])
    vertex_areas = get_vertex_area(vertices, faces)

    return vertex_areas


def contract(mesh, epsilon=1e-06, iter_lim=10, time_lim=None, precision=1e-07,
             SL=2, WH0=1, WL0='auto', operator='cotangent', progress=True, pcd_laplacian=False, verbose=False):
    assert operator in ('cotangent', 'umbrella')
    start = time.time()

    # Force into trimesh
    m = {"vertices": np.asarray(mesh.vertices), "faces": np.asarray(mesh.triangles)}
    n = len(m["vertices"])

    # Initialize attraction weights
    zeros = np.zeros((n, 3))
    WH0_diag = np.zeros(n)
    WH0_diag.fill(WH0)
    WH0 = sp.sparse.spdiags(WH0_diag, 0, WH0_diag.size, WH0_diag.size)
    # Make a copy but keep original values
    WH = sp.sparse.dia_matrix(WH0)

    # Initialize contraction weights
    if WL0 == 'auto':
        WL0 = 1e-03 * np.sqrt(averageFaceArea(m))

    WL_diag = np.zeros(n)
    WL_diag.fill(WL0)
    WL = sp.sparse.spdiags(WL_diag, 0, WL_diag.size, WL_diag.size)

    # Copy mesh
    dm = m.copy()
    m['area'] = np.sum(get_vertex_area(m['vertices'], m['faces']))
    area_ratios = [1.0]
    originalRingAreas = getOneRingAreas(dm)
    for i in range(iter_lim):
        # Get Laplace weights
        if pcd_laplacian:
            L, M = robust_laplacian.point_cloud_laplacian(m['vertices'])
        else:
            L, M = robust_laplacian.point_cloud_laplacian(m['vertices'], m['faces'])
        V = dm['vertices']
        A = sp.sparse.vstack([WL.dot(L), WH])
        b = np.vstack((zeros, WH.dot(V)))

        cpts = np.zeros((n, 3))
        for j in range(3):
            """
            # Solve A*x = b
            # Note that we force scipy's lsqr() to use current vertex
            # positions as start points - this speeds things up and
            # without it we get suboptimal solutions that lead to early
            # termination
            cpts[:, j] = lsqr(A, b[:, j],
                              atol=precision, btol=precision,
                              damp=1,
                              x0=dm.vertices[:, j])[0]
            """
            x0 = dm['vertices'][:, j]
            # Compute residual vector
            r0 = b[:, j] - A * x0
            # Use LSQR to solve the system
            dx = lsqr(A, r0,
                      atol=precision, btol=precision,
                      damp=1)[0]
            # Add the correction dx to obtain a final solution
            cpts[:, j] = x0 + dx

        # Update mesh with new vertex position
        dm['vertices'] = cpts

        # Break if face area has increased compared to the last iteration
        dm['area'] = np.sum(get_vertex_area(dm['vertices'], dm['faces']))
        if verbose:
            print(":: iter{}, area ratio compared to the initial one: {}".format(i, area_ratios[-1]))

        area_ratios.append(dm['area'] / m['area'])
        if area_ratios[-1] < 0.10:
            break

        WL = sp.sparse.dia_matrix(WL.multiply(SL))

        changeinarea = np.sqrt(originalRingAreas / getOneRingAreas(dm))
        WH = sp.sparse.dia_matrix(WH0.multiply(changeinarea))

    return dm


def iter_contract_point_cloud(pcd, iter_lim=2, contract_iter_lim=[15, 15], plot_pcd=False, mesh_radius_factor=2,
                              pcd_laplacian=False, verbose=False):
    """
    Apply the Laplacian contraction iteratively
    :param pcd: open3d.geometry.PointCloud()
    :param iter_lim: int: the number of the global iteration, how many iteration to apply Laplacian iteration
    :param contract_iter_lim: list of int : the iteration numbers of each Laplacian iteration
    :param plot_pcd: bool: if plot the point cloud in the contraction results
    :param mesh_radius_factor: float: The radius factor to produce a mesh from point cloud
    :param pcd_laplacian: bool: if we apply the Laplacian iteration based on point cloud else based on mesh
    :param verbose: if plot the inter product in the contraction process
    :return: open3d.geometry.PointCloud(): the point cloud contracted
    """
    for i in range(iter_lim):
        mesh = point_cloud_to_mesh(pcd, radius_factor= (5 * i+1) * mesh_radius_factor)
        if plot_pcd:
            open3d.visualization.draw_geometries([mesh])
        # Apply the Laplacian contraction
        m = contract(mesh, iter_lim=contract_iter_lim[i], pcd_laplacian=pcd_laplacian, verbose=verbose)
        pcd.points = open3d.utility.Vector3dVector(m['vertices'])

        if plot_pcd:
            pcd.colors = open3d.utility.Vector3dVector([0.2, 0.9, 0.1] * np.ones(m['vertices'].shape))
            open3d.visualization.draw_geometries([pcd, mesh])
            open3d.visualization.draw_geometries([pcd])
    return pcd


day = '03-22_AM'
pc_path = '../data/lyon2/processed/{}_segmented.ply'
pcd_clean_option = {'downsample_rate': 80,
                    'voxel_size': 0.7,
                    'crop': False,
                    'crop_low_bound': 0.0,
                    'crop_up_bound': 1}
mesh_radius_factor = 3
verbose = True


if __name__ == "__main__":
    pcd1 = open3d.io.read_point_cloud(pc_path.format(day))
    print("Clean Point Cloud")
    print(64 * "=")
    t_start = time.time()
    pcd, description_1 = clean_point_cloud(pcd1, option=pcd_clean_option, verbose=verbose)
    t_end = time.time()
    print(64 * "-")
    print("point cloud 1: {} points ".format(description_1['point_number']))
    print("used {:.3f}s".format(t_end - t_start))
    voxel_size = description_1['voxel_size']

    pcd = iter_contract_point_cloud(pcd,
                                    iter_lim=3,
                                    contract_iter_lim=[17, 15, 15],
                                    plot_pcd=True, pcd_laplacian=True)
    print("")
