import numpy as np
import spharapy.trimesh as tm
import open3d
from utils import point_cloud_to_mesh
from point_cloud_clean import clean_point_cloud
import time
from functional_map_ import laplacian


def get_hks(basis_functions, natural_frequencies, K, ts):
    eigvalues = natural_frequencies[:K]
    eigvectors = basis_functions[:, :K]
    res = (eigvectors[:, :, None] ** 2) * np.exp(-eigvalues[None, :, None] * ts.flatten()[None, None, :])
    return np.sum(res, 1)


def get_hks_minmax(basis_functions, natural_frequencies, K, ts):
    hks_value = get_hks(basis_functions, natural_frequencies, K, ts)
    x = (hks_value - np.min(hks_value, axis=0))
    x /= np.max(x, axis=0)
    return x


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
          'voxel_size': 1,
          'crop': False,
          'crop_low_bound': 0.0,
          'crop_up_bound': 1}
descriptor_option = {"hks_ts": np.array([ 0.001, 0.01, 0.05, 0.1, 1, 5, 10, 100, 200]),
                     "wks_energy_num": 10}

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

    print("")
    print("Transform Point Cloud to Meshes")
    print(64 * "=")
    print("::radius factor: {}".format(mesh_radius_factor))
    t_start = time.time()
    mesh1 = point_cloud_to_mesh(pcd1, radius_factor=mesh_radius_factor)
    # mesh2 = mesh1 # for debug
    t_end = time.time()
    print("used {:.3f}s".format(t_end - t_start))


    if plot_mesh:
        open3d.visualization.draw_geometries([mesh1])

    print("")
    print("Calculate Laplacian Beltramis Basis")
    print(64 * "=")
    # Transform to trimesh to perform basis function calculation
    m1 = tm.TriMesh(np.asarray(mesh1.triangles), np.asarray(mesh1.vertices))
    t_start = time.time()
    print("::calculate the first {} basis functions".format(maxBasis))

    natural_frequencies_1, basis_functions_1 = laplacian.get_mesh_laplacian_spectrum(maxBasis,
                                                                                     m1.vertlist,
                                                                                     m1.trilist)
    # sphara_basis_1 = sb.SpharaBasis(m1, 'fem')
    # basis_functions_1, natural_frequencies_1 = sphara_basis_1.basis()
    ts = np.array([0.001,10, 100, 200, 1000])
    K = 100
    hks_value = get_hks_minmax(basis_functions_1, natural_frequencies_1, K, ts)
    for i in range(hks_value.shape[1]):
        color_value = 0.0 * np.ones( (hks_value.shape[0], 3) )
        color_value[:, 0] = hks_value[:, i]
        mesh1.vertex_colors = open3d.utility.Vector3dVector(color_value)
        if plot_mesh:
            open3d.visualization.draw_geometries([mesh1])
