import numpy as np
import json
from utils import plot_skeleton, save_graph, point_cloud_to_mesh, read_graph
from point_cloud_clean import clean_point_cloud
from skeletonization.L1_skeletonization import get_skeleton_pcd
from skeletonization.mesh_contraction import iter_contract_point_cloud
from skeletonization.connection_skeleton import merge_all_clsuter, \
    connect_pcd_branch, \
    get_point_label, \
    remove_cycle
from sklearn.neighbors import NearestNeighbors
import open3d
import time
import os


def laplacian_skeletonization(pcd,
                              day,
                              dataset,
                              voxel_size,
                              options=None,
                              verbose=True,
                              visualize=True,
                              contract_pcd=False):
    """

    :param pcd: open3d.geometry.PointCloud() the input point cloud
    :param day: str: the time of the point cloud, used to extract the corresponding hyper-parameters
    :param dataset: str: the name of the data set, in ["lyon2", "tomato", "maize"]
    :param voxel_size: float, the voxel size to down-sample the point cloud
    :param options: dict: the hyper-parameters of the skeleton extraction
                    options["skeleton_extract_option"] contains the parameters we need
    :param verbose: bool: if print the processing information during running
    :param visualize: bool: if visualize the inter-production
    :param contract_pcd: bool: if contract the point cloud or load the point cloud contracted
    :return: xyz: numpy.array float of shape (N, 3): skeleton nodes' coordinates
            connection: numpy.array int of shape (M, 2): the connection between skeleton nodes
    """

    # Load the hyper parameters from "../hyper_parameters/{dataset}.json"
    if options is None:
        options_path = "../hyper_parameters/{}.json".format(dataset)
        with open(options_path, "r") as json_file:
            options = json.load(json_file)

    path_format = options['root_path']
    if day not in options["skeleton_extract_option"]:
        options["skeleton_extract_option"][day] = {}

    # If no hyper-parameters are given the (dataset, day), replace it by none
    if "contract" not in options["skeleton_extract_option"][day]:
        options["skeleton_extract_option"][day]["contract"] = None
    if "connect" not in options["skeleton_extract_option"][day]:
        options["skeleton_extract_option"][day]["connect"] = None
    contraction_option = options["skeleton_extract_option"][day]["contract"]
    connection_option = options["skeleton_extract_option"][day]["connect"]

    t_0 = time.time() # begin the timing
    if not os.path.exists(path_format.format(dataset, day)):
        os.makedirs(path_format.format(dataset, day))
    if verbose:
        print("")
        print("Try Contracting the Point Cloud")
        print(64 * "=")

    mesh = point_cloud_to_mesh(pcd, radius_factor=options["mesh_radius_factor"])  # create the mesh from point cloud
    mesh.vertex_colors = open3d.utility.Vector3dVector(0.95 * np.ones((len(mesh.vertices), 3)))

    # Contract the point cloud if needed: Iteratively apply the Laplacian Contraction and L1 Contraction
    if contract_pcd or not os.path.exists(path_format.format(dataset, day) + "skeleton_pcd.csv".format(day)):
        # If no hyper-parameters assigned, use the default values
        if not contraction_option:
            contraction_option = {}
        if "laplacian_iter_lim" not in contraction_option:
            contraction_option["laplacian_iter_lim"] = 1
        if "contract_iter_lim" not in contraction_option:
            contraction_option["contract_iter_lim"] = [20, 12]
        if "pcd_laplacian_method" not in contraction_option:
            contraction_option["pcd_laplacian_method"] = True
        if "l1_sample_size" not in contraction_option:
            contraction_option["l1_sample_size"] = 1.1
        if "l1_iter_lim" not in contraction_option:
            contraction_option["l1_iter_lim"] = 1

        t_start = time.time()
        # Apply Laplacian Contraction
        pcd = iter_contract_point_cloud(pcd,
                                        iter_lim=contraction_option["laplacian_iter_lim"],
                                        contract_iter_lim=contraction_option["contract_iter_lim"],
                                        plot_pcd=visualize,
                                        verbose=verbose,
                                        mesh_radius_factor=options["mesh_radius_factor"],
                                        pcd_laplacian=contraction_option["pcd_laplacian_method"])
        points = np.asarray(pcd.points)

        # Apply L1 contraction
        final_centers = get_skeleton_pcd(points,
                                         contraction_option["l1_sample_size"] * voxel_size,
                                         plot_iter_pcd=visualize,
                                         iter=contraction_option["l1_iter_lim"],
                                         verbose=verbose)
        t_end = time.time()
        if verbose:
            print("::skeleton contracted: {} points".format(final_centers.shape[0]))
            print("::point_contraction used {:.2f}s".format(t_end - t_start))
        np.savetxt(path_format.format(dataset, day) + "skeleton_pcd.csv".format(day), final_centers)
    else:
        # If we have already computed the skeleton nodes contracted, we load it directly
        final_centers = np.loadtxt(path_format.format(dataset, day) + "skeleton_pcd.csv".format(day))

    if verbose:
        print("::down sampling at voxel_size: {:.2f}".format(1.1 * voxel_size))

    # Create an open3d.PointCloud with the points contracted, then down-sample it with larger voxel size
    pcd.points = open3d.utility.Vector3dVector(final_centers)
    pcd.colors = open3d.utility.Vector3dVector([0.2, 0.9, 0.1] * np.ones(final_centers.shape))
    pcd = pcd.voxel_down_sample(1.1 * voxel_size)
    xyz = np.asarray(pcd.points)  # Extract the points contracted down-sampled

    if visualize:
        open3d.visualization.draw_geometries([pcd])

    if verbose:
        print("")
        print("Try Establishing Connections")
        print(64 * "-")
        print("::down sampled center number: {}".format(xyz.shape[0]))

    # If no hyper-parameters assigned, use the default values
    if not connection_option:
        connection_option = {}
    if "sigma_threshold" not in connection_option:
        connection_option["sigma_threshold"] = 0.9
    if "neighbor_number" not in connection_option:
        connection_option["neighbor_number"] = 5
    if "distance_threshold" not in connection_option:
        connection_option["distance_threshold"] = 3

    xyz[np.isnan(xyz)] = 0.0  # remove the outliers of the skeleton nodes

    NNbrs = NearestNeighbors(n_neighbors=10, algorithm='kd_tree').fit(xyz)
    distance, indices = NNbrs.kneighbors(xyz)
    threshold = connection_option["distance_threshold"] * voxel_size

    # Begin the skeleton nodes connection
    t_start = time.time()
    # Classify the skeleton nodes into branch points and joint points
    labels, vector_orientation = get_point_label(xyz,
                                                 indices,
                                                 distance,
                                                 number_NN=connection_option["neighbor_number"],
                                                 sigma_threshold=connection_option["sigma_threshold"],
                                                 visualize_label=visualize)
    # Connect the branch points with each other, connect the joint points with the nearest branch points
    # clusters are lists of the nodes index
    # clusters_connection are lists of connections within each cluster
    # There is no connection between two different cluster
    clusters, clusters_connection = connect_pcd_branch(xyz,
                                                       labels,
                                                       vector_orientation,
                                                       threshold,
                                                       indices,
                                                       distance,
                                                       number_NN=2,
                                                       plot_branch=False)
    t_end = time.time()
    # Finish the points connection
    if verbose:
        print("::labelling the points used {:.2f}s".format(t_end - t_start))
        print("::discovered {} connected clusters".format(len(clusters)))

    connection = []
    for c in clusters_connection:
        connection += c
    if verbose:
        print("::discovered {} edges".format(len(connection)))

    if visualize:
        plot_skeleton(xyz, np.array(connection), plot_point=True)

    clusters = [list(c) for c in clusters]
    clusters_ = []
    connection = []

    if "cluster_number_lim" not in connection_option:
        connection_option["cluster_number_lim"] = 1

    # Filter the clusters with fewer nodes than the threshold
    for i, cl in enumerate(clusters):
        if len(cl) > connection_option["cluster_number_lim"]:
            clusters_.append(cl)
            connection += clusters_connection[i]
    clusters = clusters_

    # Merge all clusters by iteratively merge the closest cluster
    t_start = time.time()
    connection, clusters = merge_all_clsuter(xyz, connection, clusters)
    t_end = time.time()
    if verbose:
        print("::merging the connected clusters used {:.2f}s".format(t_end - t_start))

    # Remove the cycle in the skeleton by a minimal spamming tree
    connection = remove_cycle(xyz, connection, clusters)
    if verbose:
        print("::removing the cycles in the connected graph")
        print("::final number of connection established: {}".format(len(connection)))
    if visualize:
        plot_skeleton(xyz, np.array(connection), plot_point=True)

    # Save the skeleton
    save_graph(path_format.format(dataset, day) + "skeleton_{}_connected.txt".format(day), xyz, connection)
    if verbose:
        print(64 * "=")
        print("Skeleton Extraction : used {:.2f}".format(time.time() - t_0))
    return xyz, connection


def skeleton_extraction_by_day(day,
                               data_set='lyon2',
                               load_data=True,
                               pcd_path_format='../data/{}/processed/{}_segmented.ply',
                               skeleton_path_format="../data/{}/{}/",
                               pcd_clean_option=None,
                               verbose=False,
                               visualize=False):
    """
    Execute the laplacian skeleton extraction to the plant in a specific day time

    :param load_data: bool: if load the point cloud data or recompute it
    :param skeleton_path_format: str: the string format of the path to save the str, need to blanks
    :param verbose: bool: if print the information
    :param visualize: bool: if visualize the inter-product
    :param day: the time of the acquisition
    :param data_set: the dataset in ["lyon2", "tomato", "maize"], by default "lyon2"
    :param pcd_path_format: the format of the relative path to the point cloud
    :param pcd_clean_option: the option dictionary for clean the point cloud
    :return: xyz: numpy.array float of shape (N, 3): skeleton nodes' coordinates
            connection: numpy.array int of shape (M, 2): the connection between skeleton nodes
    """

    if not pcd_clean_option:
        pcd_clean_option = {'downsample_rate': 80,
                            'voxel_size': 1,
                            'crop': False,
                            'crop_low_bound': 0.0,
                            'crop_up_bound': 1}

    # load the point cloud data from the path
    pcd_path = pcd_path_format.format(data_set, day)
    pcd = open3d.io.read_point_cloud(pcd_path)
    if verbose:
        print("Start Skeleton Extraction")
        print(128 * "=")

    # clean the point cloud: including voxel down-sample and center the point cloud at (0,0,0)
    start_time = time.time()
    pcd, description = clean_point_cloud(pcd, option=pcd_clean_option, verbose=verbose)
    voxel_size = description['voxel_size']  # take the voxel size of down sampling used to clean the point cloud

    skeleton_path = skeleton_path_format.format(data_set, day) + "skeleton_{}_connected.txt".format(day)
    if os.path.exists(skeleton_path) and load_data:
        # if there exists the skeleton file computed, load that directly
        xyz, connection = read_graph(skeleton_path)
        if verbose:
            print("\t Read the saved skeleton document")
    else:
        # if no file available or we want to recompute the skeleton intentionally, we redo the skeletal extraction
        if verbose:
            print("\t Produce the skeleton from scratch")
        xyz, connection = laplacian_skeletonization(pcd, day, data_set, voxel_size, visualize=visualize,
                                                    verbose=verbose, contract_pcd=True
                                                    )
    end_time = time.time()

    if verbose:
        print(64 * "=")
        print("\t Finished skeleton extraction: \t{} s".format(end_time - start_time))
    if visualize:
        plot_skeleton(xyz, np.array(connection), plot_point=True)
    return xyz, connection


if __name__ == "__main__":
    days = ["03-23_PM"]
    # days = ["03-05_AM", "03-06_AM", "03-07_AM", "03-08_AM", "03-09_AM", "03-10_AM", "03-11_AM"]
    for day in days:
        xyz, connection = skeleton_extraction_by_day(day,
                                                     data_set="lyon2", load_data=False, verbose=True, visualize=True)
