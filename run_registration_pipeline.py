import copy
import os
import sys
import time
import json
import numpy as np
import open3d
sys.path.insert(0, "../")
from organ_matching.organ_matching_lr import match_organ_two_days
from p2p_matching_in_organ.p2p_matching_seg_icp import form_maize_org, form_tomato_org
from visualize_p2p import visualize_pcd_registration_series
import argparse

os.chdir('./p2p_matching_in_organ')

parser = argparse.ArgumentParser()
parser.add_argument("--type",
                    help="specify the type of plant to run the registration from tomato, maize and arabidopsis, " +
                         "default tomato")
parser.add_argument("--method",
                    help="specify the method used for point level registration from 'local_icp' and 'fm', " +
                         "default local_icp")
args = parser.parse_args()
if args.type:
    assert args.type in ["tomato", "maize", "arabidopsis"]
    if args.type == "arabidopsis":
        dataset = "lyon2"
    else:
        dataset = args.type
else:
    dataset = "tomato"


if args.method:
    assert args.method in ["local_icp", "fm"]
    method = args.method
else:
    method = "local_icp"

if method == "local_icp":
    from p2p_matching_in_organ.p2p_matching_seg_icp import get_p2p_mapping_organ_collection
else:
    from p2p_matching_in_organ.p2p_matching_seg_fm import get_p2p_mapping_organ_collection

score_path = "../scores/"

if __name__ == "__main__":
    if dataset == "lyon2":
        # days = ["03-21_AM", "03-21_PM"]
        days = [ "03-22_PM", "03-23_PM"]
        # days = ["05-17_AM", "05-17_PM", "05-18_AM", "05-18_PM", "05-19_AM", "05-19_PM", "05-20_AM"]
    if dataset == "tomato":
        days = ["03-05_AM", "03-06_AM", "03-07_AM", "03-08_AM", "03-09_AM", "03-10_AM", "03-11_AM"]
    if dataset == "maize":
        days = ["03-13_AM", "03-14_AM", "03-15_AM", "03-16_AM", "03-17_AM", "03-18_AM", "03-20_AM"]
    run_registration = True

    options_path = "../hyper_parameters/{}.json".format(dataset)
    with open(options_path, "r") as json_file:
        options = json.load(json_file)

    t_0 = time.time()
    scores = []
    for i in range(len(days) - 1):
        day1 = days[i]
        day2 = days[i + 1]
        if run_registration or not os.path.exists(options['p2p_save_path_segment'].format(dataset, day1, day2)):
            print("Running registration pipeline for plant in {} at {} and {}".format(dataset, day1, day2))
            t_start = time.time()
            # Preprocess the point cloud (skeleton extraction, segmentation, organ formation)
            # Then match the organs using random forest classifier

            if dataset == "lyon2":
                matches_index, org_collection_1, org_collection_2 = \
                    match_organ_two_days(day1, day2, dataset, visualize=True, verbose=True, options=options)

                t_end = time.time()
                print("Point cloud preprocessed and organs matched, used ", t_end - t_start, " s")
                t_start = time.time()
                s = get_p2p_mapping_organ_collection(org_collection_1[:],
                                                     org_collection_2,
                                                     matches_index[:],
                                                     day1=day1,
                                                     day2=day2,
                                                     dataset=dataset,
                                                     if_match_skeleton=True,
                                                     plot_mesh=False,
                                                     plot_pcd=False,
                                                     verbose=False,
                                                     show_all=False,
                                                     apply_double_direction_refinement=True,
                                                     options=options)
                scores.append(s)
                t_end = time.time()
                print("segment_level_matched and point level registration, used ", (t_end - t_start), " s")
                print(64 * "-")
                print("")

            if dataset == "tomato":
                org_collection_1 = [form_tomato_org(day1)]
                org_collection_2 = [form_tomato_org(day2)]
                matches_index = [0]
                t_start = time.time()
                s = get_p2p_mapping_organ_collection(org_collection_1,
                                                     org_collection_2,
                                                     matches_index,
                                                     day1=day1,
                                                     day2=day2,
                                                     dataset=dataset,
                                                     plot_mesh=False,
                                                     plot_pcd=False,
                                                     if_match_skeleton=False,
                                                     verbose=True,
                                                     apply_double_direction_refinement=True,
                                                     show_all=False)
                scores.append(s)
                t_end = time.time()

            if dataset == "maize":
                org_collection_1 = [form_maize_org(day1)]
                org_collection_2 = [form_maize_org(day2)]
                matches_index = [0]
                t_start = time.time()
                s = get_p2p_mapping_organ_collection(org_collection_1,
                                                     org_collection_2,
                                                     matches_index,
                                                     day1=day1,
                                                     day2=day2,
                                                     dataset=dataset,
                                                     plot_mesh=False,
                                                     plot_pcd=False,
                                                     if_match_skeleton=True,
                                                     verbose=False,
                                                     apply_double_direction_refinement=True,
                                                     show_all=False)
                scores.append(s)
                t_end = time.time()

    t_end = time.time()
    print("")
    print(128 * "=")
    print("All pair of plants registration finished, used {} s".format((t_end - t_0)))
    print("Average evaluation scores: ", np.vstack(scores).mean(axis=0))
    if not os.path.exists(score_path):
        os.mkdir(score_path)
    np.savetxt(score_path + "{}_{}.csv".format(dataset, method), scores)
    match_path_format = options['p2p_save_path_segment']
    visualize_pcd_registration_series(days, dataset, path_format=match_path_format)
