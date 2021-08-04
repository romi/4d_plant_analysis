import json
import numpy as np
from sklearn.neighbors import KDTree


from organ_matching.organ_description import get_organ_description
from organ_matching.organ_matching_ml import get_organ_matching_from_description_ml
from organ_matching.organ_matching_utils import get_precision, \
    visualize_matching, \
    get_organ_skeletonized, \
    preprocess_data


def get_organ_matching_from_description(desc_1, desc_2):
    org_1 = np.array(range(0, desc_1.shape[0]))
    org_2 = np.array(range(0, desc_2.shape[0]))

    matches = -1 * np.ones(desc_1.shape[0])

    matches[0] = 0
    matches[1] = 1
    matches[2] = 2
    matches[-1] = int(org_2[-1])
    for i in range(3, desc_1.shape[0]-1):
        # print(desc_1[i])
        if i >= desc_2.shape[0]:
            break
        cand_1 = i
        candidates = []
        for cand in [i-1, i+1]:
            if cand < desc_2.shape[0] and np.abs(desc_2[cand, 7] - desc_2[cand, 7]) < 0.03:
                candidates.append(cand)
        candidates.append(cand_1)

        tree = KDTree(desc_2[candidates, -3:])
        winner = tree.query(desc_1[[i], -3:], k=1, return_distance=False).flatten()

        matches[i] = int(candidates[winner[0]])
    return matches


def match_organ_two_days(day1, day2, dataset, verbose=False, visualize=True, options=None):

    # desc_1 = desc_1 * weight.reshape(1, -1)
    options_path = "../hyper_parameters/lyon2.json"
    if not options:
        with open(options_path, "r") as json_file:
            options = json.load(json_file)

    org_dict_1, org_pcd_collect_1, gb1, xyz_labeled_1, xyz_stem_node_1, branch_connect_1 = \
        preprocess_data(day1, dataset, verbose=verbose, visualize=visualize, options=options)
    org_dict_2, org_pcd_collect_2, gb2, xyz_labeled_2, xyz_stem_node_2, branch_connect_2 = \
        preprocess_data(day2, dataset, verbose=verbose, visualize=visualize, options=options)

    desc_1, stem_connect_xyz_1 = get_organ_description(org_dict_1, org_pcd_collect_1, gb1, xyz_labeled_1,
                                                       xyz_stem_node_1)
    desc_2, stem_connect_xyz_2 = get_organ_description(org_dict_2, org_pcd_collect_2, gb2, xyz_labeled_2,
                                                       xyz_stem_node_2)

    matches = get_organ_matching_from_description_ml(desc_1, desc_2)
    if visualize:
        visualize_matching(org_pcd_collect_1, org_pcd_collect_2, matches, not_in_one=False)

    org_collect_1 = get_organ_skeletonized(org_pcd_collect_1, stem_connect_xyz_1, org_dict_1, xyz_labeled_1,
                                           branch_connect_1)
    org_collect_2 = get_organ_skeletonized(org_pcd_collect_2, stem_connect_xyz_2, org_dict_2, xyz_labeled_2,
                                           branch_connect_2)

    return matches, org_collect_1, org_collect_2


correct_matching = {
    ("03-22_AM", "03-22_PM"): [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15, 16],
    ("03-22_PM", "03-23_PM"): [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15, 14, 20],
    ("03-22_PM", "03-23_AM"): [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
    ("03-21_PM", "03-22_AM"): [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15],
    ("03-21_AM", "03-21_PM"): [0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 14],
    ("03-20_AM", "03-20_PM"): [0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 11],
    ("03-20_PM", "03-21_AM"): [0, 1, 2, 3, 4, 5, 6, 7, 8],
}


if __name__ == "__main__":
    day1 = "03-20_AM"
    day2 = "03-20_PM"

    matches, og1, og2 = match_organ_two_days(day1, day2, visualize=True)
    precision = []

    weight_mean = 10 * np.array([10, 10, 10, 10, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1])
    weight_std = 0 * np.ones(len(weight_mean))

    optimal_weight = weight_mean
    optimal_match = None
    optimal_precision = 0.9

    for _ in range(1):
        # weight = np.abs(np.random.normal(weight_mean, weight_std))
        # weight = weight_mean
        # print(weight)
        for (day1, day2) in correct_matching.keys():
            matches = match_organ_two_days(day1, day2, visualize=False, debug=False)[0]
            print(matches)
            print(correct_matching[(day1, day2)])
            print(64 * ".")
            precision.append(get_precision(matches, correct_matching[(day1, day2)]))
        current_precision = np.mean(precision)
        print(current_precision)
