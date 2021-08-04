import os
import json
import copy
import joblib
import numpy as np
import open3d

from organ_matching.organ_description import get_organ_description
from organ_matching.organ_matching_utils import preprocess_data

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

organ_match_ground_truth_path = "../data/lyon2/organ_match_gt.json"


def get_training_data(dataset):
    """
    is_stem,
     is_root,
     is_root_branch,
     is_top,
     height_order,
     point_number,
     seg_number,
     connect_stem_order,
     xyz_center,
     orientation_relative_stem
    :return:
    """
    train_features = []
    labels = []
    with open(organ_match_ground_truth_path) as json_file:
        organ_match_ground_truth = json.load(json_file)

    correct_matching = {}
    for k in organ_match_ground_truth:
        correct_matching[tuple(k.split(" "))] = organ_match_ground_truth[k]

    for (day1, day2) in correct_matching:
        print(day1, day2)
        gt_match = correct_matching[(day1, day2)]
        org_dict_1, org_pcd_collect_1, gb1, xyz_labeled_1, xyz_stem_node_1, branch_connect_1 = preprocess_data(day1,
                                                                                                               dataset)
        org_dict_2, org_pcd_collect_2, gb2, xyz_labeled_2, xyz_stem_node_2, branch_connect_2 = preprocess_data(day2,
                                                                                                               dataset)
        org_pcd_collect_2_copy = copy.deepcopy(org_pcd_collect_2)
        for org_ in org_pcd_collect_2_copy:
            org_.translate([0, 100, 0])
        # open3d.visualization.draw_geometries(org_pcd_collect_1 + org_pcd_collect_2_copy)
        desc_1, stem_connect_xyz_1 = get_organ_description(org_dict_1, org_pcd_collect_1, gb1, xyz_labeled_1,
                                                           xyz_stem_node_1)
        desc_2, stem_connect_xyz_2 = get_organ_description(org_dict_2, org_pcd_collect_2, gb2, xyz_labeled_2,
                                                           xyz_stem_node_2)

        for i in range(3, len(gt_match)):
            train_features.append(desc_1[i] - desc_2[gt_match[i]])

            labels.append(1)
            # data augmentation, add some noises to the matched pairs of organs
            # simulate the situation of motion difference and of acquisition failure
            for _ in range(2):
                fake_desc1 = copy.deepcopy(desc_1[i])
                fake_desc2 = copy.deepcopy(desc_2[gt_match[i]])
                add_up_1 = np.zeros(fake_desc1.shape[0])
                add_up_2 = np.zeros(fake_desc2.shape[0])
                if np.random.rand() > 0.5:
                    add_up_1[4] = 1  # remove a branch below
                if np.random.rand() > 0.5:
                    add_up_2[4] = 1  # add a branch below
                add_up_1[5] += 0.05 * np.random.randn()
                add_up_2[5] += 0.05 * np.random.randn()
                add_up_1[-3:] += 0.005 * np.random.randn(3)
                add_up_2[-3:] += 0.005 * np.random.randn(3)
                fake_desc1 += add_up_1
                fake_desc2 += add_up_2

                train_features.append(fake_desc1 - fake_desc2)
                labels.append(1)

            train_features.append(desc_1[i] - desc_2[gt_match[i] - 1])
            labels.append(0)
            for k in range(1, 5):
                if gt_match[i] < len(desc_2) - k:
                    train_features.append(desc_1[i] - desc_2[gt_match[i] + k])
                    labels.append(0)

    return np.vstack(train_features), np.array(labels)


def training(X, y, save_model=True, save_path="../organ_matching/clf_model"):
    clf = RandomForestClassifier(max_depth=9, random_state=0)
    clf.fit(X, y)
    if save_model:
        joblib.dump(clf, save_path + "/organ_matching_random_forest_clf.joblib")
    return clf


def load_model(path="../organ_matching/clf_model"):
    clf = joblib.load(path + "/organ_matching_random_forest_clf.joblib")
    return clf


def get_organ_matching_from_description_ml(desc_1, desc_2, organ_matching_model_path="../organ_matching/clf_model"):
    """
    match the organs of two plants from their descriptor values
    the plant 1 has M organs, the plant 2 has N organs
    :param organ_matching_model_path: the relative path to load the machine learning model for the classification
    :param desc_1: the descriptors values for organs in plant 1: shape: [M, n_d]
    :param desc_2: the descriptors values for organs in plant 2: shape: [N, n_d]
    :return: match index for plant 1,
            match[i] = j, means i-th organ in plant 1 and j-th organ in plant 2 are correspondent
    """

    org_1 = np.array(range(0, desc_1.shape[0]))
    org_2 = np.array(range(0, desc_2.shape[0]))

    matches = -1 * np.ones(desc_1.shape[0])
    clf = load_model(organ_matching_model_path)

    matches[0] = 0
    matches[1] = 1
    matches[2] = 2
    matches[-1] = int(org_2[-1])
    for i in range(3, desc_1.shape[0] - 1):
        # print(desc_1[i])
        if i >= desc_2.shape[0]:
            break
        cands = [j for j in range(i - 2, i + 2) if 2 < j < desc_2.shape[0]]
        cand_desc = desc_2[cands]
        X = desc_1[i] - cand_desc
        y_predict_proba = clf.predict_proba(X)[:, 1]
        cands = [cands[j] for j in range(len(cands)) if y_predict_proba[j] > 0.5]
        y_predict_proba = [y_prob for y_prob in y_predict_proba if y_prob > 0.5]

        if len(cands) == 1:
            matches[i] = cands[0]
        elif len(cands) == 0:
            matches[i] = -1
        else:
            matches[i] = cands[np.argmax(y_predict_proba)]
    return matches


if __name__ == "__main__":

    is_train = True
    if is_train:
        train_feature, label = get_training_data("lyon2")
        X_train, X_test, y_train, y_test = train_test_split(train_feature, label, test_size=0.30, random_state=0)
        clf = training(X_train, y_train, save_model=True)
        y_predict = clf.predict(X_test)
        print(32 * "-")
        print("Evaluation on test set")
        print("confusion matrix")
        print(confusion_matrix(y_test, y_predict))
        print("accuracy: \t", accuracy_score(y_test, y_predict))
        print("f1_score: \t", f1_score(y_test, y_predict))
    else:
        clf = load_model()

    # day1 = "03-22_PM"
    # day2 = "03-23_PM"
    #
    # gt_match = correct_matching[(day1, day2)]
    # org_dict_1, org_pcd_collect_1, gb1, xyz_labeled_1, xyz_stem_node_1, branch_connect_1 = preprocess_data(day1,
    #                                                                                                        "lyon2")
    # org_dict_2, org_pcd_collect_2, gb2, xyz_labeled_2, xyz_stem_node_2, branch_connect_2 = preprocess_data(day2,
    #                                                                                                        "lyon2")
    #
    # desc_1, stem_connect_xyz_1 = get_organ_description(org_dict_1, org_pcd_collect_1, gb1, xyz_labeled_1,
    #                                                    xyz_stem_node_1)
    # desc_2, stem_connect_xyz_2 = get_organ_description(org_dict_2, org_pcd_collect_2, gb2, xyz_labeled_2,
    #                                                    xyz_stem_node_2)
    # matches = get_organ_matching_from_description_ml(desc_1, desc_2)
    #
    # print(matches)
    # print(correct_matching[(day1, day2)])
