import os
import copy
import time
import json
import numpy as np
import joblib
import open3d
from scipy.optimize import linprog
from sklearn.neighbors import KDTree
import sys
sys.path.insert(0, "../")
from organ_matching.organ_matching_lr import match_organ_two_days
from segment_matching.segment_matching_utils import get_segments
from segment_matching.segment_matching_nn import get_optimal_match_with_weights

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC


def get_segment_description(segment):
    skeleton = segment["skeleton"]
    center_position = np.mean(skeleton, axis=0) / 50
    orientation = (skeleton[-1] - skeleton[0]) / (2*10**-5 + np.linalg.norm(skeleton[-1] - skeleton[0]))
    length = np.sum([np.linalg.norm(skeleton[i+1] - skeleton[i]) for i in range(skeleton.shape[0] - 1)])
    length = length / 50.0
    organ_center_offset = center_position - segment['organ_center_offset'] / 50
    degree = segment["degree"]
    point_number_proportion = segment["point_number_proportion"]
    geodesic_distance = segment['geodesic_distance']
    return np.hstack([center_position,
                      orientation,
                      length,
                      degree,
                      point_number_proportion,
                      geodesic_distance,
                      organ_center_offset])


def label_train_set(day1, day2):

    features = []
    labels = []
    dataset = "lyon2"
    organ_matching, org_collection_1, org_collection_2 = match_organ_two_days(day1, day2, dataset, visualize=False)
    org_collection_1 = org_collection_1[3:]
    organ_matching = organ_matching[3:]

    for index, org1 in enumerate(org_collection_1):
        if organ_matching[index] < 0:
            continue
        org2 = org_collection_2[int(organ_matching[index])]
        istop_1 = index == len(org_collection_1)
        istop_2 = int(organ_matching[index]) == len(org_collection_2)

        segments_1 = get_segments(org1, istop_1)
        segments_2 = get_segments(org2, istop_2)

        segment_index_1 = np.array(list(segments_1.keys()))
        segment_index_2 = np.array(list(segments_2.keys()))

        skel_pcd_1 = open3d.geometry.PointCloud()
        skel_pcd_1.points = open3d.utility.Vector3dVector(org1["skeleton_pcd"][:, :3])
        skel_pcd_2 = open3d.geometry.PointCloud()
        skel_pcd_2.points = open3d.utility.Vector3dVector(org2["skeleton_pcd"][:, :3])

        print(segment_index_1)
        print(segment_index_2)
        for seg1 in segment_index_1:
            for seg2 in segment_index_2:
                if np.abs(segments_1[seg1]["geodesic_distance"] - segments_2[seg2]["geodesic_distance"]) > 1:
                    if np.random.rand() > 0.1:
                        continue
                segment_feature_1 = get_segment_description(segments_1[seg1])
                segment_feature_2 = get_segment_description(segments_2[seg2])
                print("features:")
                print(seg1, segment_feature_1)
                print(seg2, segment_feature_2)
                open3d.visualization.draw_geometries([skel_pcd_1, segments_1[seg1]['pcd'],
                                                     skel_pcd_2, segments_2[seg2]['pcd']])
                while True:
                    manual_label = input("Please input if the pair segments are matched: ")
                    try:
                        int(manual_label)
                        break
                    except ValueError:
                        continue
                print(32 * "-")
                if int(manual_label[0]) > 0.5:
                    labels.append(1)
                else:
                    labels.append(0)
                features.append(np.abs(segment_feature_1 - segment_feature_2))
            np.savetxt("./training_data_new/{}_to_{}_features.csv".format(day1, day2), np.vstack(features))
            np.savetxt("./training_data_new/{}_to_{}_labels.csv".format(day1, day2), np.array(labels))

    return


def get_train_set(day1 = None, day2 = None, data_path = "./training_data_new/"):
    days = [("03-20_AM", "03-20_PM"), ("03-20_PM", "03-21_AM"), ("03-21_AM", "03-21_PM"), ("03-21_PM", "03-22_AM"),
            ("03-22_AM", "03-22_PM"), ("03-22_PM", "03-23_AM"), ("03-23_AM", "03-23_PM"), ("03-22_PM", "03-23_PM")]

    postfix = ["", "_", "_1", "_2"]
    if (day1, day2) in days:
        days = [(day1, day2)]

    features = []
    labels = []
    for (d1, d2) in days:
        for p_ in postfix:
            if os.path.exists(data_path + "{}_to_{}_features{}.csv".format(d1, d2, p_)):
                features.append(np.loadtxt(data_path + "{}_to_{}_features{}.csv".format(d1, d2, p_)))
                labels.append(np.loadtxt(data_path + "{}_to_{}_labels{}.csv".format(d1, d2, p_)))
    return np.vstack(features), np.hstack(labels)


def get_segment_match(organ_1, organ_2, landmark1, landmark2,
                      istop_1, istop_2, isstem_1, isstem_2, options=None, sample_method="end"):
    """
    :param istop_1: bool: if the first organ is the top organ ( in case yes, we do not apply segmentation)
    :param istop_2: bool: if the second organ is the top organ  ( in case no, we do not apply segmentation)
    :param isstem_1: bool: if the first organ is the main stem
    :param isstem_2: bool: if the second organ is the main stem
    :param options: dict: the hyper-parameters used for point cloud processing
    :param organ_1: dict: the first organ with:
                {
                "pcd": the point cloud of the organ,
                "stem_connect_point": the connect point to the main stem or root,
                "skeleton_pcd": float numpy array of shape [N, 4], the nodes of the organ's skeleton, first 3 indices are the
                                xyz values of the nodes, the 4th one is the segment index of the node
                "skeleton_connection": integer numpy array of shape [N, 2], the connection of the skeleton's node,
                "segment_index": all indices of the segments in this organ
            }
    :param organ_2: dict: the second organ
    :param landmark1: landmark matched already for organ1
    :param landmark2: landmark matched already for organ2, landmark1.shape == landmark2.shape
    :return: (
        list of the segments' point clouds in organ1,
        list of the segments' point clouds in organ2,
        list of the segments' landmarks in organ1,
        list of the segments' landmarks in organ2,
        list of the segments' semantic meanings in organ1,
        list of the segments' semantic meanings in organ2
    )
    """

    assert sample_method in ["end", "equal"]

    options_path = "../hyper_parameters/lyon2.json"
    if not options:
        with open(options_path, "r") as json_file:
            options = json.load(json_file)

    clf = load_model("random_forest", isstem=isstem_1 or isstem_2)

    segments_1 = get_segments(organ_1, istop_1)
    segments_2 = get_segments(organ_2, istop_2)

    segment_index_1 = np.array(list(segments_1.keys()))
    segment_index_2 = np.array(list(segments_2.keys()))

    match_count = np.zeros((len(segment_index_1), len(segment_index_2)))
    for seg1 in segment_index_1:
        for seg2 in segment_index_2:
            i_1 = np.where(segment_index_1 == seg1)[0][0]
            i_2 = np.where(segment_index_2 == seg2)[0][0]
            f = np.abs(get_segment_description(segments_1[seg1]) - get_segment_description(segments_2[seg2]))
            proba = clf.predict_proba(f.reshape(1,-1))
            match_count[i_1, i_2] = 10 * proba[0, 1]

    if landmark1 is not None or landmark2 is not None:
        for seg1, seg2 in zip(landmark1[:, 3], landmark2[:, 3]):
            if seg1 in segment_index_1 and seg2 in segment_index_2:
                i_1 = np.where(segment_index_1 == seg1)[0][0]
                i_2 = np.where(segment_index_2 == seg2)[0][0]
                match_count[i_1, i_2] += 1

    # linprog_res = get_optimal_match_with_weights(match_count)
    if len(segment_index_1) <= len(segment_index_2):
        linprog_res = get_optimal_match_with_weights(match_count)
        if linprog_res.success:
            x = linprog_res.x.reshape(match_count.shape)
            segment_match = []
            not_visted = segment_index_2
            for i in range(x.shape[0]):
                cands = [j for j in range(x.shape[1]) if x[i, j] > 0.5]
                if len(cands) == 0:
                    segment_match.append([segment_index_1[i],
                                          segment_index_2[np.argmax(x[i, :])]])
                else:
                    final = cands[0]
                    for j in cands:
                        if match_count[i, j] > match_count[i, final]:
                            final = j
                    segment_match.append([segment_index_1[i],
                                          segment_index_2[final]])
            segment_match = np.vstack(segment_match)
            pcd1, pcd2, skel_lm1, skel_lm2, semantic_1, semantic_2 = [], [], [], [], [], []
            for [seg_i1, seg_i2] in segment_match:
                pcd1.append(segments_1[seg_i1]['pcd'])
                pcd2.append(segments_2[seg_i2]['pcd'])
                skel_1 = segments_1[seg_i1]['skeleton']
                skel_2 = segments_2[seg_i2]['skeleton']
                semantic_1.append(segments_1[seg_i1]['semantic_meaning'])
                semantic_2.append(segments_2[seg_i2]['semantic_meaning'])

                voxel_size = options["pcd_clean_option"]["voxel_size"] * 2
                skel_1_new = [skel_1[0, :]]
                for x in skel_1:
                    if np.linalg.norm(skel_1_new[-1] - x) > voxel_size:
                        skel_1_new.append(x)

                skel_2_new = [skel_2[0, :]]
                for x in skel_2:
                    if np.linalg.norm(skel_2_new[-1] - x) > voxel_size:
                        skel_2_new.append(x)

                skel_1 = np.vstack(skel_1_new)
                skel_2 = np.vstack(skel_2_new)

                if sample_method == "end":
                    len_keep = min(skel_1.shape[0], skel_2.shape[0])
                    skel_1, skel_2 = skel_1[:len_keep], skel_2[:len_keep]

                    skel_lm1.append(np.vstack([skel_1[0:skel_1.shape[0]-1: 2], skel_1[-1]]))
                    skel_lm2.append(np.vstack([skel_2[0:skel_2.shape[0]-1: 2], skel_2[-1]]))
                else:
                    len_sample = min(skel_1.shape[0], skel_2.shape[0]) / 2
                    sample_step_1, sample_step_2 = int(skel_1.shape[0] / len_sample) + 1, \
                                                   int(skel_2.shape[0] / len_sample) + 1
                    skel_1 = skel_1[0:skel_1.shape[0]: sample_step_1]
                    skel_2 = skel_2[0:skel_2.shape[0]: sample_step_2]
                    len_keep = min(skel_1.shape[0], skel_2.shape[0])
                    skel_1, skel_2 = skel_1[:len_keep], skel_2[:len_keep]
                    skel_lm1.append(skel_1)
                    skel_lm2.append(skel_2)

            for seg_i2 in segment_index_2:
                if seg_i2 not in segment_match[:, 1]:
                    pcd2.append(segments_2[seg_i2]['pcd'])

        else:
            pcd1, pcd2, skel_lm1, skel_lm2 = [organ_1["pcd"]], [organ_2["pcd"]], [landmark1[:, :3]], [landmark2[:, :3]]
            semantic_1 = ["multiple"]
            semantic_2 = ["multiple"]
        return pcd1, pcd2, skel_lm1, skel_lm2, semantic_1, semantic_2

    else:
        linprog_res = get_optimal_match_with_weights(match_count.T)
        if linprog_res.success:
            x = linprog_res.x.reshape((match_count.T).shape)
            segment_match = []
            for i in range(x.shape[0]):
                cands = [j for j in range(x.shape[1]) if x[i, j] > 0.5]
                if len(cands) == 0:
                    segment_match.append((segment_index_1[np.argmax(x[i, :])],
                                          segment_index_2[i]))
                else:
                    final = cands[0]
                    for j in cands:
                        if match_count[j, i] > match_count[final, i]:
                            final = j
                    segment_match.append([segment_index_1[final],
                                          segment_index_2[i]])
            pcd1, pcd2, skel_lm1, skel_lm2, semantic_1, semantic_2 = [], [], [], [], [], []
            segment_match = np.vstack(segment_match)

            for [seg_i1, seg_i2] in segment_match:
                pcd1.append(segments_1[seg_i1]['pcd'])
                pcd2.append(segments_2[seg_i2]['pcd'])
                skel_1 = segments_1[seg_i1]['skeleton']
                skel_2 = segments_2[seg_i2]['skeleton']
                semantic_1.append(segments_1[seg_i1]['semantic_meaning'])
                semantic_2.append(segments_2[seg_i2]['semantic_meaning'])

                voxel_size = options["pcd_clean_option"]["voxel_size"] * 3
                skel_1_new = [skel_1[0, :]]
                for x in skel_1:
                    if np.linalg.norm(skel_1_new[-1] - x) > voxel_size:
                        skel_1_new.append(x)

                skel_2_new = [skel_2[0, :]]
                for x in skel_2:
                    if np.linalg.norm(skel_2_new[-1] - x) > voxel_size:
                        skel_2_new.append(x)

                skel_1 = np.vstack(skel_1_new)
                skel_2 = np.vstack(skel_2_new)

                if sample_method == "end":
                    len_keep = min(skel_1.shape[0], skel_2.shape[0])
                    skel_1, skel_2 = skel_1[:len_keep], skel_2[:len_keep]

                    skel_lm1.append(np.vstack([skel_1[0:skel_1.shape[0] - 1: 2], skel_1[-1]]))
                    skel_lm2.append(np.vstack([skel_2[0:skel_2.shape[0] - 1: 2], skel_2[-1]]))
                else:
                    len_sample = min(skel_1.shape[0], skel_2.shape[0]) / 2
                    sample_step_1, sample_step_2 = int(skel_1.shape[0] / len_sample) + 1, \
                                                   int(skel_2.shape[0] / len_sample) + 1
                    skel_1 = skel_1[0:skel_1.shape[0]: sample_step_1]
                    skel_2 = skel_2[0:skel_2.shape[0]: sample_step_2]
                    len_keep = min(skel_1.shape[0], skel_2.shape[0])
                    skel_1, skel_2 = skel_1[:len_keep], skel_2[:len_keep]
                    skel_lm1.append(skel_1)
                    skel_lm2.append(skel_2)

            for seg_i1 in segment_index_1:
                if seg_i1 not in segment_match[:, 0]:
                    pcd1.append(segments_1[seg_i1]['pcd'])
        else:
            pcd1, pcd2, skel_lm1, skel_lm2 = [organ_1["pcd"]], [organ_2["pcd"]], [landmark1[:, :3]], [landmark2[:, :3]]
            semantic_1 = ["multiple"]
            semantic_2 = ["multiple"]
        return pcd1, pcd2, skel_lm1, skel_lm2, semantic_1, semantic_2


def training(model,
             save_model=True,
             save_path="../segment_matching/clf_model"):
    """
    train a set of machine learning models to do the segment level matching
    :param model:  the model to train
    :param save_model: bool, if save the model
    :param save_path: str, the path to save the model trained
    :return: the model trained
    """
    X, y = get_train_set()
    print("Begin Training")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=0)
    if model == "random_forest":
        clf = RandomForestClassifier(max_depth=17, random_state=0, criterion="entropy")
    elif model == "SVC":
        clf = SVC(gamma='auto', kernel="rbf")
    elif model == "logistic_regression":
        clf = LogisticRegression()
    else:
        print("Type of classifier model not accepted")
        return

    clf.fit(X_train, y_train)
    y_predict = clf.predict(X_test)
    print(32 * "-")
    print("Evaluation on test set")
    print("confusion matrix")
    print(confusion_matrix(y_test, y_predict))
    print("accuracy: \t", accuracy_score(y_test, y_predict))
    print("f1_score: \t", f1_score(y_test, y_predict))
    if save_model:
        joblib.dump(clf, save_path + "/segment_matching_{}_clf.joblib".format(model))
    return clf


def load_model(model=None, isstem=False, path="../segment_matching/clf_model"):
    if model not in {"random_forest", "SVC", "logistic_regression"}:
        model = "random_forest"
    if isstem:
        clf = joblib.load(path + "/segment_matching_{}_clf_stem.joblib".format(model))
    else:
        clf = joblib.load(path + "/segment_matching_{}_clf.joblib".format(model))
    return clf


if __name__ == "__main__":
    day1 = "03-20_AM"
    day2 = "03-20_PM"

    # label_train_set(day1, day2)
    features, labels = get_train_set()
    # clf = load_model("random_forest")
    #
    clf = training("random_forest")
    # days = [("03-20_AM", "03-20_PM"), ("03-20_PM", "03-21_AM"), ("03-21_AM", "03-21_PM"), ("03-21_PM", "03-22_AM"),
    #         ("03-22_AM", "03-22_PM"), ("03-22_PM", "03-23_AM"), ("03-23_AM", "03-23_PM"), ("03-22_PM", "03-23_PM")]
    # for (day1, day2) in days:
    #     reforming_features(day1, day2)


    print()

