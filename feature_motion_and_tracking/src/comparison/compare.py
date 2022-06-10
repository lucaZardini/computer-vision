import json
from typing import List, Tuple

from feature_detector.feature_detector import FeatureDetectorAlgorithm
import numpy as np


KEYPOINTS_DIRECTORY = "../../keypoints/"
KEYPOINTS_EXTENTION = ".npy"
NUMBER_OF_KEYPOINTS = 17

KEYPOINTS_DIRECTORY_COMPARISON = "../../keypoints_comparison/"
KEYPOINTS_COMPARISON_EXTENTION = ".json"
KEYPOINTS_COMPARISON_SEPARATOR = "__2000___"


def save_keypoints(overlap: List[int], number_of_keypoints_first: List[int], number_of_keypoints_second: List[int], first_method: FeatureDetectorAlgorithm, second_method: FeatureDetectorAlgorithm):
    with open(KEYPOINTS_DIRECTORY_COMPARISON+first_method.value+KEYPOINTS_COMPARISON_SEPARATOR+second_method.value+KEYPOINTS_COMPARISON_EXTENTION, "w") as out_file:
        json_output = {
            "numberOfKeypointsFirst": number_of_keypoints_first,
            "numberOfKeypointsSecond": number_of_keypoints_second,
            "overlap": overlap
        }
        json.dump(json_output, out_file)


def compare_keypoints(keypoint_first: List[Tuple[int, int]], keypoint_second: List[Tuple[int, int]], number_of_keypoints: int) -> (int, int, int, int):
    keypoints_in_both = 0
    keypoints_first_len = len(keypoint_first)
    keypoints_second_len = len(keypoint_second)
    number_of_keypoints = min(len(keypoint_first), len(keypoint_second))
    if len(keypoint_first) != number_of_keypoints:
        tmp = keypoint_first
        keypoint_first = keypoint_second
        keypoint_second = tmp
    for keypoint_number in range(number_of_keypoints):
        keypoint = keypoint_first[keypoint_number][0]
        for i, second_keypoint in enumerate(keypoint_second):
            if (int(round(keypoint[0])), int(round(keypoint[1]))) == (int(round(second_keypoint[0][0])), int(round(second_keypoint[0][1]))):
                keypoints_in_both += 1
    return keypoints_in_both, number_of_keypoints, keypoints_first_len, keypoints_second_len


def compare_algorithms_keypoints(first_method: FeatureDetectorAlgorithm, second_method: FeatureDetectorAlgorithm):
    if first_method == second_method:
        return ValueError("The two methods must be different")

    first_method_keypoints = np.load(KEYPOINTS_DIRECTORY+first_method.value+KEYPOINTS_EXTENTION, allow_pickle=True)
    second_method_keypoints = np.load(KEYPOINTS_DIRECTORY+second_method.value+KEYPOINTS_EXTENTION, allow_pickle=True)

    keypoints_comparison = []
    number_of_keypoints_first = []
    number_of_keypoints_second = []
    for frame in range(len(first_method_keypoints)):
        frame_keypoints_first_method = first_method_keypoints[frame]
        frame_keypoints_second_method = second_method_keypoints[frame]
        overlap, min, kp_len_first, kp_len_second = compare_keypoints(frame_keypoints_first_method, frame_keypoints_second_method, NUMBER_OF_KEYPOINTS)
        number_of_keypoints_first.append(kp_len_first)
        number_of_keypoints_second.append(kp_len_second)
        keypoints_comparison.append(overlap)

    save_keypoints(keypoints_comparison, number_of_keypoints_first, number_of_keypoints_second, first_method, second_method)


if __name__ == "__main__":
    compare_algorithms_keypoints(first_method=FeatureDetectorAlgorithm.STAR, second_method=FeatureDetectorAlgorithm.ORB)
    compare_algorithms_keypoints(first_method=FeatureDetectorAlgorithm.STAR, second_method=FeatureDetectorAlgorithm.SIFT)
    compare_algorithms_keypoints(first_method=FeatureDetectorAlgorithm.STAR, second_method=FeatureDetectorAlgorithm.GOOD_FEATURES_TO_TRACK)
    compare_algorithms_keypoints(first_method=FeatureDetectorAlgorithm.SIFT, second_method=FeatureDetectorAlgorithm.ORB)
    compare_algorithms_keypoints(first_method=FeatureDetectorAlgorithm.GOOD_FEATURES_TO_TRACK, second_method=FeatureDetectorAlgorithm.ORB)
    compare_algorithms_keypoints(first_method=FeatureDetectorAlgorithm.SIFT, second_method=FeatureDetectorAlgorithm.GOOD_FEATURES_TO_TRACK)
    # compare_algorithms_keypoints(first_method=FeatureDetectorAlgorithm.SIFT, second_method=FeatureDetectorAlgorithm.FAST)
    # compare_algorithms_keypoints(first_method=FeatureDetectorAlgorithm.STAR, second_method=FeatureDetectorAlgorithm.FAST)
    # compare_algorithms_keypoints(first_method=FeatureDetectorAlgorithm.GOOD_FEATURES_TO_TRACK, second_method=FeatureDetectorAlgorithm.FAST)
    # compare_algorithms_keypoints(first_method=FeatureDetectorAlgorithm.FAST, second_method=FeatureDetectorAlgorithm.ORB)
