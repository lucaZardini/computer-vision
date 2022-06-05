import json
from typing import List, Tuple

from feature_detector.feature_detector import FeatureDetectorAlgorithm
import numpy as np


KEYPOINTS_DIRECTORY = "../../keypoints/"
KEYPOINTS_EXTENTION = ".npy"
NUMBER_OF_KEYPOINTS = 10000

KEYPOINTS_DIRECTORY_COMPARISON = "../../keypoints_comparison/"
KEYPOINTS_COMPARISON_EXTENTION = ".json"
KEYPOINTS_COMPARISON_SEPARATOR = "_"


def save_keypoints(overlap: List[int], number_of_keypoints: int, first_method: FeatureDetectorAlgorithm, second_method: FeatureDetectorAlgorithm):
    with open(KEYPOINTS_DIRECTORY_COMPARISON+first_method.value+KEYPOINTS_COMPARISON_SEPARATOR+second_method.value+KEYPOINTS_COMPARISON_EXTENTION, "w") as out_file:
        json_output = {
            "numberOfKeypoints": number_of_keypoints,
            "overlap": overlap
        }
        json.dump(json_output, out_file)


def compare_keypoints(keypoint_first: List[Tuple[int, int]], keypoint_second: List[Tuple[int, int]], number_of_keypoints: int) -> int:
    keypoints_in_both = 0
    if len(keypoint_first) < number_of_keypoints or len(keypoint_second) < number_of_keypoints:
        raise ValueError(f"The number {number_of_keypoints} is too large")
    for keypoint_number in range(number_of_keypoints):
        keypoint = keypoint_first[keypoint_number]
        if keypoint in keypoint_second:
            keypoints_in_both += 1
    return keypoints_in_both


def compare_algorithms_keypoints(first_method: FeatureDetectorAlgorithm, second_method: FeatureDetectorAlgorithm):
    if first_method == second_method:
        return ValueError("The two methods must be different")

    first_method_keypoints = np.load(KEYPOINTS_DIRECTORY+first_method.value+KEYPOINTS_EXTENTION, allow_pickle=True)
    second_method_keypoints = np.load(KEYPOINTS_DIRECTORY+second_method.value+KEYPOINTS_EXTENTION, allow_pickle=True)

    keypoints_comparison = []

    for frame in range(len(first_method_keypoints)):
        frame_keypoints_first_method = first_method_keypoints[frame]
        frame_keypoints_second_method = second_method_keypoints[frame]
        overlap = compare_keypoints(frame_keypoints_first_method, frame_keypoints_second_method, NUMBER_OF_KEYPOINTS)
        keypoints_comparison.append(overlap)

    save_keypoints(keypoints_comparison, NUMBER_OF_KEYPOINTS, first_method, second_method)


if __name__ == "__main__":
    # TODO: compare all methods
    pass
