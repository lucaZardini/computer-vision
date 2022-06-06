import json
import os

from feature_detector.feature_detector import FeatureDetectorAlgorithm


KEYPOINTS_DIRECTORY_COMPARISON = "../../keypoints_comparison/"
KEYPOINTS_COMPARISON_EXTENTION = ".json"
KEYPOINTS_COMPARISON_SEPARATOR = "_"


def compute_mean(first_method: FeatureDetectorAlgorithm, second_method: FeatureDetectorAlgorithm) -> int:

    path_to_file = KEYPOINTS_DIRECTORY_COMPARISON+first_method.value+KEYPOINTS_COMPARISON_SEPARATOR+second_method.value+KEYPOINTS_COMPARISON_EXTENTION
    if not os.path.exists(path_to_file):
        path_to_file = KEYPOINTS_DIRECTORY_COMPARISON + second_method.value + KEYPOINTS_COMPARISON_SEPARATOR + first_method.value + KEYPOINTS_COMPARISON_EXTENTION
        if not os.path.exists(path_to_file):
            raise FileExistsError("File not found")
    with open(path_to_file, "r") as file:
        info = json.load(file)
        number_of_keypoints = info["numberOfKeypoints"]
        overlaps = info["overlap"]
        frame_means = []
        for overlap in overlaps:
            mean_in_frame = overlap/number_of_keypoints
            frame_means.append(mean_in_frame)

        return sum(frame_means)/len(frame_means)


if __name__ == "__main__":
    mean = compute_mean(first_method=FeatureDetectorAlgorithm.FAST, second_method=FeatureDetectorAlgorithm.ORB)
