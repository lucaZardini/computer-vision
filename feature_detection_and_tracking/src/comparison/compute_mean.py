import json
import os

from feature_detector.feature_detector import FeatureDetectorAlgorithm


KEYPOINTS_DIRECTORY_COMPARISON = "../../keypoints_comparison/"
KEYPOINTS_COMPARISON_EXTENTION = ".json"
KEYPOINTS_COMPARISON_SEPARATOR = "__2000___"


def compute_mean(first_method: FeatureDetectorAlgorithm, second_method: FeatureDetectorAlgorithm) -> int:

    path_to_file = KEYPOINTS_DIRECTORY_COMPARISON+first_method.value+KEYPOINTS_COMPARISON_SEPARATOR+second_method.value+KEYPOINTS_COMPARISON_EXTENTION
    if not os.path.exists(path_to_file):
        print(path_to_file)
        path_to_file = KEYPOINTS_DIRECTORY_COMPARISON + second_method.value + KEYPOINTS_COMPARISON_SEPARATOR + first_method.value + KEYPOINTS_COMPARISON_EXTENTION
        if not os.path.exists(path_to_file):
            raise FileExistsError(f"File {path_to_file} not found")
    with open(path_to_file, "r") as file:
        info = json.load(file)
        number_of_keypoints_first = info["numberOfKeypointsFirst"]
        number_of_keypoints_second = info["numberOfKeypointsSecond"]
        overlaps = info["overlap"]
        frame_means = []
        for i, overlap in enumerate(overlaps):
            mean_in_frame = overlap/min(number_of_keypoints_first[i], number_of_keypoints_second[i])
            frame_means.append(mean_in_frame)

        return sum(frame_means)/len(frame_means)


if __name__ == "__main__":
    # mean = compute_mean(first_method=FeatureDetectorAlgorithm.FAST, second_method=FeatureDetectorAlgorithm.SIFT)
    # print("fast & sift "+ str(mean))
    # mean = compute_mean(first_method=FeatureDetectorAlgorithm.FAST, second_method=FeatureDetectorAlgorithm.STAR)
    # print("fast & star " + str(mean))
    # mean = compute_mean(first_method=FeatureDetectorAlgorithm.FAST, second_method=FeatureDetectorAlgorithm.GOOD_FEATURES_TO_TRACK)
    # print("fast & gff " + str(mean))
    # mean = compute_mean(first_method=FeatureDetectorAlgorithm.FAST, second_method=FeatureDetectorAlgorithm.ORB)
    # print("fast & orb " + str(mean))
    mean = compute_mean(first_method=FeatureDetectorAlgorithm.SIFT, second_method=FeatureDetectorAlgorithm.STAR)
    print("sift & star " + str(mean))
    mean = compute_mean(first_method=FeatureDetectorAlgorithm.SIFT, second_method=FeatureDetectorAlgorithm.GOOD_FEATURES_TO_TRACK)
    print("sift & gff " + str(mean))
    mean = compute_mean(first_method=FeatureDetectorAlgorithm.SIFT, second_method=FeatureDetectorAlgorithm.ORB)
    print("sift & orb " + str(mean))
    mean = compute_mean(first_method=FeatureDetectorAlgorithm.STAR, second_method=FeatureDetectorAlgorithm.GOOD_FEATURES_TO_TRACK)
    print("star & gff " + str(mean))
    mean = compute_mean(first_method=FeatureDetectorAlgorithm.STAR, second_method=FeatureDetectorAlgorithm.ORB)
    print("star & orb " + str(mean))
    mean = compute_mean(first_method=FeatureDetectorAlgorithm.GOOD_FEATURES_TO_TRACK, second_method=FeatureDetectorAlgorithm.ORB)
    print("gff & orb " + str(mean))

