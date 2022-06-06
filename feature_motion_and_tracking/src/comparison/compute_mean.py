from feature_detector.feature_detector import FeatureDetectorAlgorithm


KEYPOINTS_DIRECTORY_COMPARISON = "../../keypoints_comparison/"
KEYPOINTS_COMPARISON_EXTENTION = ".json"
KEYPOINTS_COMPARISON_SEPARATOR = "_"


def compute_mean(first_method: FeatureDetectorAlgorithm, second_method: FeatureDetectorAlgorithm):
    # verifica se esiste il file
    # piglialo e calcola la media

    pass


if __name__ == "__main__":
    compute_mean(first_method=FeatureDetectorAlgorithm.BRIEF, second_method=FeatureDetectorAlgorithm.ORB)
