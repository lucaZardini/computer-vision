from typing import List

from feature_detector.feature_detector import FeatureDetectorAlgorithm, FeatureDetector, FeatureDetectorBuilder


class TrackManager:
    @staticmethod
    def track_video(video):
        # create the feature detectors and the feature trackers
        feature_detectors: List[FeatureDetector] = []
        for algorithm in FeatureDetectorAlgorithm:
            feature_detectors = FeatureDetectorBuilder.build(algorithm)
