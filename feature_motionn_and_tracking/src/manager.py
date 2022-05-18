from typing import List

import cv2

from feature_detector.feature_detector import FeatureDetectorAlgorithm, FeatureDetector, FeatureDetectorBuilder, \
    SiftDetector
from feature_tracking.feature_tracking import KalmanFilter


class TrackManager:
    @staticmethod
    def track_video(video_path):
        # create the feature detectors and the feature trackers
        # feature_detectors: List[FeatureDetector] = []
        # for algorithm in FeatureDetectorAlgorithm:
        #     feature_detectors = FeatureDetectorBuilder.build(algorithm)
        cap = cv2.VideoCapture(video_path)
        sift_detector = SiftDetector()
        kalman_filter = KalmanFilter()
        for i in range(1000):
            ret, frame = cap.read()
            if i % 10 == 0:
                sift_detector.image = frame
                features, image = sift_detector.detect()
            kalman_filter.image = frame
            predicted = kalman_filter.track()
            cv2.imshow("Bohh", predicted)
            cv2.waitKey(0)

