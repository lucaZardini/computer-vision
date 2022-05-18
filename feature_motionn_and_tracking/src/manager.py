from typing import List

import cv2

from feature_detector.feature_detector import FeatureDetectorAlgorithm, FeatureDetector, FeatureDetectorBuilder, \
    SiftDetector


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
        n_frames = 0
        ret = True
        for i in range(1000):
            ret, frame = cap.read()
            # print(ret)
            # sift_detector = SiftDetector()
            # sift_detector.image = frame
            # features, image = sift_detector.detect()
            n_frames += 1

        print(n_frames)
