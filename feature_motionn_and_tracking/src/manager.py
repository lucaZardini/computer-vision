from typing import List

import cv2

from feature_detector.feature_detector import FeatureDetectorAlgorithm, FeatureDetector, FeatureDetectorBuilder, \
    SiftDetector, HarrisCornerDetector, GoodFeaturesToTrackDetector
from feature_tracking.feature_tracking import KalmanFilter, LucasKanadeOpticalFlowTracker

import numpy as np

from matplotlib import pyplot as plt


class TrackManager:

    SAMPLING = 30

    @staticmethod
    def track_video(video_path):
        # create the feature detectors and the feature trackers
        # feature_detectors: List[FeatureDetector] = []
        # for algorithm in FeatureDetectorAlgorithm:
        #     feature_detectors = FeatureDetectorBuilder.build(algorithm)
        cap = cv2.VideoCapture(video_path)
        frame_index = 0
        gff = GoodFeaturesToTrackDetector()
        lk_tracker = LucasKanadeOpticalFlowTracker()
        while cap.isOpened():

            ret, frame = cap.read()

            if not ret:
                break

            if frame_index % TrackManager.SAMPLING == 0:
                gff.image = frame
                corners = gff.detect()
                lk_tracker.initialize(frame, corners)

            else:
                corners, status, err = lk_tracker.track(frame)

            frame_copy = frame.copy()
            int_corners = corners.astype(int)
            for i, corner in enumerate(int_corners):
                x, y = corner.ravel()
                color = np.float64([i, 2 * i, 255 - i])
                cv2.circle(frame_copy, (x, y), 20, (0, 255, 0), thickness=20)

            cv2.imshow('GFF', frame_copy)

            if cv2.waitKey(1) == ord('q') or not ret:
                break

            frame_index += 1

        cap.release()
        cv2.destroyAllWindows()
