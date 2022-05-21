from abc import abstractmethod

from feature_detector.feature_detector import FeatureDetector, FeatureDetectorAlgorithm, \
    FeatureDetectorBuilder
from feature_tracking.feature_tracking import FeatureTracking, FeatureTrackingAlgorithm, FeatureTrackingBuilder

import cv2

import numpy as np


class Tracker:

    SAMPLING = 30

    def __init__(self, detector: FeatureDetector, tracking: FeatureTracking, video):
        self.detector = detector
        self.tracking = tracking
        self.video = video
        self._result = None

    @property
    def result(self):
        if self._result:
            return self._result
        else:
            return self.track()

    @abstractmethod
    def track(self):
        pass


class TrackerBuilder:

    @staticmethod
    def build(detector_algorithm: FeatureDetectorAlgorithm, tracker_algorithm: FeatureTrackingAlgorithm, video: str):
        detector = FeatureDetectorBuilder.build(detector_algorithm)
        tracker = FeatureTrackingBuilder.build(tracker_algorithm)
        return Tracker(detector=detector, tracking=tracker, video=video)


class TrackGFFwithLK(Tracker):

    def track(self):
        cap = cv2.VideoCapture(self.video)
        frame_index = 0

        while cap.isOpened():

            ret, frame = cap.read()

            if not ret:
                break

            if frame_index % self.SAMPLING == 0:
                self.detector.image = frame
                corners = self.detector.detect()
                self.tracking.initialize(frame, corners)

            else:
                corners, status, err = self.tracking.track(frame)

            frame_copy = frame.copy()
            int_corners = corners.astype(int)
            for i, corner in enumerate(int_corners):
                x, y = corner.ravel()
                cv2.circle(frame_copy, (x, y), 20, (0, 255, 0), thickness=20)

            cv2.imshow('GFF', frame_copy)

            if cv2.waitKey(1) == ord('q') or not ret:
                break

            frame_index += 1

        cap.release()
        cv2.destroyAllWindows()


class SIFTwithLK(Tracker):

    def track(self):
        cap = cv2.VideoCapture(self.video)
        frame_index = 0

        while cap.isOpened():

            ret, frame = cap.read()

            if not ret:
                break

            if frame_index % self.SAMPLING == 0:
                self.detector.image = frame
                features = self.detector.detect()
                features = np.array([[k.pt] for k in features], dtype=np.float32)
                self.tracking.initialize(frame, features)

            else:
                features, status, err = self.tracking.track(frame)

            frame_copy = frame.copy()
            int_corners = features.astype(int)
            for i, corner in enumerate(int_corners):
                x, y = corner.ravel()
                cv2.circle(frame_copy, (x, y), 20, (0, 255, 0), thickness=20)

            cv2.imshow('GFF', frame_copy)

            if cv2.waitKey(1) == ord('q') or not ret:
                break

            frame_index += 1

        cap.release()
        cv2.destroyAllWindows()
