from abc import ABC, abstractmethod
from enum import Enum

import cv2

from feature_tracking.kf import GlobalTracker


class FeatureTrackingAlgorithm(Enum):
    # KALMAN_FILTER = "kalman_filter"
    LK = "lucas_kanade"


class FeatureTrackingBuilder:
    @staticmethod
    def build(algorithm: FeatureTrackingAlgorithm):
        # if algorithm == FeatureTrackingAlgorithm.KALMAN_FILTER:
        #     return KalmanFilter()
        if algorithm == FeatureTrackingAlgorithm.LK:
            return LucasKanadeOpticalFlowTracker()
        else:
            raise ValueError


class FeatureTracking(ABC):

    @abstractmethod
    def track(self, frame):
        pass

    @abstractmethod
    def initialize(self, frame, corner):
        pass


class KalmanFilter(FeatureTracking):

    def __init__(self):
        self.features = None

    def initialize(self, frame, corner):
        self.global_tracker = GlobalTracker()
        self.frame = frame
        self.bboxes = corner

    def track(self, frame):

        classes = ['person' for _ in range(0, len(self.bboxes))]
        frame_scores = []
        for i in range(len(self.bboxes)):
            frame_scores.append(0)


        tracker_id_list, bounding_boxes, centres, scores, class_labels = self.global_tracker.pipeline(self.bboxes,
                                                                                                 frame_scores,
                                                                                                 classes,
                                                                                                 frame)
        self.bboxes = bounding_boxes
        self.frame = frame

        return tracker_id_list, bounding_boxes, centres, scores, class_labels


class LucasKanadeOpticalFlowTracker(FeatureTracking):

    def __init__(self):
        self.previous_frame = None
        self.previous_corners = None

    def initialize(self, frame, corners):
        self.previous_frame = frame
        self.previous_corners = corners

    def track(self, current_frame):
        corners, status, err = cv2.calcOpticalFlowPyrLK(self.previous_frame, current_frame, self.previous_corners, None)
        self.previous_frame = current_frame.copy()
        self.previous_corners = corners
        return corners, status, err
