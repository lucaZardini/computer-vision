from abc import ABC, abstractmethod
from enum import Enum
import numpy as np

import cv2


class FeatureTrackingAlgorithm(Enum):
    KALMAN_FILTER = "kalman_filter"
    LK = "lucas_kanade"


class FeatureTracking(ABC):

    @abstractmethod
    def track(self, frame):
        pass


class KalmanFilter(FeatureTracking):

    def __init__(self, features):
        self.features = features

    def track(self, frame):
        kalman_filter = cv2.KalmanFilter(4, 2)
        kalman_filter.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
        kalman_filter.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
        kalman_filter.processNoiseCov = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)*0.03
        kalman_filter.correct(self.features)
        self.features = kalman_filter.predict()
        return self.features


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
