from abc import ABC, abstractmethod
from enum import Enum
import numpy as np

import cv2


class FeatureTrackingAlgorithm(Enum):
    KALMAN_FILTER = "kalman_filter"


class FeatureTracking(ABC):
    def __init__(self, image=None):
        self.image = image

    @abstractmethod
    def track(self, algorithm: FeatureTrackingAlgorithm):
        pass


class KalmanFilter(FeatureTracking):

    def track(self, algorithm: FeatureTrackingAlgorithm = FeatureTrackingAlgorithm.KALMAN_FILTER):
        kalman_filter = cv2.KalmanFilter(4, 2)
        kalman_filter.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
        kalman_filter.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
        kalman_filter.processNoiseCov = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)*0.03
        kalman_filter.correct(self.image)
        prediction = kalman_filter.predict()
        return prediction
