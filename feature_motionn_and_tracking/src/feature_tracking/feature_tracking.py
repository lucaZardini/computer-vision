from abc import ABC, abstractmethod
from enum import Enum

import cv2


class FeatureTrackingAlgorithm(Enum):
    KALMAN_FILTER = "kalman_filter"


class FeatureTracking(ABC):
    def __init__(self, image):
        self.image = image

    @abstractmethod
    def track(self, algorithm: FeatureTrackingAlgorithm):
        pass



class KalmanFilter(FeatureTracking):

    def track(self, algorithm: FeatureTrackingAlgorithm = FeatureTrackingAlgorithm.KALMAN_FILTER):
        kalman_filter = cv2.KalmanFilter()
        kalman_filter.predict()
        kalman_filter.correct()