from abc import ABC, abstractmethod
from enum import Enum


class FeatureTrackingAlgorithm(Enum):
    pass


class FeatureTracking(ABC):
    def __init__(self, image):
        self.image = image

    @abstractmethod
    def track(self, algorithm: FeatureTrackingAlgorithm):
        pass
