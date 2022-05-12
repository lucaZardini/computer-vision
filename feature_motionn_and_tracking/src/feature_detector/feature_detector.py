from abc import ABC, abstractmethod
from enum import Enum


class FeatureDetectorAlgorithm(Enum):
    pass


class FeatureDetector(ABC):
    def __init__(self, image):
        self.image = image

    @abstractmethod
    def detect(self, algorithm: FeatureDetectorAlgorithm):
        pass

