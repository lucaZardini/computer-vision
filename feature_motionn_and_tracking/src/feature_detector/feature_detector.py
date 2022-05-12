from __future__ import absolute_import, annotations

from abc import ABC, abstractmethod
from enum import Enum

import cv2


class FeatureDetectorAlgorithm(Enum):
    SIFT = "sift"


class FeatureDetectorBuilder:
    @staticmethod
    def build(algorithm: FeatureDetectorAlgorithm):
        if algorithm == FeatureDetectorAlgorithm.SIFT:
            return SiftDetector()
        else:
            raise ValueError()


class FeatureDetector(ABC):
    def __init__(self, image=None):
        self.image = image

    def with_image(self, image):
        self.image = image

    @property
    def is_image_set(self) -> bool:
        if self.image is None:
            return False
        else:
            return True

    @abstractmethod
    def detect(self):
        pass


class SiftDetector(FeatureDetector):

    def detect(self):
        if not self.is_image_set:
            raise AttributeError("The image has not been set")
        else:
            sift = cv2.SIFT_create()
            kp = sift.detect(self.image, None)
