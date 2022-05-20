from __future__ import absolute_import, annotations

from abc import ABC, abstractmethod
from enum import Enum
import numpy as np

import cv2


class FeatureDetectorAlgorithm(Enum):
    SIFT = "sift"
    HARRIS_CORNER = "harris_corner"


class FeatureDetectorBuilder:
    @staticmethod
    def build(algorithm: FeatureDetectorAlgorithm):
        if algorithm == FeatureDetectorAlgorithm.SIFT:
            return SiftDetector()
        elif algorithm == FeatureDetectorAlgorithm.HARRIS_CORNER:
            return HarrisCornerDetector()
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
        sift = cv2.SIFT_create()
        kp_obj, dsc_obj = sift.detectAndCompute(self.image, None)
        return kp_obj, cv2.drawKeypoints(self.image, kp_obj, self.image, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)


class HarrisCornerDetector(FeatureDetector):

    def detect(self):
        if not self.is_image_set:
            raise AttributeError("The image has not been set")
        gray_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        gray_image = np.float32(gray_image)

        return cv2.cornerHarris(gray_image, blockSize=2, ksize=3, k=0.04)


class GoodFeaturesToTrackDetector(FeatureDetector):

    def detect(self):
        if not self.is_image_set:
            raise AttributeError("The image has not been set")
        gray_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        return cv2.goodFeaturesToTrack(gray_image, maxCorners=100, qualityLevel=0.01, minDistance=10, blockSize=3)
