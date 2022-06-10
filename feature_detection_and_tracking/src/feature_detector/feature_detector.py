from __future__ import absolute_import, annotations

from abc import ABC, abstractmethod
from enum import Enum

import cv2


class FeatureDetectorAlgorithm(Enum):
    SIFT = "sift"
    GOOD_FEATURES_TO_TRACK = "gftt"
    ORB = "orb"
    FAST = "fast"
    STAR = "star"


class FeatureDetectorBuilder:
    @staticmethod
    def build(algorithm: FeatureDetectorAlgorithm):
        if algorithm == FeatureDetectorAlgorithm.SIFT:
            return SiftDetector()
        elif algorithm == FeatureDetectorAlgorithm.GOOD_FEATURES_TO_TRACK:
            return GoodFeaturesToTrackDetector()
        elif algorithm == FeatureDetectorAlgorithm.ORB:
            return ORBDetector()
        elif algorithm == FeatureDetectorAlgorithm.FAST:
            return FASTDetector()
        elif algorithm == FeatureDetectorAlgorithm.STAR:
            return StarDetector()
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
    def name(self):
        pass

    @abstractmethod
    def detect(self):
        pass


class SiftDetector(FeatureDetector):

    def detect(self):
        if not self.is_image_set:
            raise AttributeError("The image has not been set")
        sift = cv2.SIFT_create(nfeatures=2000)
        kp, dsc = sift.detectAndCompute(self.image, None)
        return kp

    def name(self):
        return FeatureDetectorAlgorithm.SIFT.value


class GoodFeaturesToTrackDetector(FeatureDetector):

    def detect(self):
        if not self.is_image_set:
            raise AttributeError("The image has not been set")
        gray_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        kp = cv2.goodFeaturesToTrack(gray_image, maxCorners=2000, qualityLevel=0.01, minDistance=10, blockSize=3)

        return kp

    def name(self):
        return FeatureDetectorAlgorithm.GOOD_FEATURES_TO_TRACK.value


class ORBDetector(FeatureDetector):

    def detect(self):
        if not self.is_image_set:
            raise AttributeError("The image has not been set")
        gray_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)

        orb = cv2.ORB_create(nfeatures=2000)
        kp, desc = orb.detectAndCompute(gray_image, None)
        return kp

    def name(self):
        return FeatureDetectorAlgorithm.ORB.value


class FASTDetector(FeatureDetector):

    def detect(self):
        if not self.is_image_set:
            raise AttributeError("The image has not been set")
        gray_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        fast = cv2.FastFeatureDetector_create()

        fast.setNonmaxSuppression(False)
        kp = fast.detect(gray_image, None)
        return kp

    def name(self):
        return FeatureDetectorAlgorithm.FAST.value


class StarDetector(FeatureDetector):

    def detect(self):
        if not self.is_image_set:
            raise AttributeError("The image has not been set")
        star = cv2.xfeatures2d.StarDetector_create()

        kp = star.detect(self.image, None)
        if len(kp) > 2000:
            kp = kp[:2000]
        return kp

    def name(self):
        return FeatureDetectorAlgorithm.STAR.value
