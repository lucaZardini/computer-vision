from __future__ import absolute_import, annotations

from abc import ABC, abstractmethod
from enum import Enum
import numpy as np

import cv2


class FeatureDetectorAlgorithm(Enum):
    SIFT = "sift"
    HARRIS_CORNER = "harris_corner"
    GOOD_FEATURES_TO_TRACK = "gff"
    ORB = "orb"
    FAST = "fast"
    BRIEF = "brief"


class FeatureDetectorBuilder:
    @staticmethod
    def build(algorithm: FeatureDetectorAlgorithm):
        if algorithm == FeatureDetectorAlgorithm.SIFT:
            return SiftDetector()
        elif algorithm == FeatureDetectorAlgorithm.HARRIS_CORNER:
            return HarrisCornerDetector()
        elif algorithm == FeatureDetectorAlgorithm.GOOD_FEATURES_TO_TRACK:
            return GoodFeaturesToTrackDetector()
        elif algorithm == FeatureDetectorAlgorithm.ORB:
            return ORBDetector()
        elif algorithm == FeatureDetectorAlgorithm.FAST:
            return FASTDetector()
        elif algorithm == FeatureDetectorAlgorithm.BRIEF:
            return BriefDetector()
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
        sift = cv2.SIFT_create()
        kp_obj, dsc_obj = sift.detectAndCompute(self.image, None)
        return kp_obj

    def name(self):
        return FeatureDetectorAlgorithm.SIFT.value


class HarrisCornerDetector(FeatureDetector):

    def detect(self):
        if not self.is_image_set:
            raise AttributeError("The image has not been set")
        gray_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        gray_image = np.float32(gray_image)

        return cv2.cornerHarris(gray_image, blockSize=2, ksize=3, k=0.04)

    def name(self):
        return FeatureDetectorAlgorithm.HARRIS_CORNER.value


class GoodFeaturesToTrackDetector(FeatureDetector):

    def detect(self):
        if not self.is_image_set:
            raise AttributeError("The image has not been set")
        gray_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        return cv2.goodFeaturesToTrack(gray_image, maxCorners=1000, qualityLevel=0.01, minDistance=10, blockSize=3)

    def name(self):
        return FeatureDetectorAlgorithm.GOOD_FEATURES_TO_TRACK.value


class ORBDetector(FeatureDetector):

    def detect(self):
        if not self.is_image_set:
            raise AttributeError("The image has not been set")
        gray_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)

        orb = cv2.ORB_create(nfeatures=2000)

        return orb.detectAndCompute(gray_image, None)

    def name(self):
        return FeatureDetectorAlgorithm.ORB.value


class FASTDetector(FeatureDetector):

    def detect(self):
        if not self.is_image_set:
            raise AttributeError("The image has not been set")
        gray_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        fast = cv2.FastFeatureDetector_create()

        fast.setNonmaxSuppression(False)

        return fast.detect(gray_image, None)

    def name(self):
        return FeatureDetectorAlgorithm.FAST.value


class BriefDetector(FeatureDetector):

    def detect(self):
        if not self.is_image_set:
            raise AttributeError("The image has not been set")
        star = cv2.xfeatures2d.StarDetector_create()
        brief = cv2.xfeatures2d.BriefDescriptorExtractor_create()

        kp = star.detect(self.image, None)
        kp, des = brief.compute(self.image, kp)
        return kp

    def name(self):
        return FeatureDetectorAlgorithm.BRIEF.value
