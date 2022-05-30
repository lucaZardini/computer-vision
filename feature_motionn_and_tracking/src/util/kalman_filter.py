import cv2

import numpy as np


class KalmanFilter:
    def __init__(self):
        self.kalman_filter = cv2.KalmanFilter(4, 2)
        self.unmatched_detection_number = 0
        self.predicted_points = None
        self.initialize()

    def predict(self):
        self.predicted_points = self.kalman_filter.predict()
        return self.predicted_points

    def correct(self, point):
        self.kalman_filter.correct(point)

    def initialize(self):
        self.kalman_filter.measurementMatrix = \
            np.array([
                [1, 0, 0, 0],
                [0, 1, 0, 0]], np.float32)
        # A
        self.kalman_filter.transitionMatrix = \
            np.array([
                [1, 0, 1, 0],
                [0, 1, 0, 1],
                [0, 0, 1, 0],
                [0, 0, 0, 1]], np.float32)
        # w
        self.kalman_filter.processNoiseCov = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]], np.float32) * 0.003
        # v
        self.kalman_filter.measurementNoiseCov = \
            np.array([
                [1, 0],
                [0, 1]], np.float32) * 1
