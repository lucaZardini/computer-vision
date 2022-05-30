from abc import ABC, abstractmethod
from enum import Enum

import cv2
from scipy.optimize import linear_sum_assignment

from util.kalman_filter import KalmanFilter
import numpy as np


class FeatureTrackingAlgorithm(Enum):
    KALMAN_FILTER = "kalman_filter"
    LK = "lucas_kanade"


class FeatureTrackingBuilder:
    @staticmethod
    def build(algorithm: FeatureTrackingAlgorithm):
        if algorithm == FeatureTrackingAlgorithm.KALMAN_FILTER:
            return TrackUsingKalmanFilters()
        if algorithm == FeatureTrackingAlgorithm.LK:
            return LucasKanadeOpticalFlowTracker()
        else:
            raise ValueError


class FeatureTracking(ABC):

    @abstractmethod
    def track(self, frame):
        pass

    @abstractmethod
    def initialize(self, frame, corner):
        pass

    def predict(self):
        pass


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


class TrackUsingKalmanFilters(FeatureTracking):

    def __init__(self):
        self.kalman_filters = []

    def predict(self):
        points = []
        for k in self.kalman_filters:
            # for each kalman filter, predict and return the result only if the tracker has at least a match in the last 4 frames
            if k.unmatched_detection_number <= 4:
                # predict the next point
                p = k.predict()
                p = np.array([coordinate.astype(float) for coordinate in p[:2]], dtype=np.float32)
                # correct the kalman filter with the prediction
                k.correct(p)
                points.append(p)

        return np.array(points)

    def initialize(self, frames, features):
        self.kalman_filters = []
        # for each feature creates a new tracker
        for feature in features:
            kalman_filter = KalmanFilter()
            x, y = feature[0]
            point = np.array([[np.float32(x)], [np.float32(y)], [np.float32(0)], [np.float32(0)]])
            # set the initial value of the point equal in the kalman filter
            kalman_filter.kalman_filter.statePost = point
            kalman_filter.kalman_filter.statePre = point

            kalman_filter.correct(np.array([[np.float32(x)], [np.float32(y)]]))
            kalman_filter.predict()
            # save the kalman filter created
            self.kalman_filters.append(kalman_filter)

    def track(self, input_points):
        intersection_over_union_matrix = np.zeros((len(self.kalman_filters), len(input_points)), dtype=np.float32)
        for t, trk in enumerate(self.kalman_filters):
            for d, det in enumerate(input_points):
                intersection_over_union_matrix[t, d] = self._box_iou2(trk.predicted_points, det)

        # apply hungarian algorithm to the intersection over union matrix
        matched_idx = linear_sum_assignment(-intersection_over_union_matrix)

        # matched_idx is now a vector of two columns
        matched_idx = np.transpose(np.asarray(matched_idx))

        unmatched_trackers, unmatched_detections = [], []

        # for each tracker, verify if there is a match between the predicted points
        # and the detected points in the current frame
        # If there is no match, the tracker is considered as unmatched tracker

        for t, trk in enumerate(self.kalman_filters):
            if t not in matched_idx[:, 0]:
                unmatched_trackers.append(t)

        # for each detection, verify if there is a tracker that matches.
        # If there is no matches, the point is considered as an unmatched detection

        for d, det in enumerate(input_points):
            if d not in matched_idx[:, 1]:
                unmatched_detections.append(d)

        matches = []

        # For each match, verify that the match is over a certain threshold (default is 0.3).
        # If the overlap is lower than the threshold, both the tracker and the detection are considered as unmatched

        for m in matched_idx:

            if intersection_over_union_matrix[m[0], m[1]] < 0.3:
                unmatched_trackers.append(m[0])
                unmatched_detections.append(m[1])
            else:
                matches.append(m.reshape(1, 2))

        if len(matches) == 0:
            matches = np.empty((0, 2), dtype=int)
        else:
            matches = np.concatenate(matches, axis=0)

        unmatched_detections = np.array(unmatched_detections)
        unmatched_trackers = np.array(unmatched_trackers)

        # predict the new position of the point.
        if matches.size > 0:
            for trk_idx, det_idx in matches:
                # retrieve the tracker
                tracker = self.kalman_filters[trk_idx]
                # set the value of how many times it is considered as unmatched tracker to 0
                tracker.unmatched_detection_number = 0
                input_point = np.atleast_2d(input_points[det_idx][0]).T
                # correct the tracker with the detection
                tracker.correct(input_point)
                # predict the new position
                tracker.predict()

        # Deal with unmatched detections
        if len(unmatched_detections) > 0:
            for idx in unmatched_detections:
                # create a new tracker
                tracker = KalmanFilter()
                x, y = input_points[idx][0]
                point = np.array([[np.float32(x)], [np.float32(y)], [np.float32(0)], [np.float32(0)]])
                # set the default position of the point equal to the position of the point itself
                tracker.kalman_filter.statePost = point
                # correct and predict
                tracker.correct(point=np.array([[np.float32(x)], [np.float32(y)]]))
                tracker.predict()
                # save the tracker
                self.kalman_filters.append(tracker)

        # Deal with unmatched tracks
        if len(unmatched_trackers) > 0:
            for trk_idx in unmatched_trackers:
                tracker = self.kalman_filters[trk_idx]
                # increase the number of consecutive times of unmatched detection by 1
                tracker.unmatched_detection_number += 1

    @staticmethod
    def _box_iou2(a, b):
        """
        Helper function to calculate the ratio between intersection and the union of two boxes a and b
        :param a: The first point
        :param b: The second point
        """
        a = np.array([coordinate.astype(float) for coordinate in a[:2]], dtype=np.float32)
        b = np.array([coordinate.astype(float) for coordinate in b[:2]], dtype=np.float32)
        a_x, a_y = a

        a_x = a_x[0]
        a_y = a_y[0]
        b_x, b_y = b[0]

        # distance from the point to right, left, up and bottom
        box_value = 100

        # width intersection
        w_intsec = np.maximum(0, (
                    np.minimum(a_x + box_value, b_x + box_value) - np.maximum(a_x - box_value, b_x - box_value)))

        # height intersection
        h_intsec = np.maximum(0, (
                    np.minimum(a_y + box_value, b_y + box_value) - np.maximum(a_y - box_value, b_y - box_value)))

        s_intsec = w_intsec * h_intsec
        s_a = 4 * box_value * box_value
        s_b = 4 * box_value * box_value
        return float(s_intsec) / (s_a + s_b - s_intsec)
