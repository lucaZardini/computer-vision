import os
from abc import abstractmethod

from feature_detector.feature_detector import FeatureDetector, FeatureDetectorAlgorithm, \
    FeatureDetectorBuilder
from feature_tracking.feature_tracking import FeatureTracking, FeatureTrackingAlgorithm, FeatureTrackingBuilder

import cv2

import numpy as np


class Tracker:

    SAMPLING = 30

    KEYPOINT_PATH = "../keypoints/"

    def __init__(self, detector: FeatureDetector, tracking: FeatureTracking, video: str, online: bool):
        self.detector = detector
        self.tracking = tracking
        self.video = video
        self.online = online
        self._keypoints = None
        self._result = None

    @property
    def result(self):
        if self._result:
            return self._result
        else:
            return self.track()

    @property
    def file_with_keypoints_is_present(self) -> bool:
        if not self.online:
            path_to_file = self.KEYPOINT_PATH+self.detector.name()+".npy"
            return os.path.exists(path_to_file)
        return True

    def get_keypoints(self, frame_index):
        if self._keypoints is not None:
            return self._keypoints[frame_index]
        else:
            path_to_file = self.KEYPOINT_PATH + self.detector.name() + ".npy"
            self._keypoints = np.load(path_to_file, allow_pickle=True)
            return self._keypoints[frame_index]

    @abstractmethod
    def track(self):
        pass


class TrackerBuilder:

    @staticmethod
    def build(detector_algorithm: FeatureDetectorAlgorithm, tracker_algorithm: FeatureTrackingAlgorithm, video: str, online: bool):
        detector = FeatureDetectorBuilder.build(detector_algorithm)
        tracker = FeatureTrackingBuilder.build(tracker_algorithm)
        if tracker_algorithm == FeatureTrackingAlgorithm.LK:

            return LucasKanadeTracker(detector=detector, tracking=tracker, video=video, online=online)

            # if detector_algorithm == FeatureDetectorAlgorithm.GOOD_FEATURES_TO_TRACK:
            #     return TrackGFFwithLK(detector=detector, tracking=tracker, video=video, online=online)
            #
            # elif detector_algorithm == FeatureDetectorAlgorithm.SIFT:
            #     return SIFTwithLK(detector=detector, tracking=tracker, video=video, online=online)
            #
            # elif detector_algorithm == FeatureDetectorAlgorithm.ORB:
            #     return ORBwithLK(detector=detector, tracking=tracker, video=video, online=online)
            #
            # elif detector_algorithm == FeatureDetectorAlgorithm.FAST:
            #     return FASTwithLK(detector=detector, tracking=tracker, video=video, online=online)
            #
            # elif detector_algorithm == FeatureDetectorAlgorithm.STAR:
            #     return STARwithLK(detector=detector, tracking=tracker, video=video, online=online)

            # else:
            #     pass
        else:
            pass


class LucasKanadeTracker(Tracker):

    def track(self):
        cap = cv2.VideoCapture(self.video)
        frame_index = 0

        while cap.isOpened():

            ret, frame = cap.read()

            if not ret:
                break

            if frame_index % self.SAMPLING == 0:
                if self.online:
                    self.detector.image = frame
                    features = self.detector.detect()
                    if self.detector.name() != FeatureDetectorAlgorithm.GOOD_FEATURES_TO_TRACK.value:
                        features = np.array([[k.pt] for k in features], dtype=np.float32)
                else:
                    if not self.file_with_keypoints_is_present:
                        raise FileExistsError("The file does not exists")
                    features = self.get_keypoints(frame_index)
                self.tracking.initialize(frame, features)
            else:
                features, status, err = self.tracking.track(frame)

            frame_copy = frame.copy()
            int_features = features.astype(int)
            for i, corner in enumerate(int_features):
                x, y = corner.ravel()
                color = np.float64([i, 2 * i, 255 - i])
                cv2.circle(frame_copy, (x, y), 20, color, thickness=4)

            cv2.imshow(f'{self.detector.name()} and LK', frame_copy)

            if cv2.waitKey(1) == ord('q') or not ret:
                break

            frame_index += 1

        cap.release()
        cv2.destroyAllWindows()


class KalmanFilterTracker(Tracker):

    def track(self):
        cap = cv2.VideoCapture(self.video)
        frame_index = 0

        while cap.isOpened():

            ret, frame = cap.read()

            if not ret:
                break

            if frame_index % self.SAMPLING == 0:
                if self.online:
                    self.detector.image = frame
                    features = self.detector.detect()
                    if self.detector.name() != FeatureDetectorAlgorithm.GOOD_FEATURES_TO_TRACK.value:
                        features = np.array([[k.pt] for k in features], dtype=np.float32)
                    features = features[:40]
                else:
                    if not self.file_with_keypoints_is_present:
                        raise FileExistsError("The file does not exists")
                    features = self.get_keypoints(frame_index)
                    features = features[:40]
                self.tracking.initialize(frame, features)
            elif (frame_index % self.SAMPLING) % 2 == 0:
                if self.online:
                    self.detector.image = frame
                    features = self.detector.detect()
                    if self.detector.name() != FeatureDetectorAlgorithm.GOOD_FEATURES_TO_TRACK.value:
                        features = np.array([[k.pt] for k in features], dtype=np.float32)
                    features = features[:40]
                else:
                    if not self.file_with_keypoints_is_present:
                        raise FileExistsError("The file does not exists")
                    features = self.get_keypoints(frame_index)
                    features = features[:40]
                self.tracking.track(features)
            else:
                features = self.tracking.predict()

            frame_copy = frame.copy()
            int_features = features.astype(int)
            for i, corner in enumerate(int_features):
                x, y = corner.ravel()
                color = np.float64([i, 2 * i, 255 - i])
                cv2.circle(frame_copy, (x, y), 20, color, thickness=20)

            cv2.imshow(f'{self.detector.name()} with KF', frame_copy)

            if cv2.waitKey(1) == ord('q') or not ret:
                break

            frame_index += 1

        cap.release()
        cv2.destroyAllWindows()
