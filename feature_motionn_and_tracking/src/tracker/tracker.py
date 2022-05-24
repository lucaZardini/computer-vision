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

            if detector_algorithm == FeatureDetectorAlgorithm.GOOD_FEATURES_TO_TRACK:
                return TrackGFFwithLK(detector=detector, tracking=tracker, video=video, online=online)

            elif detector_algorithm == FeatureDetectorAlgorithm.SIFT:
                return SIFTwithLK(detector=detector, tracking=tracker, video=video, online=online)

            elif detector_algorithm == FeatureDetectorAlgorithm.ORB:
                return ORBwithLK(detector=detector, tracking=tracker, video=video, online=online)

            elif detector_algorithm == FeatureDetectorAlgorithm.FAST:
                return FASTwithLK(detector=detector, tracking=tracker, video=video, online=online)

            elif detector_algorithm == FeatureDetectorAlgorithm.BRIEF:
                return FASTwithLK(detector=detector, tracking=tracker, video=video, online=online)

            else:
                pass
        else:
            pass


class TrackGFFwithLK(Tracker):

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
                    corners = self.detector.detect()
                else:
                    if not self.file_with_keypoints_is_present:
                        raise FileExistsError("The file does not exists")
                    corners = self.get_keypoints(frame_index)
                self.tracking.initialize(frame, corners)
            else:
                corners, status, err = self.tracking.track(frame)

            frame_copy = frame.copy()
            int_corners = corners.astype(int)
            for i, corner in enumerate(int_corners):
                x, y = corner.ravel()
                color = np.float64([3 * i, 2 * i, 255 - (i * 4)])
                cv2.circle(frame_copy, (x, y), 20, color, thickness=5)

            cv2.imshow('GFF', frame_copy)

            if cv2.waitKey(1) == ord('q') or not ret:
                break

            frame_index += 1

        cap.release()
        cv2.destroyAllWindows()


class SIFTwithLK(Tracker):

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

            cv2.imshow('GFF', frame_copy)

            if cv2.waitKey(1) == ord('q') or not ret:
                break

            frame_index += 1

        cap.release()
        cv2.destroyAllWindows()


class ORBwithLK(Tracker):

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
                    features, descriptor = self.detector.detect()
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

            cv2.imshow('ORB and LK', frame_copy)

            if cv2.waitKey(1) == ord('q') or not ret:
                break

            frame_index += 1

        cap.release()
        cv2.destroyAllWindows()


class FASTwithLK(Tracker):

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

            cv2.imshow('ORB and LK', frame_copy)

            if cv2.waitKey(1) == ord('q') or not ret:
                break

            frame_index += 1

        cap.release()
        cv2.destroyAllWindows()


class BRIEFwithLK(Tracker):

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

            cv2.imshow('ORB and LK', frame_copy)

            if cv2.waitKey(1) == ord('q') or not ret:
                break

            frame_index += 1

        cap.release()
        cv2.destroyAllWindows()


class SIFTwithKF(Tracker):

    def track(self):
        cap = cv2.VideoCapture(self.video)
        frame_index = 0
        colours = []
        for i in range(0, 100000):
            col_np_array = np.random.choice(range(256), size=3)
            col_list = []
            for j in range(0, col_np_array.size):
                col_list.append(col_np_array[j].item())
            colours.append(tuple(col_list))

        while cap.isOpened():
            ret, frame = cap.read()

            if not ret:
                break

            # if frame_index % 30 == 0:
            self.detector.image = frame
            features = self.detector.detect()
            features = np.array([[k.pt] for k in features], dtype=np.float32)
            int_corners = features.astype(int)

            bboxes = []
            for i, corner in enumerate(int_corners):
                x, y = corner.ravel()
                top = y - 10
                left = x - 10
                bottom = y + 10
                right = x + 10
                bbox = (top, left, bottom, right)
                bboxes.append(bbox)
            self.tracking.initialize(frame, bboxes)
            # for i, corner in enumerate(int_corners):
            #     x, y = corner.ravel()
            #     top = y - 5
            #     left = x - 5
            #     bottom = y + 5
            #     right = x + 5
            #     cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            # cv2.imshow("Track", frame)
            # cv2.waitKey(40)

            # else:
            tracker_id_list, bounding_boxes, centres, scores, class_labels = self.tracking.track(frame)
            # Draw bounding boxes, trajectories and print ids of the tracked pedestrians
            track = 0
            for track_id, bbox, track_centres in zip(tracker_id_list, bounding_boxes, centres):
                top_f = bbox[0]
                left_f = bbox[1]
                bottom_f = bbox[2]
                right_f = bbox[3]

                top = int(bbox[0])
                left = int(bbox[1])
                bottom = int(bbox[2])
                right = int(bbox[3])

                # Draw the bounding box
                cv2.rectangle(frame, (left, top), (right, bottom), colours[track], 2)
                # Draw the trajectory using the centres of the previous bounding boxes
                for point in track_centres[-30:]:
                    cv2.circle(frame, (int(point[0]), int(point[1])), 2, colours[track], -1)
                    # Draw the box used for id visualisation
                cv2.rectangle(frame, (left, top - 20), (right, top), (0, 0, 255), cv2.FILLED)
                # Print the id
                font = cv2.FONT_HERSHEY_DUPLEX
                cv2.putText(frame, str(track_id), (left, top - 5), font, 0.5, (255, 255, 255), 1)
                track += 1

            # Display the result
            cv2.imshow("Track", frame)
            cv2.waitKey(40)

            frame_index += 1

        cap.release()
        cv2.destroyAllWindows()
