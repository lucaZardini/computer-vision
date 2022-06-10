import datetime

import cv2

from feature_detector.feature_detector import FeatureDetector, FeatureDetectorAlgorithm, FeatureDetectorBuilder


def compute_time(video: str, detector: FeatureDetector):
    cap = cv2.VideoCapture(video)

    while cap.isOpened():

        ret, frame = cap.read()

        if not ret:
            break

        detector.image = frame

        start_time = datetime.datetime.now()
        _ = detector.detect()
        end_time = datetime.datetime.now()
        return end_time - start_time


if __name__ == '__main__':
    times = {}
    for algorithm in FeatureDetectorAlgorithm:
        detector = FeatureDetectorBuilder.build(algorithm)
        detector_time = compute_time("../../video/video.mp4", detector)
        times[detector.name()] = detector_time.total_seconds()
    print(times)
