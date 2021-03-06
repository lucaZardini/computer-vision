from typing import List

from feature_detector.feature_detector import FeatureDetectorAlgorithm
from feature_tracking.feature_tracking import FeatureTrackingAlgorithm

from tracker.tracker import Tracker, TrackerBuilder


class TrackManager:

    @staticmethod
    def track_video(video_path: str, online: bool = True, save_video: bool = False) -> None:
        # create the feature detectors and the feature trackers
        trackers: List[Tracker] = []
        for detect_algorithm in FeatureDetectorAlgorithm:
            for track_algorithm in FeatureTrackingAlgorithm:
                trackers.append(TrackerBuilder.build(detector_algorithm=detect_algorithm, tracker_algorithm=track_algorithm, video=video_path, online=online, save_video=save_video))

        for tracker in trackers:
            if tracker.tracking.name() == FeatureTrackingAlgorithm.LK.value:
                tracker.track()
