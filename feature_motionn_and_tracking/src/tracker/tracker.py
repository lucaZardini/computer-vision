from feature_detector.feature_detector import FeatureDetector
from feature_tracking.feature_tracking import FeatureTracking


class Tracker:
    def __init__(self, detector: FeatureDetector, tracking: FeatureTracking, video):
        self.detector = detector
        self.tracking = tracking
        self.video = video
        self._result = None

    @property
    def result(self):
        if self._result:
            return self._result
        else:
            return self.track()

    def track(self):
        pass
