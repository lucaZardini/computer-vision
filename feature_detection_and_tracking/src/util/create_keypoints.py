import cv2

from feature_detector.feature_detector import SiftDetector, GoodFeaturesToTrackDetector, ORBDetector, FASTDetector, StarDetector

import numpy as np

if __name__ == "__main__":
    video_path = "../../video/video.mp4"
    sift = SiftDetector()
    gff = GoodFeaturesToTrackDetector()
    orb = ORBDetector()
    fast = FASTDetector()
    brief = StarDetector()

    cap = cv2.VideoCapture(video_path)

    sift_keypoints = []
    gff_keypoints = []
    orb_keypoints = []
    fast_keypoints = []
    brief_keypoints = []

    while cap.isOpened():

        ret, frame = cap.read()

        if not ret:
            break

        sift.image = frame
        gff.image = frame
        orb.image = frame
        fast.image = frame
        brief.image = frame

        sift_keypoint = sift.detect()
        sift_keypoint = np.array([[k.pt] for k in sift_keypoint], dtype=np.float32)
        sift_keypoints.append(sift_keypoint)
        gff_keypoint = gff.detect()  # OK
        gff_keypoints.append(gff_keypoint)
        orb_keypoint = orb.detect()
        orb_keypoint = np.array([[k.pt] for k in orb_keypoint], dtype=np.float32)
        orb_keypoints.append(orb_keypoint)
        fast_keypoint = fast.detect()
        i = 0
        kp = []
        for k in fast_keypoint:
            if i < 20000:
                kp.append([k.pt])
            i += 1
        fast_keypoint = np.array(kp, dtype=np.float32)
        fast_keypoints.append(fast_keypoint)
        brief_keypoint = brief.detect()
        brief_keypoint = np.array([[k.pt] for k in brief_keypoint], dtype=np.float32)
        brief_keypoints.append(brief_keypoint)

    np.save('../../keypoints/sift.npy', sift_keypoints, allow_pickle=True)
    np.save('../../keypoints/gftt.npy', gff_keypoints, allow_pickle=True)
    np.save('../../keypoints/orb.npy', orb_keypoints, allow_pickle=True)
    np.save('../../keypoints/fast.npy', fast_keypoints, allow_pickle=True)
    np.save('../../keypoints/star.npy', brief_keypoints, allow_pickle=True)
