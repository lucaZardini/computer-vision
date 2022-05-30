# computer-vision

The objective of this assignment was to detect and track meaningful features from a video.

The structure of the repository is the following:
* The _keypoints_ folder contains the keypoints computed offline by the various detector used;
* The _src_ folder contains the code;
* The _video_ folder contains the video used for testing.


As already anticipated, the feature detection and tracking can be computed in two ways:
* Online detection
* Offline detection

The motion tracking algorithm instead is always computed online.


In this project, the feature detector developed are:
* SIFT
* Good features to track
* ORB
* FAST
* Brief

Two motion tracking algorithm has been developed:
* Lucas Kanade Optical flow
* Kalman filter with hungarian algorithm
