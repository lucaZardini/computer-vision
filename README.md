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

## Usage

To run the project, start the main.py script. By default, it runs the feature detection algorithms implemented 
with the Lucas Kanade Optical Flow tracking algorithm. To select only one method or to run the Kalman Filter tracker,
open the manager.py script and change the line 20 according to your desire.
The following is an example to select the ORB detector with the Kalman Filter tracker
```python
if tracker.detector.name() == FeatureDetectorAlgorithm.ORB.value and tracker.tracking.name() == FeatureTrackingAlgorithm.KALMAN_FILTER.value:
```
