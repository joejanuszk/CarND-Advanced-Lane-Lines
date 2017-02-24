# Advanced Lane Finding

My work for Udacity's Self-Driving Car Nanodegree, Project 4

## Camera Calibration

My code for calibrating camera images is located in [code/calibrate.py](code/calibrate.py). This script takes the following steps:

1. In `get_images()`, the calibration images from `camera_cal/` are loaded via the `glob` API. I noticed that some of these images were 1 pixel too large, so I standardize their dimensions to 1280x720.
2. In `find_objpoints_and_imgpoints_for_camera()`, image points are determined for each image via `cv2.findChessboardCorners()` if possible. Note that some of the calibration images are cropped with chessboard corners missing, so these images are excluded from calibration.
3. `cv2.calibrateCamera()` uses the image points and corresponding object points to generate calibration data.
4. This data is pickled and saved to a file for future use without having to recalculate the calibration data each time.

We can verify the correctness of this calibration by comparing a raw chessboard photo to the distortion-corrected version:

![Distortion Correction](output_images/camera_cal_checkerboard.png "Distortion Correction")

## Image Pipeline

Functions used for various image processing and manipulation tasks can be found in [code/process_image.py](code/process_image.py).

### Distortion Correction

In `process_image.py:correct_distortion()`, images are distortion-corrected using `cv2.undistort()` and pre-computed calibration data. We can see distortion correction applied to an image of the road below. Note the slight difference between the two images:

![Distortion Correction](output_images/raw_corrected.png "Distortion Correction")
