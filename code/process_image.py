import cv2
import numpy as np
from calibrate import load_calibration_data
from processing_utils import M

ret, mtx, dist, rvecs, tvecs = load_calibration_data()

def correct_distortion(img):
    return cv2.undistort(img, mtx, dist, None, mtx)

def warp_perspective(img):
    return cv2.warpPerspective(img, M, (1280, 720), flags=cv2.INTER_LINEAR)

def convert_to_grayscale(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def scale_magnitude(values):
    return np.uint8(255 * values / np.max(values))

def apply_threshold(values, thresh):
    binary_output = np.zeros_like(values)
    binary_output[(values >= thresh[0]) & (values <= thresh[1])] = 1
    return binary_output

def abs_sobel_thresh(img, orient='x', thresh=(0, 255)):
    gray = convert_to_grayscale(img)
    if orient == 'x':
        sobel = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
    else:
        sobel = cv2.Sobel(gray, cv2.CV_64F, 0, 1)
    abs_sobel = np.absolute(sobel)
    scaled_sobel = scale_magnitude(abs_sobel)
    return apply_threshold(scaled_sobel, thresh)

def mag_thresh(img, sobel_kernel=3, thresh=(0, 255)):
    gray = convert_to_grayscale(img)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    mag = ((sobelx ** 2) + (sobely ** 2)) ** 0.5
    scaled_mag = scale_magnitude(mag)
    return apply_threshold(scaled_mag, thresh)

def dir_thresh(img, sobel_kernel=3, thresh=(0, np.pi/2)):
    gray = convert_to_grayscale(img)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    abs_sobelx = np.absolute(sobelx)
    abs_sobely = np.absolute(sobely)
    dir_grad = np.arctan(abs_sobely, abs_sobelx)
    return apply_threshold(dir_grad, thresh)

def thresholded_image(img):
    gradx = abs_sobel_thresh(img, orient='x', thresh=(20, 150))
    grady = abs_sobel_thresh(img, orient='y', thresh=(20, 150))
    mag_binary = mag_thresh(img, thresh=(30, 100))
    dir_binary = dir_thresh(img, thresh=(0.7, 1.3))
    combined = np.zeros_like(dir_binary)
    combined[((gradx == 1) & (grady == 1)) | ((mag_binary == 1) & (dir_binary == 1))] = 1
    return combined
