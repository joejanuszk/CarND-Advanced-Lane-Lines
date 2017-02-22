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
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

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

def mag_thresh_s(img, sobel_kernel=3, thresh=(0, 255)):
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    img_s = hls[:,:,2]
    sobelx = cv2.Sobel(img_s, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(img_s, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    mag = ((sobelx ** 2) + (sobely ** 2)) ** 0.5
    scaled_mag = scale_magnitude(mag)
    return apply_threshold(scaled_mag, thresh)

def mag_thresh_l(img, sobel_kernel=3, thresh=(0, 255)):
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    img_l = hls[:,:,1]
    sobelx = cv2.Sobel(img_l, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(img_l, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    mag = ((sobelx ** 2) + (sobely ** 2)) ** 0.5
    scaled_mag = scale_magnitude(mag)
    return apply_threshold(scaled_mag, thresh)

def hls_select_h(img, thresh=(0, 255)):
    # 1) Convert to HLS color space
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    # 2) Apply a threshold to the S channel
    img_h = hls[:,:,0]
    binary_output = np.zeros_like(img_h)
    binary_output[(img_h > thresh[0]) & (img_h <= thresh[1])] = 1
    # 3) Return a binary image of threshold result
    #binary_output = np.copy(img) # placeholder line
    return binary_output

def hls_select_l(img, thresh=(0, 255)):
    # 1) Convert to HLS color space
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    # 2) Apply a threshold to the S channel
    img_l = hls[:,:,1]
    binary_output = np.zeros_like(img_l)
    binary_output[(img_l > thresh[0]) & (img_l <= thresh[1])] = 1
    # 3) Return a binary image of threshold result
    #binary_output = np.copy(img) # placeholder line
    return binary_output

def hls_select_s(img, thresh=(0, 255)):
    # 1) Convert to HLS color space
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    # 2) Apply a threshold to the S channel
    s = hls[:,:,2]
    binary_output = np.zeros_like(s)
    binary_output[(s > thresh[0]) & (s <= thresh[1])] = 1
    # 3) Return a binary image of threshold result
    #binary_output = np.copy(img) # placeholder line
    return binary_output

def dir_thresh(img, sobel_kernel=3, thresh=(0, np.pi/2)):
    gray = convert_to_grayscale(img)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    abs_sobelx = np.absolute(sobelx)
    abs_sobely = np.absolute(sobely)
    dir_grad = np.arctan(abs_sobely, abs_sobelx)
    return apply_threshold(dir_grad, thresh)

def thresholded_image(img):
    gradx = abs_sobel_thresh(img, orient='x', thresh=(20, 255))
    grady = abs_sobel_thresh(img, orient='y', thresh=(40, 255))
    mag_binary = mag_thresh(img, thresh=(45, 255), sobel_kernel=5)
    mag_binary_s = mag_thresh(img, thresh=(30, 255), sobel_kernel=5)
    hls_h_binary = hls_select_h(img, thresh=(10, 105))
    hls_s_binary = hls_select_s(img, thresh=(90, 255))
    hls_l_low_binary = hls_select_l(img, thresh=(0, 80))
    hls_s_low_binary = hls_select_s(img, thresh=(0, 90))
    combined = np.zeros_like(mag_binary)
    combined[((gradx == 1) | (grady == 1) | (mag_binary == 1) | (mag_binary_s == 1)) | ((hls_s_binary == 1) & (hls_h_binary == 1))] = 1
    combined[(hls_l_low_binary == 1) & (hls_s_low_binary == 1)] = 0
    return combined
