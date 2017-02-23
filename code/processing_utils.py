import cv2
import numpy as np

src = np.float32(
    [[700, 455],
     [1080, 720],
     [200, 720],
     [580, 455]])
dst = np.float32(
    [[890+80, 0],
     [890, 720],
     [390, 720],
     [390-80, 0]])

ym_per_pix = 30/720 # meters per pixel in y dimension
xm_per_pix = 3.7/(930-370) # meters per pixel in x dimension

M = cv2.getPerspectiveTransform(src, dst)
Minv = cv2.getPerspectiveTransform(dst, src)

def evaluate_fit_at_point(fit, pt):
    return fit[0] * (pt ** 2) + fit[1] * pt + fit[2]

def get_center_offset(left_fit, right_fit):
    """Given fits for the left and right lane lines, determine the offset from center"""
    left = evaluate_fit_at_point(left_fit, 719)
    right = evaluate_fit_at_point(right_fit, 719)
    img_center = 1280 / 2.
    measured_center = np.mean([left, right])
    offset = img_center - measured_center
    offset_m = offset * xm_per_pix
    return offset_m
