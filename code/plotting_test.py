import cv2
import numpy as np
import matplotlib.pyplot as plt
from process_image import correct_distortion, warp_perspective, thresholded_image
from find_lane_lines import find_lines_via_histogram_peaks
from plotting_utils import add_lines_to_image, add_radius_to_image, add_center_offset_to_image

img_bgr = cv2.imread('../test_images/test4.jpg')
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
img_thresh = thresholded_image(correct_distortion(img_rgb))
img_binary_warped = warp_perspective(img_thresh)
left_fit, right_fit, left_curverad, right_curverad \
    = find_lines_via_histogram_peaks(img_binary_warped)
curverad = np.mean([left_curverad, right_curverad])
img_result = add_lines_to_image(img_rgb,
                                img_binary_warped,
                                left_fit,
                                right_fit)
img_result = add_radius_to_image(img_result, curverad)
img_result = add_center_offset_to_image(img_result, left_fit, right_fit)

plt.imshow(img_result)
plt.show()
