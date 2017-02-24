import glob
import cv2
import numpy as np
import matplotlib.pyplot as plt
from process_image import correct_distortion, thresholded_image, warp_perspective
from find_lane_lines import find_lines_via_histogram_peaks
from plotting_utils import get_plotting_values_for_fits

img_bgr = cv2.imread('../test_images/test4.jpg')
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
img_corrected = correct_distortion(img_rgb)
img_thresh = thresholded_image(img_corrected)
img_corrected_proj = warp_perspective(img_corrected)
img_thresh_proj = warp_perspective(img_thresh)

left_fit, right_fit, left_curverad, right_curverad \
    = find_lines_via_histogram_peaks(img_thresh_proj)
ploty, left_fitx, right_fitx = get_plotting_values_for_fits(img_thresh_proj, left_fit, right_fit)
# Create an image to draw the lines on
warp_zero = np.zeros_like(img_thresh_proj).astype(np.uint8)
img_thresh_proj_color = 255 * np.dstack((img_thresh_proj, img_thresh_proj, img_thresh_proj))
color_warp = np.dstack((warp_zero, warp_zero, warp_zero))
# Recast the x and y points into usable format for cv2.fillPoly()
pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
pts = np.hstack((pts_left, pts_right))
cv2.polylines(color_warp, np.int_([pts_left]), False, (255, 0, 0), 12)
cv2.polylines(color_warp, np.int_([pts_right]), False, (255, 0, 0), 12)
img_thresh_proj_color[color_warp[:,:,0] > 0] = 0
polyplot = cv2.addWeighted(img_thresh_proj_color, 1, color_warp, 1, 1)
fig = plt.figure()
plt.title('Polynomial fit overlaid onto warped thresholded image')
plt.imshow(polyplot)
plt.show()

'''
fig = plt.figure()
ax1 = fig.add_subplot(121)
ax1.imshow(img_rgb)
ax1.set_title('Raw image')
ax2 = fig.add_subplot(122)
ax2.imshow(img_corrected)
ax2.set_title('Corrected for distortion')
plt.show()

fig = plt.figure()
plt.imshow(img_thresh, cmap='gray')
plt.title('Thresholded corrected image')
plt.show()

fig = plt.figure()
ax1 = fig.add_subplot(121)
ax1.imshow(img_corrected_proj)
ax1.set_title('Projected corrected image')
ax2 = fig.add_subplot(122)
ax2.imshow(img_thresh_proj, cmap='gray')
ax2.set_title('Projected thresholded corrected image')
plt.show()
'''

'''
fig = plt.figure()
images = glob.glob('../test_images/test*.jpg')
plt.title('Thresholding characteristics of test images')
plt.axis('off')
for i in range(0, 6):
    img_bgr = cv2.imread(images[i])
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_out = warp_perspective(thresholded_image(correct_distortion(img_rgb)))
    img_out[:,:250] = 0
    ax = fig.add_subplot(230+(i+1))
    ax.imshow(img_out, cmap='gray')
    ax.set_title('test' + str(i+1) + '.jpg')
    ax.axis('off')
plt.show()
'''
