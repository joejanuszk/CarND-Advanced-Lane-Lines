import glob
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from process_image import correct_distortion, warp_perspective, thresholded_image, hls_select_l, hls_select_s, hls_select_h
from find_lane_lines import find_lines_via_histogram_peaks
from plotting_utils import add_lines_to_image

fig = plt.figure()
images = glob.glob('../test_images/test*.jpg')
for i in range(0, 6):
    img = mpimg.imread(images[i])
    thresh_img = thresholded_image(correct_distortion(img))
    #thresh_img = img
    #thresh_img = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    #thresh_img = thresh_img[:,:,0]
    #mean = np.mean(thresh_img[:,:,1])
    #thresh_img[:,:,1] = 127
    #mean = np.mean(thresh_img)
    #thresh_img = cv2.cvtColor(thresh_img, cv2.COLOR_HLS2RGB)
    #thresh_img[(thresh_img < mean)] = mean
    binary_warped = warp_perspective(thresh_img)
    binary_warped[:,:250] = 0
    ax = fig.add_subplot(230+(i+1))
    ax.imshow(binary_warped, cmap='gray')
    #ax.imshow(binary_warped)
plt.show()
