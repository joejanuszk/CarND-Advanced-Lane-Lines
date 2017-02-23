import glob
import cv2
import matplotlib.pyplot as plt
from process_image import correct_distortion, thresholded_image, warp_perspective

'''
img_bgr = cv2.imread('../test_images/test4.jpg')
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
img_corrected = correct_distortion(img_rgb)
img_thresh = thresholded_image(img_corrected)
img_corrected_proj = warp_perspective(img_corrected)
img_thresh_proj = warp_perspective(img_thresh)

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
