import cv2
import numpy as np
import matplotlib.pyplot as plt
from moviepy.editor import VideoFileClip
from process_image import correct_distortion, warp_perspective, thresholded_image

#clip = VideoFileClip('../project_video.mp4')
#clip = VideoFileClip('../test_video_2.mp4')
clip = VideoFileClip('../project_video_output_test_2.mp4')
#processed_clip = clip.subclip(18, 28)
#processed_clip.write_videofile('../test_video_2.mp4', fps=24, codec='mpeg4')
params = cv2.SimpleBlobDetector_Params()
params.filterByInertia = True
params.filterByArea = True
params.minArea = 200
params.minInertiaRatio = 0.5
params.maxInertiaRatio = 1
detector = cv2.SimpleBlobDetector_create(params)

count = 0
frameno = 0
for frame in clip.iter_frames():
    if count % 4 == 0:
        frameno += 1
        #thresh_img = thresholded_image(correct_distortion(frame))
        #binary_warped = warp_perspective(thresh_img)
        binary_warped = warp_perspective(correct_distortion(frame))
        #cv2.imwrite('../processing_test/sample' + str(frameno) + '.jpg', binary_warped * 255)
        cv2.imwrite('../processing_test/sample' + str(frameno) + '.jpg', cv2.cvtColor(binary_warped, cv2.COLOR_RGB2BGR))
    count += 1
print(count)
