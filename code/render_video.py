from moviepy.editor import VideoFileClip
from process_image import correct_distortion, warp_perspective, thresholded_image
from find_lane_lines import *
from plotting_utils import add_lines_to_image

# maintain state across function calls
fits = {}

def process_frame(frame):
    thresh_img = thresholded_image(correct_distortion(frame))
    binary_warped = warp_perspective(thresh_img)
    if 'left_fit' not in fits or 'right_fit' not in fits:
        left_fit, right_fit = find_lines_via_histogram_peaks(binary_warped)
        fits['left_fit'] = left_fit
        fits['right_fit'] = right_fit
    else:
        left_fit = fits['left_fit']
        right_fit = fits['right_fit']
        left_fit, right_fit = find_lines_via_polynomial_fit(binary_warped, left_fit, right_fit)
    result = add_lines_to_image(frame, binary_warped, left_fit, right_fit)
    return result

clip = VideoFileClip('../project_video.mp4')
processed_clip = clip.fl_image(process_frame)
processed_clip.write_videofile('../project_video_output.mp4', fps=24, codec='mpeg4')
