import numpy as np
from moviepy.editor import VideoFileClip
from process_image import correct_distortion, warp_perspective, thresholded_image
from find_lane_lines import *
from plotting_utils import add_lines_to_image, add_radius_to_image, add_center_offset_to_image
from collections import deque

min_radius = 200 # 0.2km
tries_before_refit = 4
lane_pixel_gap = 930 - 370
gap_margin = 120
max_gap = lane_pixel_gap + gap_margin
min_gap = lane_pixel_gap - gap_margin

def radius_exceeds_bounds(radius):
    """Check if a line's radius of curvature seems too small."""
    return radius < min_radius

def get_x_for_fit(y, fit):
    """Compute x value for a fit given y."""
    return fit[0] * (y ** 2) + fit[1] * y + fit[2]

def curves_bad_distance_apart(left_fit, right_fit):
    """Check if the curves are too far apart at any point."""
    for y in range(0, 720, 72):
        leftx = get_x_for_fit(y, left_fit)
        rightx = get_x_for_fit(y, right_fit)
        gap = abs(rightx - leftx)
        if gap > max_gap or gap < min_gap:
            return True
    return False

def misses_exceed_tries(misses):
    """Check if there have been too many misses since the last good match."""
    return misses['left'] > tries_before_refit or misses['right'] > tries_before_refit

def add_item_to_capped_queue(item, queue):
    """Add an item to a queue with a capped length."""
    if len(queue) > 5:
        queue.popleft()
    queue.append(item)

def get_average_fit(queue):
    fits = tuple(queue)
    stacked_fits = np.dstack(fits)[0]
    return np.mean(stacked_fits, axis=1)

def get_mean_fits(left_fits, right_fits):
    """Average lane line curves against each other for drawing."""
    left_fit_mean = get_average_fit(left_fits)
    right_fit_mean = get_average_fit(right_fits)
    pow_2 = [left_fit_mean[0], right_fit_mean[0]]
    pow_1 = [left_fit_mean[1], right_fit_mean[1]]
    pow_2_mean = np.mean(pow_2)
    pow_1_mean = np.mean(pow_1)
    left_fit_mean = np.array([pow_2_mean, pow_1_mean, left_fit_mean[2]])
    right_fit_mean = np.array([pow_2_mean, pow_1_mean, right_fit_mean[2]])
    return left_fit_mean, right_fit_mean

def get_mean_curverad(left_radii, right_radii):
    """Average left and right curvatures for drawing."""
    return np.mean(list(left_radii) + list(right_radii))

# maintain state across function calls
fits = {}
misses = {}
left_fits = deque()
right_fits = deque()
left_radii = deque()
right_radii = deque()

def process_frame(frame):
    thresh_img = thresholded_image(correct_distortion(frame))
    binary_warped = warp_perspective(thresh_img)
    # clip the left wall, since it's difficult to filter the shadows otherwise
    binary_warped[:,:250] = 0
    # The first frame of the video must be processed via the histogram peaks method.
    if 'left' not in fits or 'right' not in fits:
        left_fit, right_fit, left_curverad, right_curverad \
            = find_lines_via_histogram_peaks(binary_warped)
        fits['left'] = left_fit
        fits['right'] = right_fit
        misses['left'] = 0
        misses['right'] = 0
        add_item_to_capped_queue(left_curverad, left_radii)
        add_item_to_capped_queue(right_curverad, right_radii)
    # Otherwise, try to find fits via the polynomial fit method using previous fits.
    else:
        left_fit = fits['left']
        right_fit = fits['right']
        try:
            left_fit, right_fit, left_curverad, right_curverad \
                 = find_lines_via_polynomial_fit(binary_warped, left_fit, right_fit)
            # If the curves are too far apart at any point, treat it as a miss.
            if curves_bad_distance_apart(left_fit, right_fit):
                print("curves bad distance")
                misses['left'] += 1
                misses['right'] += 1
                left_fit = fits['left']
                right_fit = fits['right']
            else:
                print(left_curverad, right_curverad)
                # If a radius is a bad value, treat it as a miss.
                if radius_exceeds_bounds(left_curverad):
                    print("left curve exceeds bounds")
                    misses['left'] += 1
                    left_fit = fits['left']
                else:
                    misses['left'] = 0
                    add_item_to_capped_queue(left_curverad, left_radii)
                if radius_exceeds_bounds(right_curverad):
                    print("right curve exceeds bounds")
                    misses['right'] += 1
                    right_fit = fits['right']
                else:
                    misses['right'] = 0
                    add_item_to_capped_queue(right_curverad, right_radii)
        # The polynomial fit fails in some circumstances, particularly if
        # line points are too sparse. Treat it as a miss.
        except TypeError:
            print("polynomial fit failed")
            misses['left'] += 1
            misses['right'] += 1
            left_fit = fits['left']
            right_fit = fits['right']
        # After too many consecutive failures, 
        if misses_exceed_tries(misses):
            print("misses exceed tries; recomputing via histogram")
            left_fit, right_fit, left_curverad, right_curverad \
                = find_lines_via_histogram_peaks(binary_warped)
            fits['left'] = left_fit
            fits['right'] = right_fit
            misses['left'] = 0
            misses['right'] = 0
            add_item_to_capped_queue(left_curverad, left_radii)
            add_item_to_capped_queue(right_curverad, right_radii)
    # Cache this round's results for the next round
    add_item_to_capped_queue(left_fit, left_fits)
    add_item_to_capped_queue(right_fit, right_fits)
    left_fit_mean = get_average_fit(left_fits)
    # Apply the polynomial fits, radius of curvative, and
    # center offset results to the frame.
    right_fit_mean = get_average_fit(right_fits)
    curverad = get_mean_curverad(left_radii, right_radii)
    result = add_lines_to_image(frame, binary_warped, left_fit_mean, right_fit_mean)
    result = add_radius_to_image(result, curverad)
    result = add_center_offset_to_image(result, left_fit_mean, right_fit_mean)
    return result

clip = VideoFileClip('../project_video.mp4')
processed_clip = clip.fl_image(process_frame)
processed_clip.write_videofile('../project_video_output.mp4', fps=24, codec='mpeg4')
