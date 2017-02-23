import cv2
import numpy as np
from processing_utils import Minv, evaluate_fit_at_point, get_center_offset

def add_radius_to_image(image, curverad):
    curverad_str = 'Radius of curvature: ' + str(int(curverad)) + 'm'
    cv2.putText(image, curverad_str, (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
    return image

def get_offset_str(offset):
    rounded_offset_str = str(abs(round(offset, 2)))
    if offset > 0:
        return 'Vehicle is ' + rounded_offset_str + 'm right of center'
    if offset < 0:
        return 'Vehicle is ' + rounded_offset_str + 'm left of center'
    return 'Vehicle is centered'

def add_center_offset_to_image(image, left_fit, right_fit):
    center_offset = get_center_offset(left_fit, right_fit)
    offset_str = get_offset_str(center_offset)
    cv2.putText(image, offset_str, (20, 150), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
    return image

# Code source: Advanced Lane Finding lesson - 33. Finding the Lines
def get_plotting_values_for_fits(warped, left_fit, right_fit):
    ploty = np.linspace(0, warped.shape[0]-1, warped.shape[0] )
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    return ploty, left_fitx, right_fitx

# Code source: Advanced Lane Finding Lesson - 36. Tips and Tricks for the Project
def add_lines_to_image(image, warped, left_fit, right_fit):
    ploty, left_fitx, right_fitx = get_plotting_values_for_fits(warped, left_fit, right_fit)
    # Create an image to draw the lines on
    warp_zero = np.zeros_like(warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))
    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))
    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))
    cv2.polylines(color_warp, np.int_([pts_left]), False, (255, 0, 0), 12)
    cv2.polylines(color_warp, np.int_([pts_right]), False, (255, 0, 0), 12)
    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, Minv, (image.shape[1], image.shape[0])) 
    # Combine the result with the original image
    result = cv2.addWeighted(image, 1, newwarp, 0.3, 0)
    return result
