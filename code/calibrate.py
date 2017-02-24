import glob
import sys
import getopt
import pickle
import cv2
import matplotlib.pyplot as plt
import numpy as np

nx = 9
ny = 6
nz = 3

imgw = 720
imgh = 1280
imgshape = (imgw, imgh)

CAL_DATA_PICKLE = '../calibration_data.p'

# Code source: Advanced Lane Finding Lesson - 10. Calibrating Your Camera video
# Prepare object points, like (1, 0, 0), (2, 0, 0), ..., (8, 5, 0)
def generate_objpoints():
    objp = np.zeros((nx * ny, nz), np.float32)
    objp[:, :2] = np.mgrid[0:nx, 0:ny].T.reshape(-1, 2) # x, y coordinates
    return objp

def get_corners_for_image(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)
    return ret, corners

def find_objpoints_and_imgpoints_for_camera(images):
    """Given calibration images, determine image points and object points if possible."""
    objpoints = []
    imgpoints = []
    for image in images:
        ret, corners = get_corners_for_image(image)
        if ret:
            objpoints.append(generate_objpoints())
            imgpoints.append(corners)
    return objpoints, imgpoints

def get_images():
    """Load calibration image data."""
    image_paths = glob.glob('../camera_cal/calibration*.jpg')
    images = []
    for image_path in image_paths:
        image = cv2.imread(image_path)
        image = image[:imgw, :imgh, :] # some images are 1px too large
        images.append(image)
    return images

def generate_and_save_calibration_data():
    images = get_images()
    objpoints, imgpoints = find_objpoints_and_imgpoints_for_camera(images)
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, imgshape, None, None)
    pickle.dump((ret, mtx, dist, rvecs, tvecs), open(CAL_DATA_PICKLE, 'wb'))

def get_image_with_shown_corners(img):
    ret, corners = get_corners_for_image(img)
    return cv2.drawChessboardCorners(img, (nx, ny), corners, ret)

def load_calibration_data():
    ret, mtx, dist, rvecs, tvecs = pickle.load(open(CAL_DATA_PICKLE, 'rb'))
    return ret, mtx, dist, rvecs, tvecs

if __name__ == '__main__':
    argv = sys.argv[1:] # get args other than filename
    try:
        opts, args = getopt.getopt(argv, 'cg')
    except getopt.GetoptError:
        print('Usage: python calibrate.py [-cg]')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-c':
            generate_and_save_calibration_data()
        if opt == '-g':
            ret, mtx, dist, rvecs, tvecs = load_calibration_data()
            sample_img = cv2.imread('../camera_cal/calibration2.jpg')
            sample_dst = cv2.undistort(sample_img, mtx, dist, None, mtx)
            fig, ax_list = plt.subplots(1, 2)
            ax_list[0].imshow(get_image_with_shown_corners(sample_img))
            ax_list[0].set_title('Original (detected corners shown)')
            ax_list[1].imshow(sample_dst)
            ax_list[1].set_title('Undistorted')
            plt.show()
