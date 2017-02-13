import cv2
import numpy as np

src = np.float32(
    [[700, 450],
     [1080, 720],
     [200, 720],
     [580, 450]])
dst = np.float32(
    [[890+80, 0],
     [890, 720],
     [390, 720],
     [390-80, 0]])

M = cv2.getPerspectiveTransform(src, dst)
Minv = cv2.getPerspectiveTransform(dst, src)
