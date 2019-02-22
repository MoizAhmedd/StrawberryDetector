import cv2
from matplotlib import pyplot as plt
import numpy as np
from math import cos,sin
from __future__ import division

def find_strawberry(image):
    """
    This function allows for detection of images of strawberries by doing the following
    -Takes an image an converts to RGB color scheme
    -Scales the image to correct size
    -Cleans Image by blurring and converting to HSV
    -Defines filters
    -
    """
    #This converts color scheme
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    max_dimension = max(image.shape)

    # Max size is 700px
    scale = 700/max_dimension

    # Squares image
    image = cv2.resize(image, None, fx=scale, fy=scale)

    # Blur image, remove noise
    image_blur = cv2.GaussianBlur(image, (7, 7), 0)
    # Color scheme has to be converted again
    # Converts to HSV, separates intensity from color information
    image_blur_hsv = cv2.cvtColor(image_blur,cv2.COLOR_RGB2HSV)

    # Filter by color
    min_red = np.array([0,100,80])
    max_red = np.array([10,256,256])

    





