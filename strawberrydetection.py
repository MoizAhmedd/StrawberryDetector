from __future__ import division
import cv2
from matplotlib import pyplot as plt
import numpy as np
from math import cos,sin


#Colors
green = (0,255,0)

#Helpers
def show(image):
    plt.figure(figsize=(10,10))
    plt.imshow(image,interpolation='nearest')

def overlay_mask(mask,image):
    #Make mask rgb
    rgb_mask = cv2.cvtColor(mask,cv2.COLOR_GRAY2RGB)

    img = cv2.addWeighted(rgb_mask,0.5,image,0.5,0)
    return img

def find_biggest_contour(image):
    #Copy image
    image = image.copy()

    contours,hierarchy = cv2.findContours(image,cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    contour_sizes = [(cv2.contourArea(contour),contour) for contour in contours]
    biggest_contour = max(contour_sizes,key= lambda x:x[0])[1]

    mask = np.zeros(image.shape,np.uint8)
    cv2.drawContours(mask,[biggest_contour],-1,225,-1)
    return biggest_contour,mask

def circle_contour(image,contour):
    #Bounding ellipse


    image_with_ellipse = image.copy()
    ellipse = cv2.fitEllipse(contour)

    #Add it
    cv2.ellipse(image_with_ellipse,ellipse,green,2,cv2.CV_AA)
    return image_with_ellipse



def find_strawberry(image):
    """
    This function allows for detection of images of strawberries by doing the following
    -Takes an image an converts to RGB color scheme
    -Scales the image to correct size
    -Cleans Image by blurring and converting to HSV
    -Defines filters
    -Segmentation (separate strawberry)
    -Find largest object(Strawberrry)
    - Overlay
    - Circle biggest object
    - Convert back to original color scheme
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
    # There arrays consist of hex values for the color red
    min_red = np.array([0,100,80])
    max_red = np.array([10,256,256])

    #Create a layer/mask/filter
    mask1 = cv2.inRange(image_blur_hsv,min_red,max_red)

    min_red2 = np.array([170,100,80])
    max_red2 = np.array([180,256,256])

    mask2 = cv2.inRange(image_blur_hsv, min_red2, max_red2)

    # Combine masks
    mask = mask1 + mask2

    # Segment
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(15,15))
    mask_closed = cv2.morphologyEx(mask,cv2.MORPH_CLOSE,kernel)
    mask_clean = cv2.morphologyEx(mask_closed,cv2.MORPH_OPEN,kernel)

    #Contour Strawberry
    big_strawberry_contour, mask_strawberries = find_biggest_contour(mask_clean)

    #Overlaying masks created on image
    overlay = overlay_mask(mask_clean,image)

    #Circling biggest strawberry
    circed = circle_contour(overlay,big_strawberry_contour)
    show(circled)

    #Converting back to original schm
    bgr = cv2.cvtColor(circled,cv2.COLOR_RGB2BGR)

#Read the image
image = cv2.imread('berry.jpg')
result = find_strawberry(image)
cv2.imwrite('berry2.jpg',result)
