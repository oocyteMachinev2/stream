# this file use for define some common function like filtering, coloring, bluring , etc. 

import cv2
import numpy as np

class imageProcessor:
    def __init__(self, image_path):
        # Read image from the path
        self.original_image = cv2.imread(image_path)

        # Apply filters after creating an object
        self.gray_image = self.apply_grayscale(self.original_image)
        self.blurred_image = self.apply_gaussian_blur(self.gray_image)
        self.median_blur_image = self.apply_median_blur(self.blurred_image)
        self.threshold_image = self.apply_threshold(self.median_blur_image)
        self.canny_image = self.apply_canny(self.threshold_image)
        self.contours, self.hierarchy = self.find_contours(self.canny_image)
        
    def apply_grayscale(self, image):
        """Turn image into Grayscale."""
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    def apply_gaussian_blur(self, image, kernel_size=(3, 3), sigma=1.2):
        """Apply Gaussian Blur."""
        return cv2.GaussianBlur(image, kernel_size, sigma)

    def apply_median_blur(self, image, kernel_size = 5):
        """Apply Median Blur."""
        return cv2.medianBlur(image, kernel_size)
    
    def apply_threshold(self, image, thresh_val=160, max_val=255):
        """Apply Thresholding."""
        _, thresholded = cv2.threshold(image, thresh_val, max_val, cv2.THRESH_BINARY)
        return thresholded

    def apply_canny(self, image, threshold1=20, threshold2=130):
        """Apply Canny Edge Detection."""
        return cv2.Canny(image, threshold1, threshold2)

    def find_contours(self, image):
        """Find and return contours from the Canny Image."""
        contours, hierarchy = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return contours, hierarchy
    

# /////////////////// Test /////////////////////


# dartboard_img = imageProcessor('anhTestGieng/teneImage/log_image/arcane.jpg')

# original = dartboard_img.original_image
# gray = dartboard_img.gray_image
# blur = dartboard_img.blurred_image
# thres = dartboard_img.threshold_image
# canny = dartboard_img.canny_image
# contours_img = original.copy()
# contours = dartboard_img.contours

# for i, contour in enumerate(contours):
#     cv2.drawContours(contours_img , contours, i, (0, 255, 0), 2)
    
# cv2.imshow('Original Image', original)
# cv2.imshow('Gray Image', gray)
# cv2.imshow('Blurred Image', blur)
# cv2.imshow('Threshold Image', thres)
# cv2.imshow('Canny Image', canny)
# cv2.imshow('Contours', contours_img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
