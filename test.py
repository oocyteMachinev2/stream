from StartStreamCamera_Running import *
import cv2

img = export_image()
image = cv2.GaussianBlur(img, (5, 5), 0)  # Kích thước kernel là 5x5, có thể điều chỉnh tùy ý
image = cv2.medianBlur(image, 5)
cv2.imshow('anh', image)