import cv2
import numpy as np

# Load in image and convert to grayscale
img = cv2.imread('/data/test_wb_image.jpg')
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Create a binary thresholded image
_, thresh = cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY_INV)

# Find contours
contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# Create a black image to draw on
new_image = np.zeros(img.shape, np.uint8)

# Create a white rectangle over the text
cv2.drawContours(new_image, contours, -1, (255, 255, 255), thickness=cv2.FILLED)

# Create a mask with the same size as the image
mask = np.full(img.shape, 255, dtype=np.uint8)

cv2.imwrite('/data/out.jpg', 0.75 * new_image)