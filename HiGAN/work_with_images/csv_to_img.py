import numpy as np
from PIL import Image
import pandas as pd
from torchvision.transforms import Compose, Normalize, ToTensor
import matplotlib.pyplot as plt
import cv2
import csv

# Load the CSV file into a 2D list
with open('/home/vlad/workspace/Handwritten_Imitator/HiGAN/data/trained_image_example.csv', 'r') as f:
    reader = csv.reader(f)
    data = [list(map(float, row)) for row in reader]

# Normalize the values to be between 0 and 1
data = np.array(data)
data /= data.max()

# Multiply the values by 255 to convert them to the range of 0 to 255
data *= 255
data = data.astype('uint8')

# Create a new Image object
height, width = data.shape
img = Image.new('L', (width, height))

# Loop through each pixel in the image and set its value
for y in range(height):
    for x in range(width):
        img.putpixel((x, y), int(data[y][x]))

# Save the image to a file
img.save('output.png')
plt.imshow(img, cmap='gray')
plt.tight_layout()
plt.show()
