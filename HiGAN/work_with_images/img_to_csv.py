from copy import deepcopy

import numpy as np
from PIL import Image
import pandas as pd
from torchvision.transforms import Compose, Normalize, ToTensor
import matplotlib.pyplot as plt
import cv2


# df = pd.read_csv('test_img.csv', header=None, index_col=None)
# df = np.asarray(df)
# print(df.shape)
#
# plt.figure()
# plt.subplot(2, 1, 1)
# plt.axis('off')
# plt.imshow(df, cmap='gray')
# plt.savefig('/home/vlad/workspace/course_work/HiGANplus/HiGAN+/data/wb.png', bbox_inches='tight', pad_inches=0)

img = Image.open('../course_work/HiGANplus/HiGAN+/data/background_removed.png')
img_gray = img.convert('L')
img = 0.75 * np.asarray(img_gray)
print(img.shape)
img = cv2.resize(deepcopy(img), (img.shape[1] * 64 // img.shape[0], 64), interpolation=cv2.INTER_AREA)
img = img.astype(int)
# plt.subplot(2, 1, 2)
plt.imshow(img, cmap='gray')
plt.tight_layout()
plt.show()

pd.DataFrame(img).to_csv('test_img2.csv', header=None, index=None)

print(img.mean())

