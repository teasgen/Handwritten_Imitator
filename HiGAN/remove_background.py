import cv2
import numpy
import numpy as np


def remove_background_from_image_with_text(img_path):
    # img_path = "./data/background_removed.png"
    # res = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_RGB2GRAY)
    # return np.array(res)
    img = cv2.imread(img_path)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    thresh = cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY_INV)[1]

    contours = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[0]

    new_image = np.zeros(img.shape, np.uint8)

    cv2.drawContours(new_image, contours, -1, (255, 255, 255), thickness=cv2.FILLED)

    mask = np.full(img.shape, 255, dtype=np.uint8)
    result = cv2.bitwise_not(new_image, mask)
    result = 255 - result
    result = cv2.cvtColor(result, cv2.COLOR_RGB2GRAY)
    cv2.imwrite('result.jpg', result)
    return np.array(result)
