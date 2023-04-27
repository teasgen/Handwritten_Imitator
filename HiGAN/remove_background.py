import cv2
import numpy as np


def remove_background_from_image_with_text(img_path):
    img = cv2.imread(img_path)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    thresh = cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY_INV)[1]

    contours = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[0]

    new_image = np.zeros(img.shape, np.uint8)

    cv2.drawContours(new_image, contours, -1, (255, 255, 255), thickness=cv2.FILLED)

    mask = np.full(img.shape, 255, dtype=np.uint8)
    result = cv2.bitwise_not(new_image, mask)
    result = 255 - result
    result = 0.75 * np.array(cv2.cvtColor(result, cv2.COLOR_RGB2GRAY))

    mask = np.any(result != 0, axis=1)
    rows = np.where(mask == np.max(mask))
    mask = np.any(result != 0, axis=0)
    columns = np.where(mask == np.max(mask))
    result = result[rows[0][0]: rows[0][-1] + 1, columns[0][0]: columns[0][-1] + 1]
    return result
