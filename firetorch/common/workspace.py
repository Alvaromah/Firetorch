import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

class Images():

    @staticmethod
    def load(*route, gray=False, size=None):
        fn = os.path.join(*route)
        if gray:
            img = cv2.imread(fn, cv2.IMREAD_GRAYSCALE)
            if img is None: raise Exception('File not found!')
            img = np.expand_dims(img, axis=2)
        else:
            img = cv2.imread(fn)
            if img is None: raise Exception('File not found!')
        if size:
            img = Images.resize(img, size)
        return img

    @staticmethod
    def rotate(img, angle):
        image_center = tuple(np.array(img.shape[1::-1]) / 2)
        rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
        return cv2.warpAffine(img, rot_mat, img.shape[1::-1], flags=cv2.INTER_LINEAR)

    @staticmethod
    def resize(img, size, padding=8, inter=cv2.INTER_AREA):
        width, height = size
        h, w = img.shape[:2]
        fw, fh = 1, 1
        if w > width: fw = width / w
        if h > height: fh = height / h
        f = min(fw, fh)
        w = int(w * f)
        h = int(h * f)
        resized = cv2.resize(img, (w, h), interpolation=inter)
        if len(resized.shape) == 2:
            resized = np.expand_dims(resized, axis=2)
        arr = resized
        if padding:
            pw = (w // padding) * padding
            ph = (h // padding) * padding
            if pw != w or ph != h:
                cx, cy = (w - pw) // 2, (h - ph) // 2
                arr = resized[cy:ph+cy, cx:pw+cx]
        return arr

    @staticmethod
    def resize_square(img, size, inter=cv2.INTER_AREA):
        width, height = size
        h, w = img.shape[:2]
        fw, fh = 1, 1
        if w > width: fw = width / w
        if h > height: fh = height / h
        f = min(fw, fh)
        w = int(w * f)
        h = int(h * f)
        resized = cv2.resize(img, (w, h), interpolation=inter)
        if len(resized.shape) == 2:
            resized = np.expand_dims(resized, axis=2)
        arr = np.zeros((width, height, img.shape[2]), img.dtype)
        cx, cy = (width - w) // 2, (height - h) // 2
        arr[cy:h+cy, cx:w+cx] = resized
        return arr

    @staticmethod
    def show(img, gray=False):
        if gray:
            plt.imshow(img, cmap='gray')
        else:
            plt.imshow(img)
        plt.show()
