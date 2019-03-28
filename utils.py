import cv2
import numpy as np
def read_images(names,mode='train'):
    imgs= []
    for name in names:
        imgs.append(cv2.imread('../local/data_face/train/'+name))
    return np.asarray(imgs)