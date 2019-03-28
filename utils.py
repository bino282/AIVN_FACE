import cv2
import numpy as np
def read_images(names,mode='train'):
    imgs= []
    for name in names:
        im = cv2.imread('../local/data_face/train/'+name)
        print(im.shape)
        imgs.append(im)
    return np.asarray(imgs)