import cv2
import numpy as np
def read_images(names,mode='train'):
    imgs= []
    for name in names:
        im = cv2.imread('../local/data_face/train/'+name)
        im = cv2.resize(im, (224, 224)) 
        imgs.append(im)
    return np.asarray(imgs)