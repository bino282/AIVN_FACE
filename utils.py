import cv2
import numpy as np
def read_images(names,mode='train'):
    imgs= []
    for name in names:
        im = cv2.imread('../local/data_face/{}/'.format(mode)+name)
        im = norm_image(im)
        imgs.append(im)
    return np.asarray(imgs)

def prewhiten(x):
	mean = np.mean(x)
	std = np.std(x)
	std_adj = np.maximum(std, 1.0/np.sqrt(x.size))
	y = np.multiply(np.subtract(x, mean), 1/std_adj)
	return y 
def norm_image(img):
	crop = cv2.resize(img, (160, 160), interpolation=cv2.INTER_CUBIC )
	data= crop.reshape(160,160,3)
	data = prewhiten(data)
	return data 