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
	crop = cv2.resize(img, (160, 160), interpolation=cv2.INTER_CUBIC)
	data= crop.reshape(160,160,3)
	data = prewhiten(data)
	return data

def generate_triplets(data,labels,n_triplets):
	triplets = []
	anchor = []
	pos_sample = []
	neg_sample = []
	for x in range(n_triplets):
		idx = np.random.randint(0, len(labels))
		idx_matches = np.where(labels == labels[idx])[0]
		idx_no_matches = np.where(labels != labels[idx])[0]
		idx_a, idx_p = np.random.choice(idx_matches, 2, replace=True)
		idx_n = np.random.choice(idx_no_matches, 1)[0]
		triplets.append([idx_a, idx_p, idx_n])
	for e in triplets:
		anchor.append(data[e[0]])
		pos_sample.append(data[e[1]])
		neg_sample.append(data[e[2]])
	return np.asarray(anchor),np.asarray(pos_sample),np.asarray(neg_sample)

