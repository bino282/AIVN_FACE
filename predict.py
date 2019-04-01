import pickle
import cv2
import facenet
import tensorflow as tf
import numpy as np
import os
from utils import *
import pandas as pd
from sklearn.preprocessing import normalize
model_dir = "../local/pre_model"

def predict(face_imgs,sess):
	img_feature = sess.run([embeddings],feed_dict={images_placeholder: face_imgs,phase_train_placeholder: False }).tolist()
	return img_feature

data_train = pd.read_csv('../local/data_face/train.csv')
img_names = data_train.image.tolist()
img_arrays = read_images(img_names)
labels = data_train.label.tolist()
batch_size = 128
with tf.Graph().as_default():
	with tf.Session(config=tf.ConfigProto(log_device_placement=False)) as sess:
		facenet.load_model(model_dir)
		images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
		embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
		phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
		print ("load model succees !")
		face_vectors = []
		for i in range(img_arrays.shape[0]//batch_size):
			face_v = predict(img_arrays[i*batch_size:(i+1)*batch_size],sess)
			face_vectors.extend(face_v)


		face_vectors = np.asarray(face_vectors)
		print(face_vectors.shape)

		test_dir = '../local/data_face/test'
		test_names = os.listdir(test_dir)
		test_imgs = read_images(test_names,mode='test')
		test_vector = predict(test_imgs,sess)
		print('get vector finish')
		results = []
		face_vectors = normalize(face_vectors,axis=1,norm='l2')
		for v in test_vector:
			v = v.reshape(1,-1)
			v= normalize(v,axis=1,norm='l2')
			prob = np.dot(v,face_vectors.T).flatten()
			ids_set = set()
			ids = prob.argsort()[::-1]
			for j in  range (len(ids)):
				if (len(ids_set) < 5):
					ids_set.add(labels[ids[j]])
				else:
					break
			results.append(list(ids_set))
		print(results)
		fw = open('submission.csv','w')
		fw.write('image,label')
		fw.write('\n')
		for i in range(len(test_names)):
			fw.write(test_names[i] + ','+" ".join([str(l) for l in results[i]]))
			fw.write('\n')
		fw.close()



