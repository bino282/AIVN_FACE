from keras.models import Model
from keras.layers import Input, Conv2D, ZeroPadding2D, MaxPooling2D, Flatten, Dense, Dropout
from PIL import Image
import numpy as np
import pandas as pd
import utils
from keras.models import load_model
import os
import numpy as p
model = load_model('../local/data_face/keras-facenet/model/facenet_keras.h5')

data_train = pd.read_csv('../local/data_face/train.csv')
img_names = data_train.image.tolist()
img_arrays = utils.read_images(img_names)
labels = data_train.label.tolist()
model.summary()
face_vectors = model.predict(img_arrays)
from sklearn import svm
from sklearn.model_selection import cross_val_score
clf = svm.SVC(kernel='linear', C = 1.0,probability=True)
scores = cross_val_score(clf, face_vectors, labels, cv=5)


test_dir = '../local/data_face/test'
test_names = os.listdir(test_dir)
test_imgs = utils.read_images(test_names)
test_vector = model.predict(test_imgs)
results = []
for v in test_vector:
    prob= clf.predict_proba(v)
    ids = prob.argsort()[::-1][0:5]
    results.append(ids)
print(results)


