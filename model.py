from keras.models import Model
from keras.layers import Input, Conv2D, ZeroPadding2D, MaxPooling2D, Flatten, Dense, Dropout
from PIL import Image
import numpy as np
import pandas as pd
import utils
from keras.models import load_model

model = load_model('../local/data_face/keras-facenet/model/facenet_keras.h5')

data_train = pd.read_csv('../local/data_face/train.csv')
img_names = data_train.image.tolist()
img_arrays = utils.read_images(img_names)
labels = data_train.label.tolist()
model.summary()
model.compile(loss='sparse_categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
face_vectors = utils.l2_normalize(model.predict(img_arrays))
from sklearn import svm
from sklearn.model_selection import cross_val_score
clf = svm.SVC(kernel='linear', C = 1.0)
scores = cross_val_score(clf, face_vectors, labels, cv=5)
print(scores)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))