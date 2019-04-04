from keras.models import Model
from keras.layers import Input, Conv2D, ZeroPadding2D, MaxPooling2D, Flatten, Dense, Dropout
from PIL import Image
import numpy as np
import pandas as pd
from utils import *
from keras.models import load_model
import os
import numpy as p
from sklearn.preprocessing import normalize
model = load_model('../local/data_face/keras-facenet/model/facenet_keras.h5')



data_train = pd.read_csv('../local/data_face/train.csv')
img_names = data_train.image.tolist()
img_arrays = read_images(img_names)
labels = data_train.label.tolist()

anchor,pos,neg = generate_triplets(img_arrays,np.asarray(labels),10*len(labels))
print(model.summary())
model.compile(loss='sparse_categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
model.fit(img_arrays,labels,epochs=500,validation_split=0.1,verbose=1)
test_dir = '../local/data_face/test'
test_names = os.listdir(test_dir)
test_imgs = read_images(test_names,mode='test')
face_vectors = model.predict(img_arrays)
test_vector = model.predict(test_imgs)
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


