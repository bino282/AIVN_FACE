from keras.models import Model
from keras.layers import Input, Conv2D, ZeroPadding2D, MaxPooling2D, Flatten, Dense, Dropout
from PIL import Image
import numpy as np
import pandas as pd
from utils import *
from keras.models import load_model
import os
import numpy as p
import keras.layers as kl
from keras import backend as K
from sklearn.preprocessing import normalize
def triplet_loss(y_true, y_pred):
    margin = K.constant(1)
    return K.mean(K.maximum(K.constant(0), K.square(y_pred[:,0,0]) - 0.5*(K.square(y_pred[:,1,0])+K.square(y_pred[:,2,0])) + margin))
def euclidean_distance(vects):
    x, y = vects
    return K.sqrt(K.maximum(K.sum(K.square(x - y), axis=1, keepdims=True), K.epsilon()))
model_facenet = load_model('../local/data_face/keras-facenet/model/facenet_keras.h5')

def my_model(old_model):
    input_anchor = kl.Input(shape=(160, 160, 3))
    input_positive = kl.Input(shape=(160, 160, 3))
    input_negative = kl.Input(shape=(160, 160, 3))
    print(input_anchor)
    anchor_rep = old_model(input_anchor)
    pos_rep = old_model(input_positive)
    neg_rep = old_model(input_negative)

    positive_dist = kl.Lambda(euclidean_distance, name='pos_distance')([anchor_rep, pos_rep])
    negative_dist = kl.Lambda(euclidean_distance, name='neg_distance')([anchor_rep, neg_rep])
    tertiary_dist = kl.Lambda(euclidean_distance, name='ter_distance')([pos_rep, neg_rep])

    stacked_dists = kl.Lambda(lambda vects: K.stack(vects, axis=1), name='stacked_dists')([positive_dist, negative_dist, tertiary_dist])

    model_final = Model([input_anchor, input_positive, input_negative], stacked_dists, name='triple_siamese')

    return model_final



data_train = pd.read_csv('../local/data_face/train.csv')
img_names = data_train.image.tolist()
img_arrays = read_images(img_names)
labels = data_train.label.tolist()

anchor,pos,neg = generate_triplets(img_arrays,np.asarray(labels),len(labels))

my_model = my_model(model_facenet)
print(my_model.summary())
my_model.compile(loss=triplet_loss,optimizer='adam',metrics=['accuracy'])
my_model.fit([anchor,pos,neg],labels,epochs=500,validation_split=0.1,verbose=1)
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


