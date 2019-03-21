import pandas as pd
import cv2
data_train = pd.read_csv('train.csv')
print(data_train.image[0])
img = cv2.imread('./train/'+data_train.image[0])
cv2.imshow('Face',img)
cv2.waitKey()