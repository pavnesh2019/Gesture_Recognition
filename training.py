# THE CODE IN THIS FILE TRAINS THE MODEL ON THE DATASET

import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
import glob
import cv2
from sklearn.utils import shuffle
import os

newSize = 64

fistTrainImageFiles = glob.glob("D:/Gesture_Recognition/dataset/fistTrainImages/*.jpg")
fistTrainImageFiles.sort()
fistTrainImage = [cv2.imread(img, 0) for img in fistTrainImageFiles]
for i in range(0,len(fistTrainImage)):
    fistTrainImage[i] = cv2.resize(fistTrainImage[i],(newSize, newSize))
tn0 = np.asarray(fistTrainImage)


fistTestImageFiles = glob.glob("D:/Gesture_Recognition/dataset/fistTestImages/*.jpg")
fistTestImageFiles.sort()
fistTestImage = [cv2.imread(img, 0) for img in fistTestImageFiles]
for i in range(0,len(fistTestImage)):
    fistTestImage[i] = cv2.resize(fistTestImage[i],(newSize, newSize))
ts0  = np.asarray(fistTestImage)


oneTrainImageFiles = glob.glob("D:/Gesture_Recognition/dataset/oneTrainImages/*.jpg")
oneTrainImageFiles.sort()
oneTrainImage = [cv2.imread(img, 0) for img in oneTrainImageFiles]
for i in range(0,len(oneTrainImage)):
    oneTrainImage[i] = cv2.resize(oneTrainImage[i],(newSize, newSize))
tn1 = np.asarray(oneTrainImage)
# check for directory
# cv2.imshow("Image", oneTrainImage[0])
# cv2.waitKey(5000)


oneTestImageFiles = glob.glob("D:/Gesture_Recognition/dataset/oneTestImages/*.jpg")
oneTestImageFiles.sort()
oneTestImage = [cv2.imread(img, 0) for img in oneTrainImageFiles]
for i in range(0,len(oneTestImage)):
    oneTestImage[i] = cv2.resize(oneTestImage[i],(newSize, newSize))
ts1 = np.asarray(oneTestImage)



twoTrainImageFiles = glob.glob("D:/Gesture_Rcognition/dataset/twoTrainImages/*.jpg")
twoTrainImageFiles.sort()
twoTrainImage = [cv2.imread(img, 0) for img in twoTrainImageFiles]
for i in range(0,len(twoTrainImage)):
    twoTrainImage[i] = cv2.resize(twoTrainImage[i],(newSize, newSize))
tn2 = np.asarray(twoTrainImage)
# check for directory
# cv2.imshow("Image", oneTrainImage[0])
# cv2.waitKey(5000)


twoTestImageFiles = glob.glob("D:/Gesture_Rcognition/dataset/twoTestImages/*.jpg")
twoTestImageFiles.sort()
twoTestImage = [cv2.imread(img, 0) for img in twoTrainImageFiles]
for i in range(0,len(twoTestImage)):
    twoTestImage[i] = cv2.resize(twoTestImage[i],(newSize, newSize))
ts2 = np.asarray(twoTestImage)



allTrainImages = []
allTrainImages.extend(fistTrainImage)
allTrainImages.extend(oneTrainImage)
allTrainImages.extend(twoTrainImage)

allTestImages = []
allTestImages.extend(fistTestImage)
allTestImages.extend(oneTestImage)
allTestImages.extend(twoTestImage)

x_train = np.asarray(allTrainImages)
x_test = np.asarray(allTestImages)


y_fistTrainImage = np.empty(tn0.shape[0])
y_oneTrainImage = np.empty(tn1.shape[0])
y_twoTrainImage = np.empty(tn2.shape[0])


y_fistTestImage = np.empty(ts0.shape[0])
y_oneTestImage = np.empty(ts1.shape[0])
y_twoTestImage = np.empty(ts2.shape[0])



for i in range(0, tn0.shape[0]):
    y_fistTrainImage[i] = 0

for i in range(0, tn1.shape[0]):
    y_oneTrainImage[i] = 1

for i in range(0, tn2.shape[0]):
    y_oneTrainImage[i] = 2

for i in range(0, ts0.shape[0]):
    y_fistTestImage[i] = 0

for i in range(0, ts1.shape[0]):
    y_oneTestImage[i] = 1

for i in range(0, ts2.shape[0]):
    y_oneTestImage[i] = 2

y_train_empty = []
y_train_empty.extend(y_fistTrainImage)
y_train_empty.extend(y_oneTrainImage)
y_train_empty.extend(y_twoTrainImage)
y_train = np.asarray(y_train_empty)


y_test_empty = []
y_test_empty.extend(y_fistTestImage)
y_test_empty.extend(y_oneTestImage)
y_test_empty.extend(y_twoTestImage)
y_test = np.asarray(y_test_empty)


print(x_train.shape)
print(y_train.shape)
print()
print(x_test.shape)
print(y_test.shape)

#shuffling the data
x_train,y_train = shuffle(x_train,y_train)
x_test,y_test = shuffle(x_test,y_test)


# flatten 64*64 images to a 4096 vector for each image
num_pixels = x_train.shape[1] * x_train.shape[2]
x_train = x_train.reshape(x_train.shape[0], 1, newSize, newSize).astype('float32')
x_test = x_test.reshape(x_test.shape[0], 1, newSize, newSize).astype('float32')

# normalize inputs from 0-255 to 0-1
x_train = x_train / 255
x_test = x_test / 255

# one hot encode outputs
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)

num_classes = y_test.shape[1]
print("num_classes")
print(num_classes)

# define get model
def get_model():
    model = Sequential()
    model.add(Conv2D(32, (5, 5), input_shape=(1, newSize, newSize), data_format='channels_first', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


# build the model
model = get_model()
model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=10, batch_size=20, verbose=1)

# Final evaluation of the model
scores = model.evaluate(x_test, y_test, verbose=0)
print("Baseline Error: %.2f%%" % (100-scores[1]*100))

# Save the model
model_json = model.to_json();
with open("trainedModel.json","w") as jsonFile:
    jsonFile.write(model_json)
model.save_weights("modelWeights.h5")
print("Saved model to disk")