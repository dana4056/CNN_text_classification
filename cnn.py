from PIL import Image
import os, glob
import cv2 as cv
import numpy as np

# load_data 함수
def load_data(img_path, number):
    # A: 0, B: 1, C: 2, ...
    number_of_data = number   # 문자 이미지 개수 총합
    img_size= 28
    color = 1

    #이미지 데이터와 라벨(A: 0, B: 1, C: 2) 데이터를 담을 행렬(matrix) 영역을 생성
    imgs = np.zeros(number_of_data * img_size * img_size * color  , dtype=np.int32).reshape(number_of_data,img_size,img_size,color)
    labels = np.zeros(number_of_data, dtype=np.int32)

    idx=0

    for file in glob.iglob(img_path+'/A/*.jpg'):
        img = np.array(Image.open(file), dtype=np.int32).reshape(img_size,img_size,color)
        imgs[idx,:,:]=img    # 데이터 영역에 이미지 행렬을 복사
        labels[idx] = 0   # A: 0
        idx = idx + 1

    for file in glob.iglob(img_path+'/B/*.jpg'):
        img = np.array(Image.open(file), dtype=np.int32).reshape(img_size,img_size,color)
        imgs[idx,:,:]=img    # 데이터 영역에 이미지 행렬을 복사
        labels[idx]= 1
        idx = idx + 1

    for file in glob.iglob(img_path + '/C/*.jpg'):
        img = np.array(Image.open(file), dtype=np.int32).reshape(img_size, img_size, color)
        imgs[idx, :, :] = img  # 데이터 영역에 이미지 행렬을 복사
        labels[idx] = 2
        idx = idx + 1

    for file in glob.iglob(img_path + '/D/*.jpg'):
        img = np.array(Image.open(file), dtype=np.int32).reshape(img_size, img_size, color)
        imgs[idx, :, :] = img  # 데이터 영역에 이미지 행렬을 복사
        labels[idx] = 3  # S: 6
        idx = idx + 1
    for file in glob.iglob(img_path + '/E/*.jpg'):
        img = np.array(Image.open(file), dtype=np.int32).reshape(img_size, img_size, color)
        imgs[idx, :, :] = img  # 데이터 영역에 이미지 행렬을 복사
        labels[idx] = 4
        idx = idx + 1

    for file in glob.iglob(img_path + '/W/*.jpg'):
        img = np.array(Image.open(file), dtype=np.int32).reshape(img_size, img_size, color)
        imgs[idx, :, :] = img  # 데이터 영역에 이미지 행렬을 복사
        labels[idx] = 5
        idx = idx + 1

    for file in glob.iglob(img_path + '/S/*.jpg'):
        img = np.array(Image.open(file), dtype=np.int32).reshape(img_size, img_size, color)
        imgs[idx, :, :] = img  # 데이터 영역에 이미지 행렬을 복사
        labels[idx] = 6
        idx = idx + 1

    for file in glob.iglob(img_path + '/N/*.jpg'):
        img = np.array(Image.open(file), dtype=np.int32).reshape(img_size, img_size, color)
        imgs[idx, :, :] = img  # 데이터 영역에 이미지 행렬을 복사
        labels[idx] = 7
        idx = idx + 1


    print("학습데이터(x_train)의 이미지 개수는", idx, "입니다.")
    return imgs, labels


image_dir_path = "Data/train_dataset"
(x_train, y_train) = load_data(image_dir_path, 5022+6194+6624+5499+5217+5144+4721+5053)
x_train = x_train/255.0   # 입력은 0~1 사이의 값으로 정규화

print("x_train shape: {}".format(x_train.shape))
print("y_train shape: {}".format(y_train.shape))


# 불러온 이미지 확인
import matplotlib.pyplot as plt
plt.imshow(x_train[1000])
print(x_train[1000].shape)
plt.show()
cv.waitKey(0)
print('라벨: ', y_train[1000])

# 딥러닝 네트워크 설계
import tensorflow as tf
from tensorflow import keras
import numpy as np


n_channel_1 = 16
n_channel_2 = 32
n_dense = 64
n_train_epoch = 5

model = keras.models.Sequential()
model.add(keras.layers.Conv2D(n_channel_1, (3,3), activation='relu', input_shape=(28,28,1)))
model.add(keras.layers.MaxPool2D(2,2))
model.add(keras.layers.Conv2D(n_channel_2, (3,3), activation='relu'))
model.add(keras.layers.MaxPooling2D((2,2)))
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(n_dense, activation='relu'))
model.add(keras.layers.Dense(8, activation='softmax'))

model.summary()


# 모델 학습
model.compile(optimizer='adam',
             loss='sparse_categorical_crossentropy',
             metrics=['accuracy'])

model.fit(x_train, y_train, epochs=n_train_epoch)

# 테스트 이미지
image_dir_path = "Data/test_dataset"
(x_test, y_test)=load_data(image_dir_path, 1280+1646+1426+1021+1135+1076+1012+1069)
x_test = x_test/255.0   # 입력은 0~1 사이의 값으로 정규화

print("x_train shape: {}".format(x_test.shape))
print("y_train shape: {}".format(y_test.shape))

# 불러온 이미지 확인
import matplotlib.pyplot as plt
plt.imshow(x_test[300])
plt.show()
cv.waitKey(0)
print('라벨: ', y_test[300])

# 모델 테스트
test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=2)
print("test_loss: {} ".format(test_loss))
print("test_accuracy: {}".format(test_accuracy))

model.save("text_CNN.h5")