import cv2
import numpy as np
import os
import tensorflow as tf
from random import shuffle
from tqdm import tqdm

import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
import tflearn.datasets.mnist as mnist



Train_DIR = 'E:/dogs-vs-cats/train/train'

Train_cat = 'E:/dogs-vs-cats/cats'
Train_dog = 'E:/dogs-vs-cats/dogs'

Test_DIR = 'E:/dogs-vs-cats/test1/test1'
IMG_SIZE = 50
# image size
LR = 1e-3
# learning rate

MODEL_NAME = 'dogsvscats-{}-{}.model'.format(LR, '2conv-basic')


def label_img(img):
    # img의 라벨링을 부여하는 메소드
    # file example = cat.1.png, dog.100.png
    word_label = img.split('.')[-3]

    if word_label == 'cat' :
        return [1, 0]
    elif word_label == 'dog':
        return [0, 1]
    # one-hot encoding을 위한 특별처리


def create_train_data():
    # train data를 생성하는 메소드
    train_data = []
    for img in tqdm(os.listdir(Train_DIR)):
        label = label_img(img)

        path = os.path.join(Train_DIR, img)
        # Train_DIR의 모든 파일을 경로상으로 읽어들임
        img = cv2.resize(cv2.imread(path, cv2.IMREAD_GRAYSCALE), (IMG_SIZE, IMG_SIZE))
        # openCV의 resize로 해당 이미지의 사이즈를 통일 시킬 수 있다.(굉장히 중요함)
        # 50 x 50 x 1 사이즈로 통일.
        # 가로 x 세로 x 컬러채널

        train_data.append([np.array(img), np.array(label)])
        # train_data에 x = image, y = label 형식으로 넣음.

    shuffle(train_data)
    np.save('train_data.npy', train_data)
    return train_data

def process_test_data():
    test_data = []

    for img in tqdm(os.listdir(Test_DIR)):
        path = os.path.join(Test_DIR, img)
        img_num = img.split('.')[0]
        # test data set's file name format = 1.png 100.png

        img = cv2.resize(cv2.imread(path, cv2.IMREAD_GRAYSCALE), (IMG_SIZE, IMG_SIZE))
        test_data.append([np.array(img), img_num])

    np.save('test_data.npy', test_data)
    return test_data

train_data = create_train_data()
# 처음으로 train data를 생성하는 작업

# 이미 train set이 있을 때,
#train_data = np.load('train_data.npy')


#tensorboard --logdir=C:\Users\chlrj\PycharmProjects\Marware_Classification\log
#텐서보드 확인 방법
tf.reset_default_graph()


### DNN 구축 ###
convnet = input_data(shape=[None, IMG_SIZE, IMG_SIZE, 1], name='input')
# 50 x 50 x 1

convnet = conv_2d(convnet, 32, 2, activation='relu')
convnet = max_pool_2d(convnet, 2)

convnet = conv_2d(convnet, 64, 2, activation='relu')
convnet = max_pool_2d(convnet, 2)

convnet = conv_2d(convnet, 32, 2, activation='relu')
convnet = max_pool_2d(convnet, 2)

convnet = conv_2d(convnet, 64, 2, activation='relu')
convnet = max_pool_2d(convnet, 2)

convnet = conv_2d(convnet, 32, 2, activation='relu')
convnet = max_pool_2d(convnet, 2)

convnet = conv_2d(convnet, 64, 2, activation='relu')
convnet = max_pool_2d(convnet, 2)

convnet = fully_connected(convnet, 1024, activation='relu')
convnet = dropout(convnet, 0.8)

convnet = fully_connected(convnet, 2, activation='softmax')
# 최종 결과는 cats or dogs 이므로 2
convnet = regression(convnet, optimizer='adam', learning_rate=LR
                     , loss='categorical_crossentropy', name='targets')

model = tflearn.DNN(convnet, tensorboard_dir='log')


if os.path.exists('{}.meta'.format(MODEL_NAME)):
    model.load(MODEL_NAME)
    print('model loaded!')
# 예전에 학습시킨 모델을 불러오는 작업

train = train_data[:-500]
test = train_data[-500:]

X = np.array([i[0] for i in train]).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
Y = [i[1] for i in train]

test_x = np.array([i[0] for i in test]).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
test_y = [i[1] for i in test]

model.fit({'input': X}, {'targets': Y}, n_epoch=5, \
          validation_set=({'input': test_x}, {'targets': test_y}),
        snapshot_step=500, show_metric=True, run_id=MODEL_NAME)


model.save(MODEL_NAME)


import matplotlib.pyplot as plt

test_data = process_test_data()

# 이미 test_data가 있으면
#test_data = np.load('test_data.npy')

fig = plt.figure()

for num, data in enumerate(test_data[:12]):
    # cat = [0, 1]
    # dog = [1, 0]

    img_num = data[1]
    img_data = data[0]

    y = fig.add_subplot(3, 4, num+1)
    orig = img_data
    data = img_data.reshape(IMG_SIZE, IMG_SIZE, 1)

    model_out = model.predict([data])[0]
    # predict의 0 index는 예측 정확도가 들어가 있음.

    if np.argmax(model_out) == 1: str_label = 'Dog'
    else : str_label = 'Cat'

    y.imshow(orig, cmap='gray')
    plt.title(str_label)

    y.axes.get_xaxis().set_visible(False)
    y.axes.get_yaxis().set_visible(False)

plt.show()


with open('submission-file.csv', 'w') as f:
    f.write('id,label\n')

with open('submission-file.csv', 'a') as f:
    for data in tqdm(test_data):
        img_num = data[1]
        img_data = data[0]

        y = fig.add_subplot(3, 4, num+1)
        orig = img_data
        data = img_data.reshape(IMG_SIZE, IMG_SIZE, 1)

        model_out = model.predict([data])[0]

        f.write('{},{}\n'.format(img_num, model_out[1]))
