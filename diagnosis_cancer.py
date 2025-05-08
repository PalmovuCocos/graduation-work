import csv
import os
import cv2
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import random

from tensorflow import keras
from tensorflow.keras.layers import Dense, Flatten, Dropout, Conv2D, MaxPooling2D, Input


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

path1 = "cancer\\HAM10000_metadata.csv"
DATADIRTrain = "cancer"
category = []
training_date = []
name_img = []
image = {}
IMG_DIR = ['HAM10000_images_part_1', 'HAM10000_images_part_2']#HAM10000_images_part_2

with open(path1, newline='') as csvfile:
    reader = csv.DictReader(csvfile, delimiter=",")
    for row in reader:
        category.append(row['dx'])
        name_img.append(row['image_id']+'.jpg')

for i in range(len(category)):
    if category[i] == 'akiec':
        image[name_img[i]] = 0
    if category[i] == 'df':
        image[name_img[i]] = 1
    if category[i] == 'mel':
        image[name_img[i]] = 2
    if category[i] == 'bcc':
        image[name_img[i]] = 3
    if category[i] == 'vasc':
        image[name_img[i]] = 4
    if category[i] == 'nv':
        image[name_img[i]] = 5
    if category[i] == 'bkl':
        image[name_img[i]] = 6


print(image)
IMG_SIZE = 100
for img_dir in IMG_DIR:
    path = os.path.join(DATADIRTrain, img_dir)
    for img in os.listdir(path):
        try:
            img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_COLOR)
            new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
            training_date.append([new_array, image[img]])
        except Exception as e:
            pass




random.shuffle(training_date)


x_train = []
y_train = []
for features, label in training_date:
    x_train.append(features)
    y_train.append(label)


x_train = np.array(x_train)
y_train = np.array(y_train)

# преобразование цифр в вектор???(размером в 10)(ставится на позицию вектора число из x_train)
y_train_cat = keras.utils.to_categorical(y_train, 7)


# модель многослойной нс
model = keras.Sequential([
   Input(shape=(100, 100, 3)),  # Явное указание входной формы
   Conv2D(128, (3,3), padding='same', activation='relu'), # прогонка изображения через ядра
   MaxPooling2D((2, 2), strides=2),

   Conv2D(64, (3, 3), padding='same', activation='relu'),
   MaxPooling2D((2, 2), strides=2),

   Conv2D(32, (3, 3), padding='same', activation='relu'),
   MaxPooling2D((2, 2), strides=2),

   Conv2D(16, (3, 3), padding='same', activation='relu'),
   MaxPooling2D((2, 2), strides=2),

   Flatten(),   # создание входов для подачи изображения 28 на 28

   Dense(128, activation='relu'),
   Dense(7, activation='softmax')  # выходной слой
])
print(model.summary())

# компиляция нейронной сети
model.compile(
    optimizer='nadam',
    loss='categorical_crossentropy',  # выбрали этот критерий качества, то что тут задача классификации
    metrics=['accuracy']  # видим показатели в процентах ???
)

# запуск процесса обучения
his = model.fit(x_train[:1000],  # входное обучающее множество
          y_train_cat[:1000],  # значение изображений в виде вектора
          batch_size=40,    # размер батча (обучающая выборка делится на блоки по 32)
          epochs=15,         # эпохи
          validation_split=0.2)   # разбиение обучающей выборки на обучающую и проверочную
                                  # (будут переходить в выборку валидации 20%)
print(model.summary())
#model.evaluate(x_test, y_test_cat)
model.save('1000.h5')
print(his.history['loss'], his.history['val_loss'])
plt.plot(his.history['loss'])
plt.plot(his.history['val_loss'])
plt.show()

