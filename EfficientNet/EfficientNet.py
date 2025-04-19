# from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Activation, Flatten, Dense, Dropout
# from tensorflow.keras.applications.efficientnet import EfficientNetB7
from tf_keras.applications.efficientnet import EfficientNetB7
from tf_keras.models import Sequential



# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
# import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tf_keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tf_keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Activation, Flatten, Dense, Dropout
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory


train_dir = "train"
test_dir = "test"
valid_dir = "valid"



train_datagen = ImageDataGenerator(rescale=1./255,
                             rotation_range=20,
                             width_shift_range=0.1,
                             height_shift_range=0.1,
                             horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale=1./255)


#get the images from train datagen
train_generator = train_datagen.flow_from_directory(train_dir,
                                              target_size=(200, 200),
                                              batch_size=32,
                                              class_mode='categorical',
                                              shuffle=True)
valid_generator = test_datagen.flow_from_directory(valid_dir,
                                              target_size=(200, 200),
                                              batch_size=32,
                                              class_mode='categorical',
                                              shuffle=False)

test_generator = test_datagen.flow_from_directory(test_dir,
                                              target_size=(200, 200),
                                              batch_size=32,
                                              class_mode='categorical',
                                              shuffle=False)


print(test_generator.classes)
print(test_generator.directory)


# for image_batch , labels_batch in train_generator :
#     print(image_batch.shape)
#     print(labels_batch.shape)
#     break

# class_names = train_generator.class_indices
# class_names = list(class_names.keys())
# print(class_names)
#
#
#
# base_model = tf.keras.applications.efficientnet.EfficientNetB7(include_top = False , weights = 'imagenet' ,
#                                                                input_shape = (200,200,3), pooling= 'max')
# model2 = Sequential([ Conv2D(32, (2, 2), activation='relu', input_shape=(200, 200, 3)),
#     MaxPooling2D((2, 2)),
#     Conv2D(64, (2, 2), activation='relu'),
#     MaxPooling2D((2, 2)),
#     Conv2D(128, (2, 2), activation='relu'),
#     MaxPooling2D((2, 2)),
#     Flatten(),
#     Dense(256, activation='relu'),
#     Dropout(0.2),
#     Dense(6 ,activation='softmax')])
#
#
# model2.compile(optimizer="adam",loss="categorical_crossentropy",metrics=["accuracy"])
#
# model2.summary()
#
# history2 = model2.fit(train_generator,
#                     epochs=20,
#                     batch_size=32,
#                     validation_data=valid_generator)
#
#
# def plot_history(history, metric):
#     plt.plot(history2.history[metric])
#     plt.plot(history2.history['val_'+metric], '')
#     plt.xlabel('Epochs')
#     plt.ylabel(metric)
#     plt.legend([metric, 'val_'+metric])
#     plt.show()
#
# plot_history(history2, 'accuracy')
# plot_history(history2, 'loss')