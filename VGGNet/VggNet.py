# from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Activation, Flatten, Dense, Dropout
from tf_keras.applications.vgg16 import VGG16

resnet_model = VGG16(include_top = False , weights = 'imagenet', input_shape = (200,200,3), pooling= 'max')