from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Activation, Flatten, Dense, Dropout
from tf_keras.applications.resnet50 import ResNet50

resnet_model = ResNet50(include_top = False , weights = 'imagenet', input_shape = (200,200,3), pooling= 'max')