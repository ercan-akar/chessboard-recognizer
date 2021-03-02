import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
import cv2
import os
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras import layers
import h5py

if __name__ == '__main__':
    model_path = './nn/my_h5_model.h5'
    class_model = load_model(model_path)

    model_path2 = './nn/model.tf/'
    # weight_path2='./nn/model.tf'
    # class_model2 = load_model(model_path2)

    imported = tf.saved_model.load(model_path2)

    # squares=os.listdir('pieces')
    # square= cv2.imread(os.path.join('pieces',name),0)
    # square = cv2.resize(square,(32,32))
    x = cv2.imread('./nn/c3_R.png')
    # square = cv2.resize(square,(32,32),interpolation=cv2.INTER_CUBIC)
    # square_class = cv2.resize(square,(128,128),interpolation=cv2.INTER_CUBIC)

    # x = cv2.cvtColor(square, cv2.COLOR_GRAY2RGB)
    # x_class = cv2.cvtColor(square_class, cv2.COLOR_GRAY2RGB)

    x = img_to_array(x)
    print(x.shape)
    # x = np.expand_dims(x, axis=0)
    print(x.shape)

    tf.keras.layers.InputLayer(input_shape=(32, 32, 3))

    print(class_model.summary())

    # array = ml_model.binary_model.predict(x)
    array = class_model.predict(x)

    result = array[0]
    answer = np.argmax(result)
    print(f"{answer}")
