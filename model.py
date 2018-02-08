import argparse
import base64
import json
import cv2
from keras.models import Sequential
from keras.layers import Dropout
from keras.constraints import maxnorm
from keras.optimizers import SGD, Adadelta
from keras.layers.convolutional import Convolution2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers import InputLayer
from keras.utils import np_utils
from keras import backend as K
K.set_image_dim_ordering('th')
import pandas
from keras.layers.core import Dense, Activation, Flatten
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import numpy as np
import pandas as pd
import socketio
import eventlet
import eventlet.wsgi
import time
from PIL import Image
from PIL import ImageOps
from flask import Flask, render_template
from io import BytesIO
import matplotlib.pyplot as plt
from keras.models import model_from_json
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array

# Fix error with Keras and TensorFlow
import tensorflow as tf
tf.python.control_flow_ops = tf

width = 128
height = 38

def get_unbiased_data(data, prob):
    """
        remove data to take care of training on too much straight data
        input 
            data - dataframe containing image array and steering angles
            prob - percent of straight driving data to clip off
        output
            dataframe 
    """
    output_data = data.drop(data[(data.steering >= -0.025) & (data.steering <= 0.025)].index)
    tmp_data1 = data[(data.steering >= -0.025)]
    tmp_data2 = tmp_data1[(tmp_data1.steering <= 0.025)]
    tmp_data = pd.concat([tmp_data1, tmp_data2])
    tmp_data = tmp_data.sample(frac = prob)
    return pd.concat([output_data, tmp_data])

def prepare_df(data_x):
    """
        prepares a dataframe to be used in the generator to feed to the model
        imput - 
            data_x - dataframe loaded from csv
        output - 
            dataframe with flipped and center,left, right data points

    """
    center = data_x.loc[:, ['center', 'steering']]
    center.columns = ['img', 'steering']
    center['flip'] = False
    left = data_x.loc[:, ['left', 'steering']]
    left.columns = ['img', 'steering']
    left['steering'] = left['steering'].apply(lambda x:x+0.20)
    left['flip'] = False
    right = data_x.loc[:, ['right', 'steering']]
    right.columns = ['img', 'steering']
    right['steering'] = right['steering'].apply(lambda x:x-0.20)
    right['flip'] = False
    center_flipped = data_x.loc[:, ['center', 'steering']]
    center_flipped.columns = ['img', 'steering']
    center_flipped = center_flipped['steering'].apply(lambda x:x*-1)
    center_flipped['flip'] = True
    left_flipped = data_x.loc[:, ['left', 'steering']]
    left_flipped.columns = ['img', 'steering']
    left_flipped['steering'] = left_flipped['steering'].apply(lambda x:(x+0.20)*-1)
    left_flipped['flip'] = True
    right_flipped = data_x.loc[:, ['right', 'steering']]
    right_flipped.columns = ['img', 'steering']
    right_flipped['steering'] = right_flipped['steering'].apply(lambda x:(x-0.20)*-1)
    right_flipped['flip'] = True

    df_list = [center, center_flipped, right, right_flipped, left, left_flipped]

    data = pd.concat(df_list)

    data = get_unbiased_data(data, 0.30)

    data = data[data['img'].notnull()]

    return data

def open_image(image_path, flip=False):
    """
        opens the image and makes them into (None, 3, width, height) shape
        input - 
            image_path - path to image
            flip - if to flip the image or not
        output - 
            np.array containg pixel values
    """
    x = Image.open(image_path)
    x = crop_image(x)
    x = pre_process_image(x)
    image_array = np.asarray(x)
    if flip:
        image_array = cv2.flip(image_array,1)
    image_array = np.rollaxis(image_array,2,0)
    image_array = np.rollaxis(image_array,2,1)
    x.close()
    transformed_image_array = image_array[None, :, :, :]
    return image_array

def pre_process_image(image):
    """
        resize to 128, 38
    """
    width, height = image.size
    basewidth = 128
    wpercent = (basewidth/float(width))
    hsize = int((float(height)*float(wpercent)))
    x = image.resize((basewidth,hsize), Image.ANTIALIAS)
    return x

def crop_image(image):
    """
        cropping the image to remove horizon and car portions
    """
    width, height = image.size
    top_x = 0
    top_y = height * 1.0 / 5
    bottom_x = width
    bottom_y = 8.0 * height / 10
    cropped_image = image.crop((top_x, top_y, bottom_x, bottom_y))
    return cropped_image

def normalize_grayscale(image_data):
    """
        normalizing image from -0.5 to 0.5
    """
    a = -0.5
    b = 0.5
    grayscale_min = 0
    grayscale_max = 255
    return a + ( ( (image_data - grayscale_min)*(b - a) )/( grayscale_max - grayscale_min ) )

def call_model():
    """
        returns model
    """
    model = Sequential()
    model.add(Convolution2D(24, 5, 5, border_mode='valid', activation='elu', subsample=(2,2), input_shape=(3, 128, 38)))
    model.add(Dropout(0.5))
    model.add(Convolution2D(36, 5, 5, activation='relu', border_mode='valid', subsample=(2,2)))
    model.add(Dropout(0.5))
    model.add(Convolution2D(48, 5, 5, activation='relu', border_mode='valid', subsample=(2,2)))
    model.add(Dropout(0.5))
    model.add(Convolution2D(64, 2, 2, activation='relu', border_mode='valid'))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(786, activation='relu', W_constraint=maxnorm(3)))
    model.add(Dropout(0.5))
    model.add(Dense(50, activation='relu', W_constraint=maxnorm(3)))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='tanh'))
    # model.add(Dense(1, init='normal'))
    ada = Adadelta(lr=1.0, rho=0.95, epsilon=1e-08, decay=0.0)
    model.compile(loss='mean_squared_error', optimizer=ada)
    return model

def generate_data(data, batch_size=128, validation=False):
    """
        data generator - for streaming to the model
    """
    batch_x = np.zeros((batch_size, 3, width, height))
    batch_y = np.zeros(batch_size)
    if validation:
        start = int(len(data) * 0.8)
        end = int(len(data))
    else:
        start = 0
        end = int(len(data) * 0.8)
    while 1:
        for item in range(batch_size):
            frame = data.iloc[[np.random.randint(start, end)]].reset_index()
            x = open_image(frame['img'][0].strip(), frame['flip'][0])
            y = frame['steering']
            x = np.asarray(x)
            y = np.asarray(y)
            x = normalize_grayscale(x)
            batch_x[item] = x
            batch_y[item] = y
        yield batch_x, batch_y

epochs = 12
lrate = 0.01
batch_size = 256

model = call_model()
print(model.summary())

data_x = pd.read_csv('./driving_log.csv')
data = prepare_df(data_x)

print(len(data))

generate_data(data)
model.fit_generator(generate_data(data, batch_size), samples_per_epoch=12800, nb_epoch=epochs, validation_data=generate_data(data, validation=True), nb_val_samples=2560)
model_json = model.to_json()
with open("model_temp.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model_temp.h5")
print("Saved model to disk")

