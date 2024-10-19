import time
import keras
import numpy as np
import tensorflow as tf

from datetime import datetime
import keras
from keras import layers
from keras.layers import Input, Dropout, Dense, Flatten, BatchNormalization, Activation, Conv2D, MaxPooling2D, UpSampling2D, Conv2DTranspose
from keras.models import Model
from keras.applications.resnet import ResNet50
from keras.applications.xception import Xception
from keras.applications import DenseNet201
from keras.preprocessing import image
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.optimizers import Adam, RMSprop

def Unet_Model(input_shape, init_filter, drop_rate, up_sampling=False, 
         regularization=False, batch_norm=False):
    # Set Up Regularization
    if regularization == None:
        kernel_regularizer = None
    elif regularization == 'l1':
        kernel_regularizer = l1(0.001)
    elif regularization == 'l2':
        kernel_regularizer = l2(0.001)
    
    # Input Layer
    input_layer = Input(input_shape)
    
    # Contraction Path
    # 1st Contraction
    c1 = Conv2D(filters=init_filter, kernel_size=(3,3), kernel_initializer='he_normal', padding='same', kernel_regularizer=kernel_regularizer)(input_layer)
    if batch_norm:
        c1 = BatchNormalization()(c1)
    c1 = Activation('relu')(c1)
    
    c1 = Conv2D(filters=init_filter, kernel_size=(3,3), kernel_initializer='he_normal', padding='same', kernel_regularizer=kernel_regularizer)(c1)
    if batch_norm:
        c1 = BatchNormalization()(c1)
    c1 = Activation('relu')(c1)
    
    p1 = MaxPooling2D((2,2))(c1)
    p1 = Dropout(drop_rate)(p1)

    # 2nd Contraction
    c2 = Conv2D(filters=init_filter*2, kernel_size=(3,3), kernel_initializer='he_normal', padding='same', kernel_regularizer=kernel_regularizer)(p1)
    if batch_norm:
        c2 = BatchNormalization()(c2)
    c2 = Activation('relu')(c2)
    
    c2 = Conv2D(filters=init_filter*2, kernel_size=(3,3), kernel_initializer='he_normal', padding='same', kernel_regularizer=kernel_regularizer)(c2)
    if batch_norm:
        c2 = BatchNormalization()(c2)
    c2 = Activation('relu')(c2)

    p2 = MaxPooling2D((2,2))(c2)
    p2 = Dropout(drop_rate)(p2)

    # 3rd Contraction
    c3 = Conv2D(filters=init_filter*4, kernel_size=(3,3), kernel_initializer='he_normal', padding='same', kernel_regularizer=kernel_regularizer)(p2)
    if batch_norm:
        c3 = BatchNormalization()(c3)
    c3 = Activation('relu')(c3)
    
    c3 = Conv2D(filters=init_filter*4, kernel_size=(3,3), kernel_initializer='he_normal', padding='same', kernel_regularizer=kernel_regularizer)(c3)
    if batch_norm:
        c3 = BatchNormalization()(c3)
    c3 = Activation('relu')(c3)

    p3 = MaxPooling2D((2,2))(c3)
    p3 = Dropout(drop_rate)(p3)

    # 4th Contraction
    c4 = Conv2D(filters=init_filter*8, kernel_size=(3,3), kernel_initializer='he_normal', padding='same', kernel_regularizer=kernel_regularizer)(p3)
    if batch_norm:
        c4 = BatchNormalization()(c4)
    c4 = Activation('relu')(c4)
    
    c4 = Conv2D(filters=init_filter*8, kernel_size=(3,3), kernel_initializer='he_normal', padding='same', kernel_regularizer=kernel_regularizer)(c4)
    if batch_norm:
        c4 = BatchNormalization()(c4)
    c4 = Activation('relu')(c4)

    p4 = MaxPooling2D((2,2))(c4)
    p4 = Dropout(drop_rate)(p4)

    # 5th Contraction
    c5 = Conv2D(filters=init_filter*16, kernel_size=(3,3), kernel_initializer='he_normal', padding='same', kernel_regularizer=kernel_regularizer)(p4)
    if batch_norm:
        c5 = BatchNormalization()(c5)
    c5 = Activation('relu')(c5)
    
    c5 = Conv2D(filters=init_filter*16, kernel_size=(3,3), kernel_initializer='he_normal', padding='same', kernel_regularizer=kernel_regularizer)(c5)
    if batch_norm:
        c5 = BatchNormalization()(c5)
    c5 = Activation('relu')(c5)
    c5 = Dropout(drop_rate)(c5)

    # Expansion Path
    # 1st Expansion
    if up_sampling:
        u6 = UpSampling2D(size=(2,2), data_format=None, interpolation='bilinear')(c5)
    else:
        u6 = Conv2DTranspose(filters=init_filter*8, kernel_size=(3,3), strides=(2,2), padding='same')(c5)
    u6 = layers.concatenate([u6, c4])
    c6 = Dropout(drop_rate)(u6)
    
    c6 = Conv2D(filters=init_filter*8, kernel_size=(3,3), kernel_initializer='he_normal', padding='same', kernel_regularizer=kernel_regularizer)(u6)
    if batch_norm:
        c6 = BatchNormalization()(c6)
    c6 = Activation('relu')(c6)

    c6 = Conv2D(filters=init_filter*8, kernel_size=(3,3), kernel_initializer='he_normal', padding='same', kernel_regularizer=kernel_regularizer)(c6)
    if batch_norm:
        c6 = BatchNormalization()(c6)
    c6 = Activation('relu')(c6)

    # 2nd Expansion
    if up_sampling:
        u7 = UpSampling2D(size=(2,2), data_format=None, interpolation='bilinear')(c6)
    else:
        u7 = Conv2DTranspose(filters=init_filter*4, kernel_size=(3,3), strides=(2,2), padding='same')(c6)
    u7 = layers.concatenate([u7, c3])
    c7 = Dropout(drop_rate)(u7)
    
    c7 = Conv2D(filters=init_filter*4, kernel_size=(3,3), kernel_initializer='he_normal', padding='same', kernel_regularizer=kernel_regularizer)(u7)
    if batch_norm:
        c7 = BatchNormalization()(c7)
    c7 = Activation('relu')(c7)

    c7 = Conv2D(filters=init_filter*4, kernel_size=(3,3), kernel_initializer='he_normal', padding='same', kernel_regularizer=kernel_regularizer)(c7)
    if batch_norm:
        c7 = BatchNormalization()(c7)
    c7 = Activation('relu')(c7)

    # 3rd Expansion
    if up_sampling:
        u8 = UpSampling2D(size=(2,2), data_format=None, interpolation='bilinear')(c7)
    else:
        u8 = Conv2DTranspose(filters=init_filter*2, kernel_size=(3,3), strides=(2,2), padding='same')(c7)
    u8 = layers.concatenate([u8, c2])
    c8 = Dropout(drop_rate)(u8)
    
    c8 = Conv2D(filters=init_filter*2, kernel_size=(3,3), kernel_initializer='he_normal', padding='same', kernel_regularizer=kernel_regularizer)(u8)
    if batch_norm:
        c8 = BatchNormalization()(c8)
    c8 = Activation('relu')(c8)

    c8 = Conv2D(filters=init_filter*2, kernel_size=(3,3), kernel_initializer='he_normal', padding='same', kernel_regularizer=kernel_regularizer)(c8)
    if batch_norm:
        c8 = BatchNormalization()(c8)
    c8 = Activation('relu')(c8)

    # 4th Expansion
    if up_sampling:
        u9 = UpSampling2D(size=(2,2), data_format=None, interpolation='bilinear')(c8)
    else:
        u9= Conv2DTranspose(filters=init_filter*8, kernel_size=(3,3), strides=(2,2), padding='same')(c8)
    u9 = layers.concatenate([u9, c1], axis=3)
    c9 = Dropout(drop_rate)(u9)
    
    c9 = Conv2D(filters=init_filter, kernel_size=(3,3), kernel_initializer='he_normal', padding='same', kernel_regularizer=kernel_regularizer)(u9)
    if batch_norm:
        c9 = BatchNormalization()(c9)
    c9 = Activation('relu')(c9)

    c9 = Conv2D(filters=init_filter, kernel_size=(3,3), kernel_initializer='he_normal', padding='same', kernel_regularizer=kernel_regularizer)(c9)
    if batch_norm:
        c9 = BatchNormalization()(c9)
    c9 = Activation('relu')(c9)

    # Output layer
    output_layer = Conv2D(1, (1, 1), activation='sigmoid')(c9)

    UNet_Model = Model(inputs=[input_layer], outputs=[output_layer])

    return UNet_Model

def CNN_Model(input_shape, num_layer, num_unit, drop_rate):
    # Input layer
    img_input = Input(shape=input_shape)
    mask_input = Input(shape=input_shape)

    # Concat Input
    input_concat = layers.Concatenate()([img_input, mask_input, mask_input])    
    
    # Pre-Trained DenseNet201
    pretrained_model = DenseNet201(weights="imagenet", include_top=False, input_tensor=input_concat)
    for layer in pretrained_model.layers:
        layer.trainable = False
    
    # Flatten Layer
    x = Flatten()(pretrained_model.output)

    # Fully Connected Layer
    for _ in range(num_layer):
        x = Dense(num_unit, activation="relu")(x)
    x = Dropout(drop_rate)(x)

    # Output Layer
    predictions = tf.keras.layers.Dense(3, activation="softmax")(x)

    CNN_model = Model(inputs=[img_input, mask_input], outputs=predictions)
    return CNN_model