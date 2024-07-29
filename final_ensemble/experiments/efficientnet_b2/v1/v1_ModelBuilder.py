import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras import mixed_precision

import tensorflow_addons as tfa

from vit_keras import vit

mixed_precision.set_global_policy('mixed_float16')



class ModelBuilder():
    def __init__(self):
        pass

    def build_efficientnet():
        pretrained_efficientnet_model = tf.keras.applications.EfficientNetB2(
            input_shape=(224, 224, 3),
            include_top=False,
            weights='imagenet',
            pooling='max'
        )
                
        num_classes = 200

        inputs = layers.Input(shape = (224,224,3), name='inputLayer')
        x = inputs
        pretrain_out = pretrained_efficientnet_model(x, training = False)
        x = layers.Dense(256)(pretrain_out)
        x = layers.Activation(activation="relu")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.4)(x)
        x = layers.Dense(num_classes)(x)
        outputs = layers.Activation(activation="softmax", dtype=tf.float32, name='activationLayer')(x)
        return Model(inputs=inputs, outputs=outputs)
    
    
    # Reference: https://github.com/faustomorales/vit-keras
    def build_vision_transformer():
        vit_pretrained_model = vit.vit_b16(
            image_size = 224,
            activation = 'softmax',
            pretrained = True,
            include_top = False,
            pretrained_top = False,
            classes = 200
        )
        
        return tf.keras.Sequential([
            vit_pretrained_model,
            layers.Flatten(),
            layers.BatchNormalization(),
            layers.Dropout(0.2),
            layers.Dense(1024, activation=tfa.activations.gelu),
            layers.Dense(200, activation='softmax')
        ],
        name = 'vision_transformer')
