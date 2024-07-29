import os
import numpy as np
import pandas as pd

import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator

from sklearn.model_selection import train_test_split

from .DataAugmenter import augmentData
from Domain import ModelType, ModelRunMode

class DataGenerator():
    def __init__(self, mode: ModelRunMode, data_path):
        self.data_path = data_path
        self.mode = mode
        
    def gemerateTrainValDataFor(self, model_type: ModelType, image_size, batch_size):
        if model_type == ModelType.EFF_NET_B2:
            # There is no need to call the built-in preprocessing function of tensorflow,
            # because the preprocessing is already done in the model
            # ref.: https://www.tensorflow.org/api_docs/python/tf/keras/applications/efficientnet/preprocess_input
            training_datagen = ImageDataGenerator(
                rotation_range=20,
                width_shift_range=0.2,
                height_shift_range=0.2,
                shear_range=0.2,
                zoom_range=0.2,
                horizontal_flip=True,
                fill_mode='nearest'
            )
            validation_datagen = ImageDataGenerator()
            
        elif model_type == ModelType.VIT:
            
            training_datagen = ImageDataGenerator(
                rescale = 1./255,
                rotation_range=110,
                width_shift_range=0.2,
                height_shift_range=0.2,
                shear_range=0.15,
                zoom_range=0.6,
                horizontal_flip=True,
                fill_mode='nearest',
                featurewise_center=True,
                featurewise_std_normalization=True,
                preprocessing_function = augmentData,
            )
            
            validation_datagen = ImageDataGenerator(
                rescale = 1./255,
                featurewise_center=True,
                featurewise_std_normalization=True,
            )
        else:
            raise ValueError(f'Invalid model type: {model_type}')
        
        return self.__generateTrainValDataWith(training_datagen, validation_datagen, image_size, batch_size)
        
        
        
    def generateTestDataFor(self, model_type: ModelType, image_size, batch_size):
        if model_type == ModelType.EFF_NET_B2:
            # There is no need to call the built-in preprocessing function of tensorflow,
            # because the preprocessing is already done in the model
            # ref.: https://www.tensorflow.org/api_docs/python/tf/keras/applications/efficientnet/preprocess_input
            generator = ImageDataGenerator()
            
        elif model_type == ModelType.VIT:
            generator = ImageDataGenerator(rescale = 1./255, 
                                       featurewise_center=True, 
                                       featurewise_std_normalization=True)
            
        else:
            raise ValueError(f'Invalid model type: {model_type}')
        
        return self.__generateTestDataWith(generator, image_size, batch_size)
    
    
    
    def __generateTrainValDataWith(self, 
                                   training_generator: ImageDataGenerator,
                                   validation_generator: ImageDataGenerator, 
                                   image_size, batch_size):
        
        bird_type_classes = np.load(os.path.join(self.data_path, "class_names.npy"), allow_pickle=True).item()
        bird_type_classes_swapped = {value: key for key, value in bird_type_classes.items()}
        
        train_img_df = pd.read_csv(os.path.join(self.data_path, "train_images.csv"))
        train_img_df['class_name'] = train_img_df['label'].map(bird_type_classes_swapped) 
        train_img_df['image_path'] = train_img_df['image_path'].apply(lambda x: f'{self.data_path}/{x[1:]}')
        
        # Train-val split -> Highly imbalanced classes: stratify to split every class 80-20
        train_df, val_df = train_test_split(
            train_img_df,  
            train_size= 0.80 , 
            shuffle=True,
            random_state=124,
            stratify=train_img_df['label']
        )
        
        train_images = training_generator.flow_from_dataframe(
            dataframe=train_df,
            x_col='image_path',
            y_col='class_name',
            target_size=image_size,
            color_mode='rgb',
            class_mode='categorical',
            batch_size=batch_size,
            shuffle=True,
            seed=42,
        )
                
        val_images = validation_generator.flow_from_dataframe(
            dataframe=val_df,
            x_col='image_path',
            y_col='class_name',
            target_size=image_size,
            color_mode='rgb',
            class_mode='categorical',
            batch_size=batch_size,
            shuffle=False
        )
        
        return train_images, val_images

    
    
    def __generateTestDataWith(self, generator: ImageDataGenerator, image_size, batch_size):
        
        
        test_images_df = pd.read_csv(os.path.join(self.data_path, "test_images_path.csv"))
        test_images_df['image_path'] = test_images_df['image_path'].apply(lambda x: f'{self.data_path}/{x[1:]}')
        
        return generator.flow_from_dataframe(dataframe=test_images_df,
                                            x_col='image_path',
                                            y_col=None,
                                            target_size=image_size,
                                            color_mode='rgb',
                                            class_mode=None,
                                            batch_size=batch_size,
                                            shuffle=False,
                                            seed=42)