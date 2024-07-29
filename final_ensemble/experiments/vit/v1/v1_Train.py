import os
import sys
import shutil
import numpy as np

import tensorflow as tf

from sklearn.utils import class_weight

from Config import Config
from Domain import ModelRunMode, ModelType
from Utils import DataGenerator, ModelBuilder

# It is needed to import to import the files, as its not possible to persist the objects imported from the Utils module
# import DataGenerator as DataGeneratorFile
# import ModelBuilder as ModelBuilderFile

def __train_vit(experiment_id):
    print(tf.config.list_physical_devices())
    
    assert type(experiment_id) == str, "[folder_name] The path for the folder must be a string!"
    
    experiment_dir = f'{Config.VIT_EXPERIMENT_BASE_PATH.value}/{experiment_id}'
    
    BATCH_SIZE = 16
    IMAGE_SIZE = (224, 224)

    os.makedirs(f'{experiment_dir}/saved_weights', exist_ok=True)
    shutil.copyfile(__file__, f'{experiment_dir}/{experiment_id}_Train.py')
    shutil.copyfile('./Utils/DataGenerator.py', f'{experiment_dir}/{experiment_id}_DataGenerator.py')
    shutil.copyfile('./Utils/ModelBuilder.py', f'{experiment_dir}/{experiment_id}_ModelBuilder.py')
    shutil.copyfile('./Utils/DataAugmenter.py', f'{experiment_dir}/{experiment_id}_DataAugmenter.py')
    
    csv_logger = tf.keras.callbacks.CSVLogger(f'{experiment_dir}/traininglogs.csv')
    
    data_generator = DataGenerator(mode=ModelRunMode.TRAINING, data_path=Config.DATA_PATH.value)

    train_images, val_images = data_generator.gemerateTrainValDataFor(model_type=ModelType.VIT, 
                                                                      image_size=IMAGE_SIZE,
                                                                      batch_size=BATCH_SIZE)
    
    vision_transformer_model = ModelBuilder.build_vision_transformer()
    
    print("=======================================================================")
    print("Vision Transformer Summary")
    print(vision_transformer_model.summary())
    print("=======================================================================")
    
    vision_transformer_model.compile(
        optimizer=tf.keras.optimizers.legacy.SGD(0.001, momentum=0.35),
        loss = tf.keras.losses.CategoricalCrossentropy(label_smoothing = 0.2), 
        metrics = ['accuracy']
    )
    
    # Class weights to handle class imbalance
    class_weights = class_weight.compute_class_weight(
        class_weight = 'balanced',
        classes = np.unique(train_images.classes),
        y = train_images.classes
    )
    class_weights_dict = {i:w for i,w in enumerate(class_weights)}
    
    return vision_transformer_model.fit(
        train_images,
        steps_per_epoch=len(train_images),
        validation_data=val_images,
        validation_steps=len(val_images),
        epochs=300,
        callbacks=[
            tf.keras.callbacks.ModelCheckpoint(                
                filepath=f'{experiment_dir}/saved_weights/' + '{epoch:02d}_{val_loss:.2f}.h5',
                save_weights_only=True,
                monitor='val_loss',
                mode='min',
                save_best_only=False
                ),
            tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss',
                                                 factor=0.6, patience=5,
                                                 mode='min'),
            tf.keras.callbacks.EarlyStopping(monitor = "val_loss",
                                             patience = 20,
                                             restore_best_weights = True, 
                                             mode='min'),
            csv_logger
            ],
        class_weight=class_weights_dict
    )
    
    
    
def __train_efficientnet_b2(experiment_id):
    print(tf.config.list_physical_devices())
    
    assert type(experiment_id) == str, "[folder_name] The path for the folder must be a string!"
    
    experiment_dir = f'{Config.EFF_NET_EXPERIMENT_BASE_PATH.value}/{experiment_id}'
    
    BATCH_SIZE = 32
    IMAGE_SIZE = (224, 224)

    os.makedirs(f'{experiment_dir}/saved_weights', exist_ok=True)
    shutil.copyfile(__file__, f'{experiment_dir}/{experiment_id}_Train.py')
    shutil.copyfile('./Utils/DataGenerator.py', f'{experiment_dir}/{experiment_id}_DataGenerator.py')
    shutil.copyfile('./Utils/ModelBuilder.py', f'{experiment_dir}/{experiment_id}_ModelBuilder.py')
    shutil.copyfile('./Utils/DataAugmenter.py', f'{experiment_dir}/{experiment_id}_DataAugmenter.py')
    
    csv_logger = tf.keras.callbacks.CSVLogger(f'{experiment_dir}/traininglogs.csv')
    
    data_generator = DataGenerator(mode=ModelRunMode.TRAINING, data_path=Config.DATA_PATH.value)

    train_images, val_images = data_generator.gemerateTrainValDataFor(model_type=ModelType.EFF_NET_B2, 
                                                                      image_size=IMAGE_SIZE, 
                                                                      batch_size=BATCH_SIZE)
    
    
    efficientnet_b2_model = ModelBuilder.build_efficientnet()
    
    efficientnet_b2_model.compile(
        optimizer=tf.keras.optimizers.legacy.Adam(0.0001),
        loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False),
        metrics=['accuracy']
    )
    
    class_weights = class_weight.compute_class_weight(
        class_weight = 'balanced',
        classes = np.unique(train_images.classes),
        y = train_images.classes
    )
    class_weights_dict = { i:w for i,w in enumerate(class_weights) }
    
    efficientnet_b2_model.fit(
        train_images,
        steps_per_epoch=len(train_images),
        validation_data=val_images,
        validation_steps=len(val_images),
        epochs=300,
        callbacks=[tf.keras.callbacks.ModelCheckpoint(
                filepath=f'{experiment_dir}/saved_weights/' + '{epoch:02d}_{val_loss:.2f}.h5',
                save_weights_only=True,
                monitor='val_loss',
                mode='min',
                save_best_only=True),
                tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss',
                                                    factor=0.2,
                                                    patience=10,
                                                    mode='min'),
                tf.keras.callbacks.EarlyStopping(monitor = "val_loss", 
                                                 patience = 100,
                                                 restore_best_weights = True), 
                csv_logger
            ],
            class_weight=class_weights_dict
    )
    
    
    
    
if __name__ == '__main__':
    if sys.argv[1] == "vit":
         __train_vit(experiment_id = sys.argv[2])
    elif sys.argv[1] == "effnet":
        __train_efficientnet_b2(experiment_id = sys.argv[2])
    else:
        raise ValueError(f'Invalid model type: {sys.argv[1]}')
   