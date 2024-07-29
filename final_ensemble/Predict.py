
import os
import sys
import numpy as np
import pandas as pd

import tensorflow as tf

from Config import Config
from Utils import ModelBuilder, DataGenerator
from Domain import ModelRunMode, ModelType

def __predict(prediction_id):
    efficientnet_b2_model = ModelBuilder.build_efficientnet()
    efficientnet_b2_model.compile()
    efficientnet_b2_model.load_weights(Config.EFF_NET_B2_WEIGHT_PATH.value)
    
    vision_transformer_model = ModelBuilder.build_vision_transformer()
    vision_transformer_model.compile()            
    vision_transformer_model.load_weights(Config.VIT_WEIGHT_PATH.value)
    
    print("=======================================================================")
    print("EfficientNet B2 Summary")
    print(efficientnet_b2_model.summary())
    print("=======================================================================")
    
    print("=======================================================================")
    print("Vision Transformer Summary")
    print(vision_transformer_model.summary())
    print("=======================================================================")
    
    batch_size = 32
    image_size = (224,224)
    data_generator = DataGenerator(mode=ModelRunMode.TEST, data_path=Config.DATA_PATH.value)
    
    efficientnet_b2_test_images = data_generator.generateTestDataFor(model_type= ModelType.EFF_NET_B2,
                                                                     image_size=image_size,
                                                                     batch_size=batch_size)
    
    vision_transformer_test_images = data_generator.generateTestDataFor(model_type=ModelType.VIT,
                                                                        image_size=image_size,
                                                                        batch_size=batch_size)

    
    efficientnet_predictions_predictions = efficientnet_b2_model.predict(efficientnet_b2_test_images)
    vision_transformer_predictions_predictions = vision_transformer_model.predict(vision_transformer_test_images)
    
    combined_predictions = efficientnet_predictions_predictions + (vision_transformer_predictions_predictions * 3)
    
    combined_prediction_indices = np.argmax(combined_predictions, axis=1)
    
    os.makedirs(f'predictions', exist_ok=True)
    
    df = pd.DataFrame({'id': np.arange(len(combined_prediction_indices)) + 1, 'label': combined_prediction_indices + 1})
    df.set_index('id', inplace=True)
    prediction_file_name = f'{prediction_id}_{Config.PREDICTION_FILE_POSTFIX.value}'
    path = os.path.join(Config.PREDICTION_BASE_PATH.value, prediction_file_name)
    df.to_csv(path)

    
if __name__ == '__main__':
    __predict(sys.argv[1])