import tensorflow as tf
import keras
from keras.layers import Conv2D, MaxPooling2D, Input, Reshape,BatchNormalization,LeakyReLU,Dense
from keras import Model
import numpy as np
from keras import regularizers, initializers
from keras.layers import Permute, Lambda, add
import keras.backend as K

def conv_batch_lrelu(input_tensor, numfilter, dim, strides=1):
    input_tensor = Conv2D(numfilter, (dim, dim), strides=strides, padding='same',
                        kernel_regularizer=regularizers.l2(0.0005),
                        kernel_initializer=initializers.TruncatedNormal(stddev=0.1),
                        use_bias=False
                    )(input_tensor)
    input_tensor = BatchNormalization()(input_tensor)
    return LeakyReLU(alpha=0.1)(input_tensor)


class TinyYOLOv2:
    def __init__(self, image_size,is_learning_phase=False):
        K.set_learning_phase(int(is_learning_phase))
        K.reset_uids()

        self.image_size = image_size
        self.n_cells = self.image_size // 32

        self.m = self.buildModel()

    def loadWeightsFromDarknet(self, file_path):
        load_weights(self.m, file_path)

    def loadWeightsFromKeras(self, file_path):
        self.m.load_weights(file_path)

    def buildModel(self):
        model_in = Input((self.image_size, self.image_size, 3))
        
        model = model_in
        for i in range(0, 5):
            model = conv_batch_lrelu(model, 16 * 2**i, 3)
            model = MaxPooling2D(2, padding='valid')(model)

        model = conv_batch_lrelu(model, 512, 3)
        model = MaxPooling2D(2, 1, padding='same')(model)
        
        model = conv_batch_lrelu(model, 1024, 3)
        
#         model = Dense(1024)(model)
        
#         model = Dense(1024)(model)        
                
        model = Conv2D(5, (1, 1), padding='same', activation='linear')(model)

#         model_out = Reshape(
#             [self.n_cells, self.n_cells,4 + 1 ]
#             )(model)

        return Model(inputs=model_in, outputs=model)



