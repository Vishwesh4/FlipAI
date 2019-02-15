import tensorflow as tf
import tensorflow.keras
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Input, Reshape,BatchNormalization,LeakyReLU,Dense,Activation,Flatten
from tensorflow.keras import Model
import numpy as np
from tensorflow.keras import regularizers, initializers
from tensorflow.keras.layers import Permute, Lambda, add
import tensorflow.keras.backend as K
from tensorflow.keras import losses
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import datetime as dt
import os

def conv_batch_lrelu(input_tensor, numfilter, dim, strides=1):
    input_tensor = Conv2D(numfilter, (dim, dim), strides=strides, padding='same',
                        kernel_regularizer=regularizers.l2(0.0005),
                        kernel_initializer=initializers.TruncatedNormal(stddev=0.1),
                        use_bias=False
                    )(input_tensor)
    input_tensor = BatchNormalization()(input_tensor)
    return LeakyReLU(alpha=0.1)(input_tensor)

class Timer():

    def __init__(self):
        self.start_dt = None

    def start(self):
        self.start_dt = dt.datetime.now()

    def stop(self):
        end_dt = dt.datetime.now()
        print('Time taken: %s' % (end_dt - self.start_dt))

class BBregression():
    def __init__(self, image_size,is_learning_phase=False):
        K.set_learning_phase(int(is_learning_phase))
        K.reset_uids()

        self.image_size = image_size
        self.n_cells = self.image_size // 32

        self.m = self.buildModel()
        self.model_compile()
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

        model = conv_batch_lrelu(model, 256, 3)
        model = MaxPooling2D(2, 1, padding='same')(model) 
        # model = conv_batch_lrelu(model, 512, 3)        
        model = Flatten()(model)
        model = Dense(4,activation='sigmoid')(model)

        return Model(inputs=model_in, outputs=model)

    def model_compile(self):
        self.m.compile(loss= 'mse', optimizer=Adam(lr=0.001),metrics =['acc'])
        self.m.summary()
        print('[Model] Model Compiled')

    def train_generator(self, data_gen, epochs, batch_size, steps_per_epoch, save_dir):
        timer = Timer()
        timer.start()
        print('[Model] Training Started')
        print('[Model] %s epochs, %s batch size, %s batches per epoch' % (epochs, batch_size, steps_per_epoch))
            
        save_fname = os.path.join(save_dir, '%s-e%s.h5' % (dt.datetime.now().strftime('%d%m%Y-%H%M%S'), str(epochs)))
        callbacks = [
            ModelCheckpoint(filepath=save_fname, monitor='loss', save_best_only=True)
        ]
        self.m.fit(
            data_gen,
            steps_per_epoch=steps_per_epoch,
            epochs=epochs,
            callbacks=callbacks,verbose=1
        )
        
        print('[Model] Training Completed. Model saved as %s' % save_fname)
        timer.stop()