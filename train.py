from core.model import BBregression
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
import scipy.io
import scipy.misc
import numpy as np
import pandas as pd
from core.Dataloader import Dataloader
import cv2
import random
from keras import backend as K
import math

model = BBregression(480)

df_train = pd.read_csv('./Data/df_train.csv',index_col=0)
df_test = pd.read_csv('./Data/df_test.csv', index_col=0)

data = Dataloader(df=df_train)
labels = tf.placeholder(tf.float32, shape=(None, 4))
generator = data.build_iterator()

losses = tf.losses.mean_squared_error(labels,model.m.output)

train_step = tf.train.AdamOptimizer(0.0001).minimize(losses)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for epochs in range(1):
        for batches in range(3):
            batch = generator.get_next()
            train_data= sess.run(batch)
            _,losses = sess.run([train_step,losses],feed_dict={model.m.input: train_data[0],labels: train_data[1]})
            print "Epoch:",epochs," Losses:",losses
        model.m.save("./saved_models/model_saved")    
# acc_value = accuracy(labels, preds)
# with sess.as_default():
#     print acc_value.eval(feed_dict={img: mnist_data.test.images,
#                                     labels: mnist_data.test.labels,
#                                     K.learning_phase(): 0})

