
# %%
import os
from scipy.stats.morestats import ppcc_max
from scipy.stats.stats import pearsonr
import tensorflow as tf 
import numpy as np
from tensorflow.keras.layers import Input, \
                                    BatchNormalization, \
                                    Conv1D, \
                                    Dropout, \
                                    MaxPooling1D
from tensorflow.keras.models import Model

batch_size = 16
epochs = 100
channels_num = 720

class GELU(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(GELU, self).__init__(**kwargs)
    def __call__(self, x):
       # return tf.keras.activations.sigmoid(1.702 * x) * x
        return tf.keras.activations.sigmoid(tf.constant(1.702) * x) * x


def exponential_linspace_int(start, end, num, divisible_by=24):
    """Exponentially increasing values of integers."""
    def _round(x):
        return int(np.round(x / divisible_by) * divisible_by)

    base = np.exp(np.log(end / start) / (num - 1))
    return [_round(start * base**i) for i in range(num)]


def conv_block(x, C=channels_num, W=1, D=1, kernel_initializer='he_normal', l2_scale=0):
    x = BatchNormalization()(x)
    x = GELU()(x)
    x = Conv1D(
        filters=int(C),
        kernel_size=W,
        #strides=strides,
        padding='same',
        dilation_rate=D,
        kernel_initializer=kernel_initializer,
        kernel_regularizer=tf.keras.regularizers.l2(l2_scale))(x)
    return x

def residual_block(input, D=1):
    x = conv_block(input, C=0.5*channels_num, W=3, D=D)
    x = conv_block(x, C=channels_num, W=1, D=1)
    x = Dropout(0.3)(x)
    output = x + input
    return output

def basenji_model(input_shape=[120000, 4], L=11):
    inputs = Input(input_shape)
    x = Conv1D(
        filters=int(0.375*channels_num),
        kernel_size=15,
        padding='same',
        dilation_rate=1,
        kernel_initializer='he_normal',
        kernel_regularizer=tf.keras.regularizers.l2(0))(inputs)
    
    x = MaxPooling1D(pool_size= 3 )(x)  #50k
    Ci_steps = exponential_linspace_int(0.5*channels_num, channels_num, 6, divisible_by=24)
    for Ci in Ci_steps:
        x = conv_block(x=x, C=Ci, W=5, D=1)
        x = MaxPooling1D(pool_size=2)(x)
    
    Di=1
    for i in range(L):
        x = residual_block(x, D=Di)
        Di=Di*1.5
        Di=int(np.round(Di))

    x = conv_block(x, C=2*channels_num, W=1, D=1)
    x = Dropout(0.05)(x)
    x = GELU()(x)
    x = Conv1D(filters = 1,
               kernel_size=1, 
               padding='same')(x)
    x = tf.keras.layers.Flatten()(x)
    x= tf.keras.layers.Dense(1)(x)
    model = Model(inputs = inputs, outputs = x)
    return model


def decay(epoch):
  if epoch < 3:
    return 1e-3
  else:
    return 1e-5


for r in range(3):
    callbacks = [
            # tf.keras.callbacks.TensorBoard(log_dir='/home/yangq/Basenji2-my/basenji-master/model/logs{0}'.format(r)),
            tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_dir,
                                        monitor='val_loss',
                                        verbose=0,
                                        mode='auto' ,
                                        save_weights_only=True,
                                        save_best_only= True),
            tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=8, verbose=0),
            tf.keras.callbacks.LearningRateScheduler(decay)
        ]         


    model = basenji_model(input_shape=(120000,4))
    model.compile(loss=tf.keras.losses.MeanSquaredError(),
                    optimizer=tf.keras.optimizers.Adam()
                    ) 

        