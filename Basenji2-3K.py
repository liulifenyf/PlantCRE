import tensorflow as tf 
import numpy as np
from tensorflow.keras.layers import Input, \
                                    BatchNormalization, \
                                    Conv1D, \
                                    Dropout, \
                                    MaxPooling1D
from tensorflow.keras.models import Model,load_model


batch_size = 32
epochs = 100
channels_num = 512
 
def exponential_linspace_int(initial_value, target_value, num_layers):
    factor = (target_value / initial_value) ** (1 / num_layers))
    values = []
    # Calculate and store values
    value = initial_value
    for _ in range(num_layers+1):
        values.append(np.round(value))
        value *= factor
    return values[1:]


class GELU(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(GELU, self).__init__(**kwargs)
    def __call__(self, x):
       # return tf.keras.activations.sigmoid(1.702 * x) * x
        return tf.keras.activations.sigmoid(tf.constant(1.702) * x) * x

def conv_block(x, C=channels_num, W=1, D=1, kernel_initializer='he_normal', l2_scale=0):
    x = Conv1D(
        filters=int(C),
        kernel_size=W,
        #strides=strides,
        padding='same',
        dilation_rate=D,
        kernel_initializer=kernel_initializer,
        kernel_regularizer=tf.keras.regularizers.l2(l2_scale))(x)
    x = BatchNormalization()(x)
    x = GELU()(x)

    return x

def residual_block(input, D=1):
    x = conv_block(input, C=0.5*channels_num, W=3, D=D)
    x = conv_block(x, C=channels_num, W=1, D=1)
    x = Dropout(0.3)(x)
    output = x + input
    return output

def basenji_model(input_shape=[3000, 4], W = 15,L=11):
    inputs = Input(input_shape)
    x = Conv1D(
        filters=int(0.375*channels_num),
        kernel_size=W,
        padding='same',
        activation='exponential',
        dilation_rate=1,
        kernel_initializer='he_normal',
        kernel_regularizer=tf.keras.regularizers.l2(0))(inputs)
    x = BatchNormalization()(x)
    x = GELU()(x)
    x = MaxPooling1D(pool_size= 3 )(x)  
    Ci_steps = exponential_linspace_int(0.5*channels_num, channels_num, 6, divisible_by=1) 
    for Ci in Ci_steps:
        x = conv_block(x=x, C=Ci, W=5, D=1)
        x = MaxPooling1D(pool_size=2)(x)
    
    Di=[1,2,3,4]
    for i in range(len(Di)): 
        x = residual_block(x, D=Di[i]) 


    x = conv_block(x, C=channels_num//2, W=1, D=1)
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
            tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_dir,
                                        monitor='val_loss',
                                        verbose=0,
                                        mode='auto' ,
                                        # save_weights_only=False,
                                        save_best_only= True),
            tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, verbose=0),
            tf.keras.callbacks.LearningRateScheduler(decay)
        ]         


    model = basenji_model(input_shape=(3000,4), W=15,L=3)
    model.compile(loss=tf.keras.losses.MeanSquaredError(),
                    optimizer=tf.keras.optimizers.Adam()
                    ) 
