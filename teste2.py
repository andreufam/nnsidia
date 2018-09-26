import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf

from keras.layers import Conv1D, Input, Add, Activation, Dropout
from keras.models import Sequential, Model
from keras.regularizers import l2
from keras.initializers import TruncatedNormal
from keras.layers.advanced_activations import LeakyReLU, ELU
from keras import optimizers

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)


def DC_CNN_Block(nb_filter, filter_length, dilation, l2_layer_reg):
    def block(block_input):        
        residual =    block_input
        
        layer_out =   Conv1D(filters=nb_filter, kernel_size=filter_length, 
                      dilation_rate=dilation, 
                      activation='linear', padding='causal', use_bias=False,
                      kernel_initializer=TruncatedNormal(mean=0.0, stddev=0.05, 
                      seed=42), kernel_regularizer=l2(l2_layer_reg))(block_input)                    
        selu_out =    Activation('selu')(layer_out)
        
        skip_out =    Conv1D(1,1, activation='linear', use_bias=False, 
                      kernel_initializer=TruncatedNormal(mean=0.0, stddev=0.05, 
                      seed=42), kernel_regularizer=l2(l2_layer_reg))(selu_out)
        
        c1x1_out =    Conv1D(1,1, activation='linear', use_bias=False, 
                      kernel_initializer=TruncatedNormal(mean=0.0, stddev=0.05, 
                      seed=42), kernel_regularizer=l2(l2_layer_reg))(selu_out)
                      
        block_out =   Add()([residual, c1x1_out])
        
        return block_out, skip_out
    return block

def DC_CNN_Model(length):
    visible = Input(shape=(length, 1))
    
    l1a, l1b = DC_CNN_Block(32,2,1,0.001)(visible)    
    l2a, l2b = DC_CNN_Block(32,2,2,0.001)(l1a) 
    l3a, l3b = DC_CNN_Block(32,2,4,0.001)(l2a)
    l4a, l4b = DC_CNN_Block(32,2,8,0.001)(l3a)
    l5a, l5b = DC_CNN_Block(32,2,16,0.001)(l4a)
    l6a, l6b = DC_CNN_Block(32,2,32,0.001)(l5a)
    l7a, l7b = DC_CNN_Block(32,2,64,0.001)(l6a)

    l8 =   Add()([l1b, l2b, l3b, l4b, l5b, l6b, l7b])  
    
    l9 =   Activation('relu')(l8)
           
    yhat =  Conv1D(1,1, activation='linear', use_bias=False, 
            kernel_initializer=TruncatedNormal(mean=0.0, stddev=0.05, seed=42),
            kernel_regularizer=l2(0.001))(l9)

    model = Model(inputs=visible, outputs=yhat)
    
    adam = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, decay=0.0)

    model.compile(loss='mae', optimizer=adam, metrics=['mse'])
    return model

import pandas as pd
airline = pd.read_csv('data/airline.csv')
airline['passengers'].plot()
forecast_horizon = 40
Xpast = airline['passengers'].iloc[:-forecast_horizon]
Xreal = airline['passengers'].iloc[-forecast_horizon:] 

Xpast = np.atleast_2d(np.asarray(Xpast)).T
length = len(Xpast)-1
model = DC_CNN_Model(length)
X = Xpast[:-1].reshape(1,length,1)
y = Xpast[1:].reshape(1,length,1)  
model.fit(X, y, epochs=500)