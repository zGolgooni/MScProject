from keras.models import Sequential
from keras.layers.core import Dense
from keras.layers import LSTM
from keras.layers.core import Dense, Dropout, Activation
import pandas
from prepare_data import normalize_data, load_data, look_back, horizon, min_range, max_range
import numpy as np

num_hidden_nodes = 100

def create_model(hidden_nodes=num_hidden_nodes, input_size=look_back, output_size=horizon):
    # create model -> here a simple LSTM
    model = Sequential()
    model.add(LSTM(hidden_nodes, input_dim=input_size, return_sequences=False))
    model.add(Dropout(0.5))
    model.add(Dense(output_size))
    model.add(Activation('linear'))
    model.compile(loss="mean_squared_error", optimizer="rmsprop")
    return model


def fit_model(name,model,train_x,train_y, batch=10000, epoch=50, validation=0.2):
    model.fit(train_x, train_y, batch_size=batch, nb_epoch=epoch, validation_split=validation)
    print("OK! lets check it!!")
    model.save_weights(name+'.h5')
    return model


def load_model(name,model):
    model.load_weights(name)
    return model
