from keras.models import Sequential
from keras.layers.core import Dense
from keras.layers import LSTM
from keras.layers.core import Dense, Dropout, Activation
import pandas
from prepare_data import normalize_data, load_data, look_back, horizon, min_range, max_range
import numpy as np

def create_model(name, train_x,train_y, hidden_nodes, input, output=horizon, batch=10000, epoch=50, validation=0.2):
    # create model -> here a simple LSTM
    model = Sequential()
    model.add(LSTM(hidden_nodes, input_dim=input, return_sequences=False))
    model.add(Dropout(0.5))
    model.add(Dense(output))
    model.add(Activation('linear'))
    model.compile(loss="mean_squared_error", optimizer="rmsprop")
    model.fit(train_x, train_y, batch_size=batch, nb_epoch=epoch, validation_split=validation)
    print("OK! lets check it!!")
    model.save_weights(name+'.h5')
    return model


def predict_sample(path, name, model, total_length):
    dataset = pandas.read_csv(path + name + '.txt', delimiter='\t', skiprows=4)
    x_signal = dataset.values[:total_length, 0]
    y_signal = dataset.values[:total_length, 1]
    predicted_signal = np.zeros(total_length)
    num_avg = np.zeros(total_length)

    dataset = pandas.read_csv(path + name + '.txt', delimiter='\t', skiprows=4)
    x_signal = dataset.values[:, 0]
    y_signal = dataset.values[:, 1]
    normalized_signal = normalize_data(pandas.DataFrame(y_signal), max_range, min_range)
    #normalized_signal, params = smoother(normalized_signal[:, 0])
    sample_x, sample_y = load_data(pandas.DataFrame(normalized_signal), look_back)

    for i in range(0, total_length - look_back):
        xx = sample_x[i]
        yy = sample_y[i]
        xx_reshaped = np.reshape(xx, (xx.shape[0], 1, xx.shape[1]))
        predicted = model.predict(xx_reshaped)
