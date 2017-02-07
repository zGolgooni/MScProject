__author__ = 'Zeynab'
import numpy as np
import pandas
from sklearn.preprocessing import MinMaxScaler
from biosppy.signals.tools import smoother

#set params
look_back = 500
horizon = 5
min_range = -50
max_range = 50
total_length = 60000


def read_sample(path, name):
    #print('1')
    dataset = pandas.read_csv(path + name + '.txt', delimiter='\t', skiprows=4)
    #print('2')
    x_signal = dataset.values[:total_length, 0]
    y_signal = dataset.values[:total_length, 1]
    y_signal = normalize_data(pandas.DataFrame(y_signal), max_range, min_range)
    return dataset, x_signal, y_signal


#function to load data -> number of look back pints = 100
def load_data(data, n_prev=look_back, n_next=horizon):
    doc_x = []
    doc_y = []
    for i in range(len(data)-n_prev-horizon):
        doc_x.append(data.iloc[i:i+n_prev].as_matrix())
        doc_y.append(data.iloc[i+n_prev:i+n_prev+n_next].as_matrix())
    als_x = np.array(doc_x)
    als_y = np.array(doc_y)
    return als_x, als_y


#Normalize data in specified range
def normalize_data(dataset, max=max_range, min=min_range):
    scaler = MinMaxScaler(feature_range=(min, max))
    data = scaler.fit_transform(dataset)
    #move to fit baseline to zero
    index = np.where(dataset == 0)
    value = data[index[0][0]]
    data = data-value
    return data


def prepare_for_lstm(signal):
    sample_x, sample_y = load_data(pandas.DataFrame(signal), look_back, horizon)
    sample_x_reshaped = np.reshape(sample_x, (sample_x.shape[0], 1, sample_x.shape[1]))
    return sample_x_reshaped, sample_y