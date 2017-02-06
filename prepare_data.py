__author__ = 'Zeynab'
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from biosppy.signals.tools import smoother

#set params
look_back = 200
horizon = 5
min_range = -50
max_range = 50


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
