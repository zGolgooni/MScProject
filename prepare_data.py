__author__ = 'ZG'
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from settings import look_back


#function to load data -> number of look back pints = 100
def load_data(data, n_prev=look_back):
    doc_x, doc_y = [], []
    for i in range(len(data)-n_prev):
        doc_x.append(data.iloc[i:i+n_prev].as_matrix())
        doc_y.append(data.iloc[i+n_prev].as_matrix())
    als_x = np.array(doc_x)
    als_y = np.array(doc_y)

    return als_x, als_y


#Normalize data in specified range
def normalize_data(dataset, max_range=100, min_range=0):
    scaler = MinMaxScaler(feature_range=(min_range, max_range))
    data = scaler.fit_transform(dataset)
    return data
