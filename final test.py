__author__ = 'Zeynab'
import csv,pandas
import numpy as np
import numpy as np
from biosppy.signals.tools import smoother,filter_signal
import matplotlib.pyplot as plt
import plotly.plotly as py
import plotly.graph_objs as go
from plotly import tools
from sklearn.preprocessing import MinMaxScaler


#set params
look_back = 200
min_range = -100
max_range = 100


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
    #move to fit baseline to zero
    index = np.where(dataset == 0)
    value = data[index[0][0]]
    data = data-value
    return data


with open('/Users/Zeynab/Desktop/before 95.08.csv') as csvfile:
    readCSV = csv.reader(csvfile)
    paths = []
    names = []
    sampling_rates = []
    labels = []
    next(readCSV)
    for row in readCSV:
        path = row[0]
        name = row[1]
        sampling_rate = row[2]
        label = row[3]

        paths.append(path)
        names.append(name)
        sampling_rates.append(sampling_rate)
        labels.append(label)

    print(paths)
    print(names)
    print(sampling_rates)


for i in range(len(names) +1):
    name = names[i]
    dataset = np.loadtxt(paths[i] + names[i] + '.txt', skiprows=1)
    sampling_rate = sampling_rates[i]
    x_signal = dataset[:, 0]
    y_signal = dataset[:, 1]
    normalized_signal = normalize_data(pandas.DataFrame(y_signal), max_range, min_range)
    """
    smoothed_signal, params = smoother(normalized_signal)

    trace1 = go.Scatter(y=y_signal[:10000], x=x_signal[:10000], name='Signal')
    trace2 = go.Scatter(y=smoothed_signal[:10000], x=x_signal[:10000], name='smoothed')
    trace3 = go.Scatter(y=normalized_signal[:10000], x=x_signal[:10000], name='smoothed')
    layout = go.Layout(title=name)
    figure = go.Figure(data=[trace1, trace2, trace3], layout=layout)
    py.plot(figure, filename='biosppy test of samples ' + name)
    """
    sample_x, sample_y = load_data(pandas.DataFrame(normalized_signal), look_back)
    sample_x_reshaped = np.reshape(sample_x, (sample_x.shape[0], 1, sample_x.shape[1]))
    if i == 0:
        train_x_reshaped = sample_x_reshaped
        train_y = sample_y
    else:
        train_x_reshaped = np.concatenate([train_x_reshaped, sample_x_reshaped])
        train_y = np.concatenate([train_y, sample_y])
    print("%d ----->%s,  sampling rate=%s" % ((i + 1),name,sampling_rate))


print('train sample is ready! :)')