__author__ = 'ZG'
import numpy as np
import pandas
from keras.layers import LSTM
from keras.layers.core import Dense, Dropout, Activation
from keras.models import Sequential
import plotly.plotly as py
import plotly.graph_objs as go
from prepare_data import load_data, normalize_data
import settings, simple_LSTM
#import matplotlib.pyplot as plt
#from fastdtw.fastdtw import dtw
#from BlandAltman import bland_altman_plot
#function to load data -> number of look back pints = 100


look_back = settings.look_back
min_range = settings.min_range
max_range = settings.max_range


#create model -> here a simple LSTM
model = Sequential()
model.add(LSTM(300, input_dim=look_back, return_sequences=False))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('linear'))
model.compile(loss="mean_squared_error", optimizer="rmsprop")


#load train data
# ~50500 points from each file
path = '/Users/Zeynab/PycharmProjects/Control/'
normal_train = ['SSC1-Nif 95.7.3 0nM Control','Control Bam QTC_col1','ES-CM N1','ES-CM Nr50001','SSC2 R1 4 29 Nov iso Control','SSC2-iso 95.6.30 0nM Ch65 Control','Control Bam R09_mine2','SSC2 R2 7 30 Nov baseline Control','SSC2 R2 DIV8 R2 6 16 Nov baseline Control','iPS-CM Nr.1','SSC1-Nif 95.7.10 10nM Control','ES-CM N20001','ES-CM Nr50002']
not_trained = ['ES-CM Nr5','SSC2 R1 DIV8 R2 1 12 Nov Control','SSC2-Nif 95.6.30 0nM Ch26 Control','SSC R2 19 Oct baseline Control','SSC1-iso 95.7.3 0nM Control']


train_x_reshaped = np.empty([0, 1, look_back])
train_y = np.empty([0, 1])

for i, filename in enumerate(normal_train):
    dataset = pandas.read_csv(path + filename + '.csv', usecols=[1], engine='python')
    dataset = pandas.DataFrame(normalize_data(dataset, max_range, min_range))
    sample_x, sample_y = load_data(dataset[:50500], look_back)
    sample_x_reshaped = np.reshape(sample_x, (sample_x.shape[0], 1, sample_x.shape[1]))
    if i == 0:
        train_x_reshaped = sample_x_reshaped
        train_y = sample_y
    else:
        train_x_reshaped = np.concatenate([train_x_reshaped, sample_x_reshaped])
        train_y = np.concatenate([train_y, sample_y])
    print("%d ----->" % (i + 1) + filename)

print("train data is ready")


# fit model by train data
model.fit(train_x_reshaped, train_y, batch_size=300, nb_epoch=80, validation_split=0.15)
print("OK! lets check it!!")

model.save_weights('my_model.h5')

#test model by some samples
path1 = '/Users/Zeynab/PycharmProjects/test data/'
path2 = '/Users/Zeynab/PycharmProjects/DATA-Mine/Arrhythmic/'
some_test_samples = ['CPVT1 Nr1 RA52 95.4.8 Baseline Arrhythmic', 'SSC2 R2 4 29 Nov baseline Arrhythmic', 'SSC R2 19 Oct iso Arrhythmic','SSC1 R3 Nif 95.7.12 baseline Arrythmic','iPS-CM Nr.1','ES-CM N20001','Control Bam R09', 'SSC1-Nif 95.7.10 10nM Control', 'SSC R2 19 Oct baseline Control']
arrhythmic_samples = ['CPVT1 Nr1 RA52 95.4.8 Baseline Arrhythmic','CPVT1 Nr1 RA52 95.4.8 iso Arrhythmic','SSC R2 2 19 Oct baseline Arrhythmic','SSC R2 19 Oct iso Arrhythmic', 'SSC1 R3 Nif 100 nM 95.7.12 baseline Arrythmic', 'SSC1-iso 95.7.8 100nM Ch74 Arrhythmic', 'SSC2 R2 4 29 Nov baseline Arrhythmic','SSC2 R2 8 30 Nov iso Arrhythmic', 'SSC2 R2 10 30 Nov baseline Arrhythmic', 'SSC2 R2 10 30 Nov iso Arrhythmic', 'SSC2-Nif 95.6.30 0nM Ch52 Arrhythmic']

for i,name in enumerate(arrhythmic_samples):
    dataset = pandas.read_csv(path2+name+'.csv', usecols=[1], engine='python')
    dataset = pandas.DataFrame(normalize_data(dataset,100))
    x,y = load_data(dataset[:59000])
    xx = np.reshape(x, (x.shape[0], 1, x.shape[1]))

    predicted1 = model.predict(xx)
    rmse = np.sqrt(((predicted1 - y) ** 2).mean(axis=0))

    print(name+"------> rmse1 = %d" %rmse)
    #distance, p= dtw(y, predicted1)
    #print(distance)
    #bland_altman_plot(y,predicted1,name)

    trace1 = go.Scatter(y=y[:10000], name='Real signal')
    trace2 = go.Scatter(y=predicted1[:10000], name='Predicted by model')
    layout = go.Layout(title=name)
    figure = go.Figure(data=[trace1, trace2], layout=layout)
    py.plot(figure, filename='model1_' + name)


#single sample test
name='SSC2 R2 10 30 Nov iso Arrhythmic'
dataset = pandas.read_csv(path+ name + '.csv', usecols=[1], engine='python')
dataset = pandas.DataFrame(normalize_data(dataset,100))
x,y = load_data(dataset[:59000])
xx = np.reshape(x, (x.shape[0], 1, x.shape[1]))

predicted1 = model.predict(xx)
rmse = np.sqrt(((predicted1 - y) ** 2).mean(axis=0))

print(name+"------> rmse1 = %d" %rmse)
    #distance, p= dtw(y, predicted1)
    #print(distance)
    #bland_altman_plot(y,predicted1,name)

trace1 = go.Scatter(y=y[:7000], name='Real signal')
trace2 = go.Scatter(y=predicted1[:7000], name='Predicted by model')
layout = go.Layout(title=name)
figure = go.Figure(data=[trace1, trace2], layout=layout)
py.plot(figure, filename='model1_' + name)
