__author__ = 'ZG'
import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import math
from settings import look_back, dimension1
import plotly.plotly as py
import plotly.graph_objs as go
#import PrepareData


def create_model(train_x, train_y):
    model = Sequential()
    model.add(LSTM(dimension1, input_dim=look_back))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(train_x, train_y, nb_epoch=2, batch_size=2000)#, verbose=2)
    return model


def evaluate_model(model, dataset, test1_x, test1_y, test2_x, test2_y):
    evaluate_by_score(model, dataset, test1_x, test1_y)
    evaluate_by_score(model, dataset, test2_x, test2_y)
    evaluate_by_plot(model, dataset, test1_x, test2_x)
    return


def evaluate_by_score(model, dataset, test_x, test_y):
    score = model.evaluate(test_x, test_y, verbose=0)
    score = math.sqrt(score)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit_transform(dataset)
    score = scaler.inverse_transform(np.array([[score]]))
    print('Score: %.2f RMSE' % (score))
    return


def evaluate_by_plot(model, dataset, part1_x, part2_x):
    predict1 = model.predict(part1_x)
    predict2 = model.predict(part2_x)
    # shift predictions for plotting
    predict_plot1 = np.empty_like(dataset)
    predict_plot1[:, :] = np.nan
    predict_plot1[look_back:len(predict1)+look_back, :] = predict1
    # shift test predictions for plotting
    predictPlot2 = np.empty_like(dataset)
    predictPlot2[:, :] = np.nan
    predictPlot2[len(predict1)+(look_back*2)+1:len(dataset)-1, :] = predict2
    #data = PrepareData.normalize_data(dataset)

    trace1 = go.Scatter(y=dataset)
    trace2 = go.Scatter(y=predict_plot1)
    trace3 = go.Scatter(y=predictPlot2)

    py.plot([trace1, trace2, trace3])


    #plotly.offline.iplot([trace1,trace2,trace3])
    return
