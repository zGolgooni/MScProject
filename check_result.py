__author__ = 'Zeynab'
import pandas
import numpy as np
import plotly.plotly as py
import plotly.graph_objs as go
from plotly import tools
from scipy.stats import norm
from prepare_data import read_sample, normalize_data, load_data, look_back, horizon, min_range, max_range, total_length
from lstm_model import create_model


def predict_signal(path, name, model):
    dataset, x_signal, y_signal = read_sample(path, name)
    sample_x, sample_y = load_data(pandas.DataFrame(y_signal), look_back)
    sample_x_reshaped = np.reshape(sample_x, (sample_x.shape[0], 1, sample_x.shape[1]))
    predicted = model.predict(sample_x_reshaped)
    predicted_signal = np.zeros(total_length)

    for i in range(0, total_length):
        counter = 1
        temp = predicted[i,0]
        #print('Check: i = %d',i)
        for j in range(1,5):
            if (i - j) >= 0:
                #print('if loop: predicted[%d,%d]', (i-j), j)
                temp += predicted[(i-j), j]
                counter += 1
        if counter > 5:
            print('error: more than 5 sample counted! at predicted[%d], counter = %d',i,counter)
        predicted_signal[i] = temp/counter
    return predicted_signal


def check_rmse(path, name, model):
    dataset, x_signal, y_signal = read_sample(path, name)
    predicted_signal = predict_signal(dataset, model, total_length)
    rmse = np.sqrt(((predicted_signal - y_signal[look_back:total_length+look_back]) ** 2).mean(axis=0))
    return predicted_signal, rmse


def check_label(path, name, model, normal_mu, normal_std, arrhythmic_mu, arrhythmic_std):
    dataset, x_signal, y_signal = read_sample(path, name)
    predicted_signal, rmse = check_rmse(path, name, model)

    p_normal = norm.pdf(rmse, normal_mu, normal_std)
    p_arrhythmic = norm.pdf(rmse, arrhythmic_mu, arrhythmic_std)

    if p_normal > p_arrhythmic:
        predicted_label = 'Normal'
    else:
        predicted_label = 'Arrhythmic'

    print("(%s), rmse = %d, predicted label = %s" % (name, rmse, predicted_label))
    return predicted_signal, rmse, predicted_label


def check_plot(path, name, real_label, model):
    dataset, x_signal, y_signal = read_sample(path, name)
    predicted_signal, rmse = check_rmse(path, name, model)
    print('%s: rmse=%f, real label = %s, predicted label=%s',name,rmse,real_label)
    plot_size = 6000
    trace1 = go.Scatter(y=y_signal[:plot_size], x=x_signal[:plot_size], name='Real signal')
    trace2 = go.Scatter(y=predicted_signal[plot_size], x=x_signal[plot_size], name='Predicted by model')
    layout = go.Layout(title=name)
    figure = go.Figure(data=[trace1, trace2], layout=layout)
    py.plot(figure, filename=name)
    return predicted_signal, rmse



#normalized_signal, params = smoother(normalized_signal[:, 0])
#pre_rmse = np.sqrt(((predicted - sample_y[:,:,0]) ** 2).mean(axis=0))
#rmse = pre_rmse.mean()