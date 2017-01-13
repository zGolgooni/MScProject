__author__ = 'Zeynab'
import numpy as np
from biosppy.signals.tools import smoother,filter_signal
import matplotlib.pyplot as plt
import plotly.plotly as py
import plotly.graph_objs as go
from plotly import tools


path1 = '/Users/Zeynab/PycharmProjects/Control/'
name='ES-CM N1'
sampling_rate = 2000
signal = np.loadtxt(path1+name+'.txt',skiprows=1)
x_signal = signal[:, 0]
y_signal = signal[:, 1]
smoothed_signal,params = smoother(y_signal)
filter_order = 2
filtered_signal, s_rate,params2 = filter_signal(signal=y_signal, ftype='FIR', band='lowpass', order=filter_order, frequency=sampling_rate/4, sampling_rate=sampling_rate)

smoothed2_signal,params = smoother(filtered_signal)

trace1 = go.Scatter(y=y_signal[:10000], x=x_signal[:10000], name='Signal')

trace2 = go.Scatter(y=smoothed_signal[:10000], x=x_signal[:10000], name='smoothed')
trace3 = go.Scatter(y=filtered_signal[:10000], x=x_signal[:10000], name='filtered')
trace4 = go.Scatter(y=smoothed2_signal[:10000], x=x_signal[:10000], name='smoothed after filtering')


layout = go.Layout(title=name)
figure = go.Figure(data=[trace1, trace2, trace3,trace4], layout=layout)
py.plot(figure, filename='biosppy test of r peaks detection ' + name)


"""
plt.plot(x_signal[:10000], y_signal[:10000])
plt.plot(x_signal[:10000],smoothed_signal[:10000])
plt.plot(x_signal[:10000],filtered_signal[:10000])

plt.show()
"""
