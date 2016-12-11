__author__ = 'Zeynab'
import numpy as np
import pandas
import plotly.plotly as py
import plotly.graph_objs as go
from plotly import tools
import plotly
from prepare_data import normalize_data
from tools.lowpass_filter import lowpass_filter

path1 = '/Users/Zeynab/PycharmProjects/Control/'
path2 = '/Users/Zeynab/PycharmProjects/DATA-Mine/Arrhythmic/'
name = 'SSC2 R2 8 30 Nov iso Arrhythmic'
dataset = pandas.read_csv(path2 + name + '.csv', usecols=[1], engine='python')
data = pandas.DataFrame(normalize_data(dataset, 100))

data = np.array(data)
dataset = np.array(dataset)

print('data length = %d' %(data.__len__()))
trace1 = go.Scatter(y=data[:10000])
trace2 = go.Scatter(y=dataset[:10000])
dataset = go.Data([trace1])

layout = go.Layout(title=name, xaxis={'title': 'time (s)'}, yaxis={'title': 'Voltage (microV)'})

fig = tools.make_subplots(rows=1, cols=2)
fig.append_trace(trace1, 1, 1)
fig.append_trace(trace2, 1, 2)
py.plot(fig, filename=name)

print('finished!!! :)')