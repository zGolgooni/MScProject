import csv,pandas
import numpy as np
import plotly.plotly as py
import plotly.graph_objs as go
from plotly import tools
from scipy.stats import norm
from keras.models import Sequential
from prepare_data import normalize_data, load_data, look_back, min_range, max_range
from lstm_model import create_model
from biosppy.signals.tools import smoother

main_path = '/home/mll/Golgooni/Msc_project'
train_files = ['/My data/95.10.21.csv']
test_files = ['/My data/95.10.15.csv','/My data/before 95.08.csv','/My data/from 95.8.2 till 95.9.17.csv']

paths = []
names = []
sampling_rates = []
labels = []
for file in train_files:
    with open(main_path + file) as csvfile:
        readCSV = csv.reader(csvfile)
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
        print(file + 'is read as train sample!')

print('Normal and train samples:')
#load train data
train_x_reshaped = np.empty([0, 1, look_back])
train_y = np.empty([0, 1])
for i in range(len(names)):
    if labels[i] == 'Normal':
        dataset = pandas.read_csv(main_path + paths[i] + names[i] + '.txt', delimiter='\t', skiprows=4)
        x_signal = dataset.values[:, 0]
        y_signal = dataset.values[:, 1]
        normalized_signal = normalize_data(pandas.DataFrame(y_signal), max_range, min_range)
        normalized_signal, params = smoother(normalized_signal[:,0])
        sample_x, sample_y = load_data(pandas.DataFrame(normalized_signal), look_back)
        sample_x_reshaped = np.reshape(sample_x, (sample_x.shape[0], 1, sample_x.shape[1]))
        if i == 0:
            train_x_reshaped = sample_x_reshaped
            train_y = sample_y
        else:
            train_x_reshaped = np.concatenate([train_x_reshaped, sample_x_reshaped])
            train_y = np.concatenate([train_y, sample_y])
        print("%d ----->%s, %s, sampling rate=%s" % ((i + 1), names[i], labels[i], sampling_rates[i]))

model = create_model('run1', train_x=train_x_reshaped,train_y=train_y,hidden_nodes=100,input=look_back)

#do 2nd step
arrhythmic_rmse = []
normal_rmse = []
for i in range(len(names)):
    dataset = pandas.read_csv(main_path + paths[i] + names[i] + '.txt', delimiter='\t', skiprows=4)
    x_signal = dataset.values[:, 0]
    y_signal = dataset.values[:, 1]
    normalized_signal = normalize_data(pandas.DataFrame(y_signal), max_range, min_range)
    normalized_signal, params = smoother(normalized_signal[:, 0])
    sample_x, sample_y = load_data(pandas.DataFrame(normalized_signal), look_back)
    sample_x_reshaped = np.reshape(sample_x, (sample_x.shape[0], 1, sample_x.shape[1]))
    predicted = model.predict(sample_x_reshaped)
    rmse = np.sqrt(((predicted - sample_y) ** 2).mean(axis=0))
    if labels[i] == 'Normal':
        normal_rmse.append(rmse)
    else:
        arrhythmic_rmse.append(rmse)
    print("(%s), rmse = %d, real = %s" % (name, rmse, labels[i]))

normal_mu, normal_std = norm.fit(normal_rmse)
arrhythmic_mu, arrhythmic_std = norm.fit(arrhythmic_rmse)

#test for train data
tp = 0
fp = 0
tn = 0
fn = 0
n = 0
for i in range(len(names)):
    dataset = pandas.read_csv(main_path + paths[i] + names[i] + '.txt', delimiter='\t', skiprows=4)
    x_signal = dataset.values[:, 0]
    y_signal = dataset.values[:, 1]
    normalized_signal = normalize_data(pandas.DataFrame(y_signal), max_range, min_range)
    normalized_signal, params = smoother(normalized_signal[:, 0])
    sample_x, sample_y = load_data(pandas.DataFrame(normalized_signal), look_back)
    sample_x_reshaped = np.reshape(sample_x, (sample_x.shape[0], 1, sample_x.shape[1]))
    predicted = model.predict(sample_x_reshaped)
    rmse = np.sqrt(((predicted - sample_y) ** 2).mean(axis=0))
    p_normal = norm.pdf(rmse, normal_mu, normal_std)
    p_arrhythmic = norm.pdf(rmse, arrhythmic_mu, arrhythmic_std)

    if p_normal > p_arrhythmic:
        predicted_label = 'Normal'
    else:
        predicted_label = 'Arrhythmic'
    print("(%s), rmse = %d, real = %s, predicted = %s" % (name, rmse, labels[i], predicted))
    n += 1
    if labels[i] == 'Normal':
        if predicted_label == 'Normal':
            tp += 1
        else:
            fp += 1
    else:
        if predicted_label == 'Arrhythmic':
            tn += 1
        else:
            fn += 1
print(model.get_config())
print('result for train data')
print('tp = %f, tn = %f, fp = %f, fn = %f, total-> %f' %(tp,tn,fp,fn,((tp+tn)/n)))

# test for test data
paths = []
names = []
sampling_rates = []
labels = []
for file in test_files:
    with open(main_path + file) as csvfile:
        readCSV = csv.reader(csvfile)
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
        print(file + 'is read as test sample!')


tp = 0
fp = 0
tn = 0
fn = 0
n = 0
for i in range(len(names)):
    dataset = pandas.read_csv(main_path + paths[i] + names[i] + '.txt', delimiter='\t', skiprows=4)
    x_signal = dataset.values[:, 0]
    y_signal = dataset.values[:, 1]
    normalized_signal = normalize_data(pandas.DataFrame(y_signal), max_range, min_range)
    sample_x, sample_y = load_data(pandas.DataFrame(normalized_signal), look_back)
    sample_x_reshaped = np.reshape(sample_x, (sample_x.shape[0], 1, sample_x.shape[1]))
    predicted = model.predict(sample_x_reshaped)
    rmse = np.sqrt(((predicted - sample_y) ** 2).mean(axis=0))
    p_normal = norm.pdf(rmse, normal_mu, normal_std)
    p_arrhythmic = norm.pdf(rmse, arrhythmic_mu, arrhythmic_std)

    if p_normal > p_arrhythmic:
        predicted_label = 'Normal'
    else:
        predicted_label = 'Arrhythmic'
    print("%d: (%s), rmse = %d, real = %s, predicted = %s" % (i,name, rmse, labels[i], predicted_label))
    n += 1
    if labels[i] == 'Normal':
        if predicted_label == 'Normal':
            tp += 1
        else:
            fp += 1
    else:
        if predicted_label == 'Arrhythmic':
            tn += 1
        else:
            fn += 1
print('result for test data:')
print('tp = %f, tn = %f, fp = %f, fn = %f, total-> %f' %(tp,tn,fp,fn,((tp+tn)/n)))

#check for single sample
j = i
dataset = pandas.read_csv(main_path + paths[j] + names[j] + '.txt', delimiter='\t', skiprows=4)
x_signal = dataset.values[:, 0]
y_signal = dataset.values[:, 1]
normalized_signal = normalize_data(pandas.DataFrame(y_signal), max_range, min_range)
sample_x, sample_y = load_data(pandas.DataFrame(normalized_signal), look_back)
sample_x_reshaped = np.reshape(sample_x, (sample_x.shape[0], 1, sample_x.shape[1]))
predicted = model.predict(sample_x_reshaped)
rmse = np.sqrt(((predicted - sample_y) ** 2).mean(axis=0))
p_normal = norm.pdf(rmse, normal_mu, normal_std)
p_arrhythmic = norm.pdf(rmse, arrhythmic_mu, arrhythmic_std)

if p_normal > p_arrhythmic:
    predicted_label = 'Normal'
else:
    predicted_label = 'Arrhythmic'
print("(%s), rmse = %d, real = %s, predicted = %s" % (name, rmse, labels[i], predicted_label))

trace1 = go.Scatter(y=sample_y[:7000], x=sample_x, name='Real signal')
trace2 = go.Scatter(y=predicted[:7000], x=sample_x, name='Predicted by model')
layout = go.Layout(title=names[i])
figure = go.Figure(data=[trace1, trace2], layout=layout)
py.plot(figure, filename=names[i])
