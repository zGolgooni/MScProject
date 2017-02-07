__author__ = 'Zeynab'


import numpy as np
from scipy.stats import norm
from prepare_files import load_files
from prepare_data import read_sample, normalize_data, load_data, prepare_for_lstm, look_back, horizon, min_range, max_range, total_length
from lstm_model import create_model, fit_model, load_model
from check_result import check_rmse, check_label,check_plot
from test_model import test_files
#from biosppy.signals.tools import smoother


main_path = '/Users/Zeynab/PycharmProjects/Msc_project/'
train_files = ['/My data/95.10.21.csv']
#test_files = ['/My data/95.10.15.csv']#,'/My data/before 95.08.csv','/My data/from 95.8.2 till 95.9.17.csv']

#load train data
paths, names, sampling_rates, labels = load_files(main_path, train_files)

mode = 1
if mode == 0:
    #do 1st step
    train_x_reshaped = np.empty([0, 1])
    train_y = np.empty([0, 1])
    counter = 0
    for i in range(len(names)):
        if (labels[i] == 'Normal') & (counter < 5):
            counter +=1
            dataset, x_signal, y_signal = read_sample(paths[i], names[i])
            sample_x, sample_y = prepare_for_lstm(y_signal)
            sample_y = sample_y[:, :, 0]
            if i == 0:
                train_x_reshaped = sample_x
                train_y = sample_y
            else:
                train_x_reshaped = np.concatenate([train_x_reshaped, sample_x])
                train_y = np.concatenate([train_y, sample_y])
            print("%d ----->%s, %s, sampling rate=%s" % ((i + 1), names[i], labels[i], sampling_rates[i]))

    model = create_model(hidden_nodes=100)
    fit_model('horizon5-v1', model, train_x=train_x_reshaped, train_y=train_y, batch=10000, epoch=50, validation=0.2)
else:
    model = create_model(hidden_nodes=100)
    #model = load_model(model,'horizon5-v1.h5')
    model.load_weights('horizon5-v1.h5')

#do 2nd step
if mode == 0:
    arrhythmic_rmse = []
    normal_rmse = []
    for i in range(len(names)):
        predicted_signal, rmse = check_rmse(paths[i], names[i], model = model)
        if labels[i] == 'Normal':
            normal_rmse.append(rmse)
        else:
            arrhythmic_rmse.append(rmse)
        print("(%s), rmse = %f, real = %s, sampling rate=" % (names[i], rmse, labels[i], sampling_rates[i]))

    normal_mu, normal_std = norm.fit(normal_rmse)
    arrhythmic_mu, arrhythmic_std = norm.fit(arrhythmic_rmse)
    print('horizon5-v1:')
    print(' Normal-> mu = %f, std = %f', normal_mu, normal_std)
    print(' Arryhthmic-> mu = %f, std = %f', arrhythmic_mu, arrhythmic_std)
else:
    normal_mu =4.405886357
    normal_std = 1.81739906683
    arrhythmic_mu = 5.44678358922
    arrhythmic_std = 1.72965401846
#classification_model =

#test result for samples

test_files('/Users/Zeynab/PycharmProjects/Msc_project/','/My data/95.10.21.csv',model=model, normal_mu=normal_mu, normal_std=normal_std, arrhythmic_mu=arrhythmic_mu, arrhythmic_std=arrhythmic_std)
#test_files('/Users/Zeynab/PycharmProjects/Msc_project/','/My data/95.10.15.csv',model=model, normal_mu=normal_mu, normal_std=normal_std, arrhythmic_mu=arrhythmic_mu, arrhythmic_std=arrhythmic_std)

