__author__ = 'Zeynab'

from prepare_files import load_files
from prepare_data import read_sample, normalize_data, load_data, prepare_for_lstm, look_back, horizon, min_range, max_range, total_length
from lstm_model import create_model, fit_model, load_model
from check_result import check_rmse, check_label,check_plot


def test_files(path, file, model, normal_mu, normal_std, arrhythmic_mu, arrhythmic_std):
    paths, names, sampling_rates, labels = load_files(path, file)
    tp = 0
    fp = 0
    tn = 0
    fn = 0
    n = 0
    for i in range(len(names)):
        predicted_signal, rmse, predicted_label = check_label(paths[i], names[i], model, normal_mu, normal_std, arrhythmic_mu, arrhythmic_std)
        #check_plot(paths[i], names[i], labels[i], model)
        print("(%s), rmse = %d, real = %s, predicted = %s" % (names[i], rmse, labels[i], predicted_label))
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
    print('result for data = %s , %d samples (%d N, %d A)', file,n, (fn+tp), (fp+tn))
    print('tp = %f, tn = %f, fp = %f, fn = %f, total-> %f' %(tp,tn,fp,fn,((tp+tn)/n)))