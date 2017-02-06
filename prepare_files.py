__author__ = 'Zeynab'
import csv


def load_files(path, files):
    paths = []
    names = []
    sampling_rates = []
    labels = []
    for file in files:
        with open(path + file) as csvfile:
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
    return paths,names,sampling_rates,labels