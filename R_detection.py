__author__ = 'Zeynab'
import numpy as np
from biosppy.signals import ecg
import plotly.plotly as py
import plotly.graph_objs as go
from plotly import tools
import matplotlib.pyplot as plt


"""
Detect by using biosppy library and then climbing hill algorithm
"""
checked = ['ES_CM N1','ES-CM Nr3','SSC2 R2 7 30 Nov baseline Control','ES-CM Nr50001','SSC1-Nif 95.7.10 10nM Control','ES-CM N20001']
#path='/Users/Zeynab/PycharmProjects/new series/Bam 95.8.11/Iso/'
path1 = '/Users/Zeynab/PycharmProjects/Control/'
path2 = '/Users/Zeynab/PycharmProjects/DATA-Mine/Arrhythmic/'
normal_train = ['SSC1-Nif 95.7.3 0nM Control', 'Control Bam QTC','ES-CM N1','ES-CM Nr50001','SSC2 R1 4 29 Nov iso Control','SSC2-iso 95.6.30 0nM Ch65 Control','Control Bam R09','SSC2 R2 7 30 Nov baseline Control','SSC2 R2 DIV8 R2 6 16 Nov baseline Control','iPS-CM Nr.1','SSC1-Nif 95.7.10 10nM Control','ES-CM N20001','ES-CM Nr50002','ES-CM Nr3','SSC1 R2 iso-pro 95.7.12 baseline Control','SSC1-baseline 5 Oct Control','Control SSc2 10 Oct R2 Ver','ES-CM N2','SSC1 R1-baseline 7 Oct Control','SSC2 R2 DIV8 R2 6 16 Nov iso Control','SSC R2 1 19 Oct baseline Control','SSC2 R2 3 29 Nov baseline Control']

#name = 'CPVT1 Nr1 RA52 95.4.8 iso Arrhythmic'
name='SSC2 R1 DIV8 R2 1 12 Nov Control'
signal = np.loadtxt(path1+name+'.txt', skiprows=1)
rate=2000.
length = signal.shape[0]
out = ecg.ecg(signal=signal[:,1], sampling_rate=rate, show=False)
temp_rpeaks = out.__getitem__('rpeaks')

rpeaks = np.empty([temp_rpeaks.shape[0], 1],dtype=int)
for i,r in enumerate(temp_rpeaks):
    #find maximum
    rpeaks[i] = r
    max_best_so_far = r
    end = False
    while not end:
        end = True
        for j in range(25):
            neighbor1 = max_best_so_far - j
            neighbor2 = max_best_so_far + j
            if abs(signal[neighbor1, 1]) > abs(signal[max_best_so_far,1]):
                max_best_so_far = neighbor1
                end = False
            if abs(signal[neighbor2, 1]) > abs(signal[max_best_so_far,1]):
                max_best_so_far = neighbor2
                end = False
    rpeaks[i] = max_best_so_far


print('finished finalizing r points! :)')
trace1 = go.Scatter(y=signal[:20000,1], x=signal[:20000,0], name='Signal')

trace2 = go.Scatter(y=signal[rpeaks, 1], x=signal[rpeaks,0],mode='markers', name='final R peaks')
trace3 = go.Scatter(y=signal[temp_rpeaks, 1], x=signal[temp_rpeaks,0],mode='markers', name='temp rpeaks')


layout = go.Layout(title=name)
figure = go.Figure(data=[trace1, trace2, trace3], layout=layout)
py.plot(figure, filename='biosppy test of r peaks detection ' + name)

#number of beats and its rate
total_seconds = length/rate
total_beats = rpeaks.shape[0]
bps_total = total_beats/total_seconds
bpm_total = bps_total * 60


#r-r interval
rr_interval =[]
for i in range(0,rpeaks.shape[0]-1):
    rr_interval.append((rpeaks[i + 1] - rpeaks[i])/rate)
rr_average = np.average(rr_interval)
rr_variance = np.var(rr_interval)

rr_variance=0
rr_problems = []
for i, r in enumerate(rr_interval):
    distance = abs(r-rr_average)
    if distance >= rr_average/2:
        print('problem at r in point %d, index rr_interval = %d' % (i, r))
        rr_problems.append(i)


#set threshold for r-r interval variance

#r peaks amplitude
r_values = signal[rpeaks,1]
r_values_average = np.average(abs(r_values))
r_values_var = np.var(abs(r_values))
r_values_problems = []
for i, r in enumerate(r_values):
    distance = abs(abs(r)-abs(r_values_average))
    if distance >= r_values_average/2:
        print('problem at r in point %d, index r_value = %d' % (i, r))
        r_values_problems.append(i)









#####################################


"""
rpeaks = np.empty([temp_rpeaks.shape[0], 1],dtype=int)
for i,r in enumerate(temp_rpeaks):
    #find maximum
    signal_slice = abs(signal[r - 50: r+50, :])
    max_val = np.amax(signal_slice)
    max_index = np.argmax(signal_slice) -50 + r
    rpeaks[i] = signal[max_index,0]

for i,r in enumerate(temp_rpeaks):
    #find maximum
    rpeaks[i] = r
    max_best_so_far = r
    end = False
    while not end:
        end = True
        for j in range(25):
            neighbor1 = max_best_so_far - j
            neighbor2 = max_best_so_far + j
            if abs(signal[neighbor1, 1]) > abs(signal[max_best_so_far,1]):
                max_best_so_far = neighbor1
                end = False
            if abs(signal[neighbor2, 1]) > abs(signal[max_best_so_far,1]):
                max_best_so_far = neighbor2
                end = False
    rpeaks[i] = max_best_so_far

print('finished finalizing r points! :)')
"""

"""
signal = np.loadtxt(path1+name+'.txt',skiprows=1)

out = ecg.ecg(signal=signal[:20000,1], sampling_rate=1000., show=False)
temp_rpeaks = out.__getitem__('rpeaks')

rpeaks = np.empty([temp_rpeaks.shape[0], 1],dtype=int)
for i,r in enumerate(temp_rpeaks):
    #find maximum
    rpeaks[i] = r
    max_best_so_far = r
    end = False
    while not end:
        end = True
        for j in range(15):
            neighbor1 = max_best_so_far - j
            neighbor2 = max_best_so_far + j
            if signal[neighbor1, 1] > signal[max_best_so_far,1]:
                max_best_so_far = neighbor1
                end = False
            if signal[neighbor2, 1] > signal[max_best_so_far,1]:
                max_best_so_far = neighbor2
                end = False

    #find minimum
    min_best_so_far = r
    end = False
    while not end:
        end = True
        for j in range(15):
            neighbor1 = min_best_so_far - j
            neighbor2 = min_best_so_far + j
            if signal[neighbor1, 1] < signal[min_best_so_far,1]:
                min_best_so_far = neighbor1
                end = False
            if signal[neighbor2, 1] < signal[min_best_so_far,1]:
                min_best_so_far = neighbor2
                end = False
    if abs(signal[min_best_so_far,1]) > abs(signal[max_best_so_far,1]):
        rpeaks[i] = min_best_so_far
        print("hey min selected! %d" %rpeaks[i])
    else:
        rpeaks[i] = max_best_so_far
        print("hey max selected! %d" %rpeaks[i])
print('finished finalizing r points! :)')
trace1 = go.Scatter(y=signal[:20000,1], x=signal[:20000,0], name='Signal')

trace2 = go.Scatter(y=signal[rpeaks, 1], x=signal[rpeaks,0],mode='markers', name='final R peaks')
trace3 = go.Scatter(y=signal[temp_rpeaks, 1], x=signal[temp_rpeaks,0],mode='markers', name='temp rpeaks')


layout = go.Layout(title=name)
figure = go.Figure(data=[trace1, trace2, trace3], layout=layout)
py.plot(figure, filename='biosppy test of r peaks detection ' + name)


"""

