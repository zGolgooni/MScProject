__author__ = 'Zeynab'
import numpy as np
from biosppy.signals import ecg
import plotly.plotly as py
import plotly.graph_objs as go
from plotly import tools
import matplotlib.pyplot as plt

from scipy.optimize import curve_fit



index=10
a = rpeaks[index]
b=rpeaks[index +1]
l=b-a

x=signal[a:b,0]
y=signal[a:b,1]
"""
def gauss_function(x, a, x0, sigma):
    return a*np.exp(-(x-x0)**2/(2*sigma**2))

popt,pcov=curve_fit(gauss_function,x,y)
"""
coefficient = np.polyfit(x,y,5)
poly_y=[]
p = np.poly1d(coefficient)
for i in x:
    poly_y.append(p(i))

plt.plot(x,y)
plt.plot(x,poly_y)
plt.show()



"""
Detect by using biosppy library and then climbing hill algorithm
"""
checked = ['ES_CM N1','ES-CM Nr3','SSC2 R2 7 30 Nov baseline Control','ES-CM Nr50001','SSC1-Nif 95.7.10 10nM Control','ES-CM N20001']
#path='/Users/Zeynab/PycharmProjects/new series/Bam 95.8.11/Iso/'
path1 = '/Users/Zeynab/PycharmProjects/Control/'
path2 = '/Users/Zeynab/PycharmProjects/DATA-Mine/Arrhythmic/'
normal_train = ['SSC1-Nif 95.7.3 0nM Control', 'Control Bam QTC','ES-CM N1','ES-CM Nr50001','SSC2 R1 4 29 Nov iso Control','SSC2-iso 95.6.30 0nM Ch65 Control','Control Bam R09','SSC2 R2 7 30 Nov baseline Control','SSC2 R2 DIV8 R2 6 16 Nov baseline Control','iPS-CM Nr.1','SSC1-Nif 95.7.10 10nM Control','ES-CM N20001','ES-CM Nr50002','ES-CM Nr3','SSC1 R2 iso-pro 95.7.12 baseline Control','SSC1-baseline 5 Oct Control','Control SSc2 10 Oct R2 Ver','ES-CM N2','SSC1 R1-baseline 7 Oct Control','SSC2 R2 DIV8 R2 6 16 Nov iso Control','SSC R2 1 19 Oct baseline Control','SSC2 R2 3 29 Nov baseline Control']

#name = 'CPVT1 Nr1 RA52 95.4.8 iso Arrhythmic'
name='SSC1-iso 95.7.3 0nM Control'
signal = np.loadtxt(path1+name+'.txt',skiprows=1)
rate=1000.
