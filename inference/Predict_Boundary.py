#adjusted from https://github.com/saakur/EventSegmentation/blob/master/getBoundaries.py
import numpy as np
from scipy import signal
from math import factorial
import warnings
import os

def warn(*args, **kwargs):
    pass
warnings.warn = warn

def movingaverage(interval, window_size):
    window = np.ones(int(window_size))/float(window_size)
    return np.convolve(interval, window, 'same')

def savitzky_golay(y, window_size, order, deriv=0, rate=1):
    try:
        window_size = np.abs(np.int(window_size))
        order = np.abs(np.int(order))
    except ValueError:
        raise ValueError("window_size and order have to be of type int")
    if window_size % 2 != 1 or window_size < 1:
        raise TypeError("window_size size must be a positive odd number")
    if window_size < order + 2:
        raise TypeError("window_size is too small for the polynomials order")
    order_range = range(order+1)
    half_window = (window_size -1) // 2
    b = np.mat([[k**i for i in order_range] for k in range(-half_window, half_window+1)])
    m = np.linalg.pinv(b).A[deriv] * rate**deriv * factorial(deriv)
    firstvals = y[0] - np.abs( y[1:half_window+1][::-1] - y[0] )
    lastvals = y[-1] + np.abs(y[-half_window-1:-1][::-1] - y[-1])
    y = np.concatenate((firstvals, y, lastvals))
    return np.convolve( m[::-1], y, mode='valid')


def Predict_Boundary(mse_path,smooth_factor,range_consider):
    predfile_list=[x for x in os.listdir(mse_path) if "mse.txt" in x]
    for item in predfile_list:
        study_file=os.path.join(mse_path,item)
        predErrors = []
        Frame_No_list=[]
        with open(study_file, 'r') as file:
            line = file.readline()
            while line:
                data = line.strip("\n")
                data=data.split()
                frameNo=data[0]
                predError = data[1]
                frameNo, predError = int(frameNo), float(predError)
                Frame_No_list.append(frameNo)
                predErrors.append(predError)
                line=file.readline()
        predErrors_Ori = predErrors
        predErrors = movingaverage(predErrors, smooth_factor)
        predErrors = np.gradient(np.array(predErrors)).tolist()

        predBoundaries = signal.argrelextrema(np.array(predErrors), np.greater, order=range_consider)[0].tolist()
        #predBoundaries.append(len(predErrors_Ori)-1)
        Final_Boundaray=[]
        for bound in predBoundaries:
            real_bound=Frame_No_list[bound]
            Final_Boundaray.append(real_bound)
        predBoundaries=Final_Boundaray
        outFile = study_file.replace('_mse.txt', '_predBoundaries_'+str(smooth_factor)+'_'+str(range_consider)+'.txt')
        with open(outFile, 'w') as of:
            for p in predBoundaries:
                of.write('%d\n' % p)

