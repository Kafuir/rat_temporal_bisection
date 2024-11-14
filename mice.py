import os
import re
import math
from scipy.stats import norm
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

def make_jpg (x, y, parameters, name): #makes cool jpgs
    #print(y)
    d = 0
    e = 0
    fig = plt.figure() #actually these are pngs
    ax1 = fig.add_subplot(111)
    ax2 = fig.add_subplot(111, frame_on = False)
    #print(4)
    if parameters[0] != -1 and parameters[1] != -1:
        x_0 = list(map(lambda x: x/10000, range(0, 40001)))
        p = list(map (pse, x_0, [parameters[0]]*len(x_0), [parameters[1]]*len(x_0)))
        ax1.plot(x_0, p)
        d = derivative(x_0, p)
        #print('Max', d)
        e = tabul(parameters)
        ax1.scatter(e, 0.5, color = 'red', marker = 'o')
    if type(name) is str:
        #print (x, y[1])
        ax2.scatter(x,y[1])
        ax2.errorbar(x, y[1], y[0], fmt='none', ecolor = 'blue') 
    else:
        ax2.scatter(x, y)
    #print(5)

    ax1.set_xlabel('Time')
    ax1.set_ylabel('Probability')
    ax2.set_xlim([-0.1, 4.1])
    ax2.set_ylim([-0.1, 1.1])
    ax1.set_xlim([-0.1, 4.1])
    ax1.set_ylim([-0.1, 1.1])
    ax1.xaxis.set_major_locator(mticker.FixedLocator([0, 1, 1.7, 2.5, 3.3, 4]))
    ax2.xaxis.set_major_locator(mticker.FixedLocator([0, 1, 1.7, 2.5, 3.3, 4]))
    ax1.yaxis.set_major_locator(mticker.FixedLocator([0, 0.5, 1]))
    ax2.yaxis.set_major_locator(mticker.FixedLocator([0, 0.5, 1]))
    ax1.grid()
    if os.path.isdir('pngs') is False:
        os.mkdir('pngs')
    plt.savefig(f'pngs/{str(name)[:-2]}.png')
    plt.close()
    return [d, round(e, 3)]

def pse (x, A, B): #not actually PSE, it's just an equation that is used for curve_fit to find it out with sum machine learning
    y = norm.cdf(0.6745*(x - A)/B)#*
    return y

def pse1 (x, A, B):
    y = 1/(1+math.exp(-B*(x-A)))
    return y

def read_txt(filename):
    result = []
    with open(filename, 'r') as file:
        for line in file:
            result.append([float(i) for i in line.split()])
    return result

def fit_mice(numbers, times):
    try:
        parameters, covariance = curve_fit(pse, times, numbers, maxfev=10000) #machine learning magic
    except Exception as e:
        print (e)
        parameters = [-1, -1]
    return parameters

def seek_mean (total):
    a, b = 0, 0
    for entry in total:
        a += entry[0]
        b += entry[1]
    return [a/len(total), b/len(total)]

def derivative (x, y):
    der = []
    max_value = -99999
    max_el = None
    for i in range(len(x)-1):
        der_el = ((y[i+1]-y[i])/(x[i+1]-x[i]))
        if der_el > max_value:
            max_value = der_el
            max_el = [round(x[i+1], 3), round(y[i+1], 3)]
    return [max_value, max_el] #returns the point of der

def tabul(param, y = 0.5):
    e = 0.001
    delta = 1
    x = 2
    check = pse(x, param[0], param[1])
    while delta > e and abs(round(y-check, 2)) > 0.001:
        #print(param)
        if (check < y):
            x += delta
            delta /= 2
        if (check == y):
            #print ('Accurate', round(x, 2))
            return x
        if (check > y):
            x -= delta
            delta /= 2
        check = pse(x, param[0], param[1])
        #print(delta)
    #print ('Close', round(x, 2), round(y-check, 3))
    return x

def CI (data):
    mean = np.mean(data)
    #print(mean)
    std_dev = np.std(data, ddof=1)  # Using Bessel's correction
    n = len(data)
    confidence_level = 0.95
    z_score = 1.96  # For 95% confidence level

    confidence = z_score * std_dev / np.sqrt(n)
    center = np.median(data)
    #print (center)
    return [std_dev, center]


def do_things(filename):
    print ('NUMBER T50 ETA SLOPE') #SLOPE_POINT TABULATED_ETA')
    x = read_txt(filename)
    times  = [1, 1.7, 2.5, 3.3, 4]
    total, data = [], []
    for y in x:
        par = fit_mice(y[1:], times)
        total.append(par)
        data.append(y[1:])
        #print(par)
        out = make_jpg(times, y[1:], par, y[0])
        print(out)
        print (int(y[0]), round(par[0],3), round(par[1],3), round(out[0][0],3), round(par[1]*out[0][0],3))#, out[0][1], out[1])
        
    data2 = np.array(data).T.tolist()
    #for a in data2:
    #    print(a)
    par = seek_mean(total)
    err2 = list(map(CI, data2))
    err = np.array(err2).T.tolist()
    out = make_jpg(times, err, par, filename + 'AA')
    print ('AVERAGE', round(par[0],3), round(par[1],3), round(out[0][0],3), round(par[1]*out[0][0],3))#, out[0][1], out[1])
    #for u in range (0, 40):
    #    print(round(pse(u/10, par[0], par[1])-pse1(u/10, par[0], par[1]), 3)) #quite interesting

do_things('saline_nofail.txt')
