import sys
import csv
import datetime as dt
import matplotlib
import matplotlib.pyplot as plt
import math
import numpy as np
import padasip as pa

def predictor2(x, delay=2, order=3, mu=.98, eps=0.98):
    xd=np.pad(x, (delay, 0), mode='constant')[:len(x)]
    yd=np.zeros(len(x))
    y=np.zeros(len(x))
    e=np.zeros(len(x))
    n=order+1
    f1=pa.filters.FilterNLMS(n, mu=mu, w='zeros')
    for k in range(order, len(x)):
        a = xd[k-order:k+1]
        yd[k] = f1.predict(a)
        f1.adapt(x[k], a)
    return xd, y, yd, e

def predictor(x, delay=2, order=3, mu=0.98, eps=0.1):
    n = order + 1
    w = np.zeros(n)
    xd = np.pad(x, (delay, 0), mode='constant')[:len(x)]
    yd = np.zeros(len(x))
    y = np.zeros(len(x))
    e = np.zeros(len(x))
    for k in range(order, len(x)):
        a = np.arange(k-order, k+1)
        yd[k] = np.dot(w, xd[a])
        y[k] = np.dot(w, x[a])
        e[k] = x[k] - yd[k]
        nu = mu / (eps + np.dot(xd[a], xd[a]))
        w += nu * e[k] * xd[a]
    return xd, y, yd, e

if __name__ == "__main__":

    data = []
    #with open('data/DAT_ASCII_EURGBP_M1_201605.csv') as csvfile:
    with open('data/TR4DER_PETR3.SA_20090923.csv') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        reader.next()
        for row in reader:
            data.append(float(row[7]))

    # N = 1000
    # n = 15
    # u = 0.5 * np.sin(np.arange(0, N/10., N/10000.))
    # v = np.random.random(N)
    # x = u + v

    x = data
    x = np.linspace(0, 100, 1001)

    delay=15
    order=100

    s = np.array(x)
    x = np.pad(s, (delay, 0), mode='constant')

    N = len(s)
    y = np.zeros(N)
    r = np.zeros(N)
    f = pa.filters.FilterNLMS(order, mu=0.98, w='zeros')

    print N
    for k in xrange(order, N):
        a = x[k:k-order:-1]
        # a = x[k-delay]
        y[k] = f.predict(a)
        f.adapt(s[k], a)
        r[k] = np.dot(f.w, s[k:k-order:-1])
        #print 'k={:3} | s[k]={:4.2f} r[k]={:4.2f} | x[k]={:4.2f} y[k]={:4.2f}'.format(k, s[k], r[k], x[k], y[k]), a

    #xd, y, yd, e = predictor2(x, mu=mu, order=order, delay=delay)
    #xd, y, yd, e = predictor2(x, mu=mu, eps=eps, order=order, delay=delay)

    #k=np.arange(2000, N)
    #plt.plot(k, s[k], '.-', color='red', label='x')
    #plt.plot(k, y[k], '.-', color='blue', label='y')

    plt.plot(s, '.-', color='red', label='s')
    #plt.plot(y, '.-', color='blue', label='y')
    plt.plot(r, '.-', color='green', label='r')

    plt.legend(loc='lower right')
    plt.show()
