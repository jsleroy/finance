import sys
import csv
import datetime as dt
import matplotlib
matplotlib.rcParams['backend'] = "Qt4Agg"
import matplotlib.pyplot as plt
import math
import numpy as np
import padasip as pa
import pandas as pd

def predictor2(x, delay=2, order=3, mu=.98, eps=0.1):
    xd=np.pad(x, (delay, 0), mode='constant')[:len(x)]
    yd=np.zeros(len(x))
    y=np.zeros(len(x)+delay)
    e=np.zeros(len(x))
    n=order+1
    #f1=pa.filters.FilterNLMS(n, mu=mu, w='zeros')
    f1=pa.filters.rls.FilterRLS(n, mu=mu, eps=eps, w='zeros')
    for k in range(order, len(x)):
        a = xd[k-order:k+1]
        y[k] = f1.predict(a)
        f1.adapt(x[k], a)
    return xd, y, yd, e

def predictor(x, delay=2, order=3, mu=.98):
    xd=np.pad(x, (delay, 0), mode='constant')[:len(x)]
    yd=np.zeros(len(x))
    y=np.zeros(len(x))
    e=np.zeros(len(x))
    n=order+1
    f1=pa.filters.FilterNLMS(n, mu=mu, w='zeros', eps=0.1)
    for k in range(order, len(x)):
        a = xd[k-order:k+1]
        yd[k] = f1.predict(a)
        y[k] = np.dot(f1.w, x[k-order:k+1])
        f1.adapt(x[k], a)
    return xd, y, yd, e

if __name__ == "__main__":
    data = []
    #with open('data/DAT_ASCII_EURGBP_M1_201605.csv') as csvfile:
    with open('data/TickData_EURGBPecn_2016929_1740.csv') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for row in reader:
            data.append(float(row[1]))

    N = 1000
    n = 15
    u = 0.5 * np.sin(np.arange(0, N/10., N/10000.))
    v = np.random.random(N)
    x = u + v

    # x = np.linspace(-np.pi, np.pi, 201)
    # x = np.array(data)

    mu=0.98
    delay=10
    order=100
    # for mu in (0.1, 0.5, 0.98):
    xd, y, yd, e = predictor(x, mu=mu, order=order, delay=delay)

    #c = np.correlate(x, y, "full")

    #plt.subplot(211)
    #plt.plot(x[-view:], '.--', color='black')
    #plt.plot(xd[-view-delay:], '.-', color='red')
    #plt.plot(yd[-view-delay:], '.-', color='green')
    #plt.plot(y[-view+delay:], '.--', color='blue')

    k=np.arange(800,1000)

    plt.plot(k, x[k], '.--', color='black', label='x')
    # plt.plot(k, xd[k], '.-', color='red', label='xd')
    #plt.plot(yd, '.-', color='green', label='yd')
    plt.plot(k, y[k], '.-', color='blue')

    #plt.subplot(212)
    #plt.xcorr(x, y[:len(x)])
    #n = len(y)
    #Y = np.fft.fft(y[k])
    #plt.plot(abs(Y[1:100]), 'r.-')
    #plt.plot(y, 'r.-')
    #plt.plot(yd, 'g.-')
    #lt.subplot(212)
    #plt.plot(c[-view:], 'b.-')
    plt.legend()
    plt.show()


    # c_pad=bk_low(14400, 14400)
    # bid_f_pad=bk_filter_2(bid_v,c_pad)
    # plt.plot(bid_v, 'b', bid_f_pad, 'g')
    # plt.show()
    # simulate(bid_v[1500:], bid_f_pad[1500:], 2)
    # 
    # spread_v = [(ask_v[i]-bid_v[i])*10000 for i in range(0,len(bid_v))]
    # c120=bk(120,3600,36000)
    # c480=bk(480,3600,3600000)
    # bid_f_120=bk_filter(bid_v,c120)
    # bid_f_480=bk_filter(bid_v,c480)
    # bid_f_480_2=bk_filter_2(bid_v,c480)
    # delta=bid_v[600]-bid_f_120[600]
    # bid_c_120 = map(lambda x: x+delta, bid_f_120)
    # delta=bid_v[600]-bid_f_480[600]
    # bid_c_480 = map(lambda x: x+delta, bid_f_480)
    # delta=bid_v[600]-bid_f_480_2[600]
    # bid_c_480_2 = map(lambda x: x+delta, bid_f_480_2)
    # plt.subplot(211)
    # plt.plot(bid_v[500:-500], 'r', bid_c_120[500:-500], 'g', bid_c_480[500:-500], 'b', bid_c_480_2[500:-500], 'y')
    # #Shift a destra della dimensione del campione, per valutare lo sfasamento
    # plt.subplot(212)
    # bid_s_c_120 = bid_c_120[-120:]+bid_c_120[:-120]
    # bid_s_c_480 = bid_c_480[-480:]+bid_c_480[:-480]
    # plt.plot(bid_v[1000:], 'r', bid_s_c_120[1000:], 'g', bid_s_c_480[1000:], 'b')
    # plt.show()



def simulate(actual, filtered, fign, band=.0005, spread=.0003):
    prev = filtered[0]
    price = 0
    gain = [0]
    cumul_gain = 0
    order_buy_time=[]
    order_buy_price=[]
    order_sell_time=[]
    order_sell_price=[]
    for i in range(1, len(actual)):
        cur = filtered[i]
        order_gain = 0
        if price == 0:
            # not yet placed an order
            if cur < prev:
                # current value is less than previous, going down
                prev = cur
            elif cur > prev + band:
                # min found, go for long position
                price = (actual[i]+spread)*1000
                print "Buying on ", i, " at (bid + ", spread*10000, "pip spread) ", actual[i]+spread, " (", actual[i], ")"
                cur = prev
                order_buy_time.append(i)
                order_buy_price.append(actual[i]+spread)
        else:
            #order placed
            if cur > prev:
                # going up
                prev = cur
            elif cur < prev - band:
                # max found, sell the order
                order_gain = (actual[i]*1000 - price)
                cumul_gain += order_gain
                print "Selling on ", i, " at ", actual[i], " gained: ", order_gain, " (Tot gain: ", cumul_gain, ")"
                cur = prev
                price = 0
                order_sell_time.append(i)
                order_sell_price.append(actual[i])
        gain.append(gain[i-1] + order_gain)

    plt.figure(fign)
    plt.plot(actual, 'b', filtered, 'g')
    plt.plot(order_buy_time, order_buy_price, 'oy')
    plt.plot(order_sell_time, order_sell_price, 'or')
    plt.show()


#simulate(bid_v[1000:], bid_s_c_480[1000:], 1)
#simulate(bid_v[1000:], bid_s_c_120[1000:], 2)
#
#
#t=range(2560)
#f4=map(lambda x: math.sin(2*math.pi/4*x), t)
#f8=map(lambda x: math.sin(2*math.pi/8*x), t)
#f16=map(lambda x: math.sin(2*math.pi/16*x), t)
#f32=map(lambda x: math.sin(2*math.pi/32*x), t)
#f64=map(lambda x: math.sin(2*math.pi/64*x), t)
#f128=map(lambda x: math.sin(2*math.pi/128*x), t)
#f256=map(lambda x: math.sin(2*math.pi/256*x), t)
#f3=map(lambda x: math.sin(2*math.pi/3*x), t)
#f17=map(lambda x: math.sin(2*math.pi/17*x), t)
#f31=map(lambda x: math.sin(2*math.pi/31*x), t)
#f100=map(lambda x: math.sin(2*math.pi/100*x), t)
#f145=map(lambda x: math.sin(2*math.pi/145*x), t)
#f196=map(lambda x: math.sin(2*math.pi/196*x), t)
#f307=map(lambda x: math.sin(2*math.pi/307*x), t)
#x=np.array(f4)+np.array(f8)+np.array(f16)+np.array(f32)+np.array(f64)+np.array(f128)+np.array(f256)
#x2=np.array(f3)+np.array(f17)+np.array(f31)+np.array(f100)+np.array(f145)+np.array(f196)+np.array(f307)
#coeff=bk(48, 50, 100)
#y=bk_filter(x, coeff)
#plt.plot(t,x, t, y, 'r', t, f64, 'g')
#plt.show()
