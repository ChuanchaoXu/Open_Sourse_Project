# -*- coding: utf-8 -*-
# author:llx  time: 2018/1/3

import requests
import time as time_sleep
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime

def getHTMLText(url):
    try:
        r = requests.get(url, timeout=30)
        r.raise_for_status()
        r.encoding = r.apparent_encoding  # 'utf-8' #
        return r.text
    except:
        return " "


def getData(data):
    data = data.split(',')
    # name = data[0]
    todayOpen = data[1]
    yesterdayClose = data[2]
    nowPrice = data[3]
    todayHigh = data[4]
    todayLow = data[5]

    bidPrice = data[6]
    askPrice = data[7]
    quantity = data[8]
    money = data[9]

    buy1Volume = data[10]
    buy1Price = data[11]
    buy2Volume = data[12]
    buy2Price = data[13]
    buy3Volume = data[14]
    buy3Price = data[15]
    buy4Volume = data[16]
    buy4Price = data[17]
    buy5Volume = data[18]
    buy5Price = data[19]

    sell1Volume = data[20]
    sell1Price = data[21]
    sell2Volume = data[22]
    sell2Price = data[23]
    sell3Volume = data[24]
    sell3Price = data[25]
    sell4Volume = data[26]
    sell4Price = data[27]
    sell5Volume = data[28]
    sell5Price = data[29]

    date = data[30]
    time = data[31]

    data = {'todayOpen': todayOpen, 'yesterdayClose': yesterdayClose, 'nowPrice': nowPrice, 'todayHigh': todayHigh,
            'todayLow': todayLow, 'bidPrice': bidPrice, 'askPrice': askPrice, 'quantity': quantity, 'money': money,
            'buy1Volume': buy1Volume, 'buy1Price': buy1Price, 'buy2Volume': buy2Volume, 'buy2Price': buy2Price,
            'buy3Volume': buy3Volume, 'buy3Price': buy3Price, 'buy4Volume': buy4Volume, 'buy4Price': buy4Price,
            'buy5Volume': buy5Volume, 'buy5Price': buy5Price, 'sell1Volume': sell1Volume, 'sell1Price': sell1Price,
            'sell2Volume': sell2Volume, 'sell2Price': sell2Price, 'sell3Volume': sell3Volume, 'sell3Price': sell3Price,
            'sell4Volume': sell4Volume, 'sell4Price': sell4Price, 'sell5Volume': sell5Volume, 'sell5Price': sell5Price,
            'date': date, 'time': time}
    return data


def main():
    code = r'sh510900'
    url = 'http://hq.sinajs.cn/list=' + code
    store_time = list()
    store_price = list()
    fig = plt.figure(figsize=(15, 5))
    ax = fig.add_subplot(1,1,1)
    plt.xlim((datetime(2018, 1, 5, 13, 00), datetime(2018, 1, 5, 15, 31)))
    while True:
        data = getHTMLText(url)
        data = getData(data)
        time = data['time']
        nowPrice = float(data['nowPrice'])
        store_time.append(time)
        store_price.append(nowPrice)
        Price = pd.DataFrame(store_price, index=store_time, columns=['price'])
        Price.index = Price.index.to_datetime()
        index = list(Price.index)
        if len(index) > 5:
            # plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m/%d/%Y %H:%M:%S'))
            # plt.xlim((datetime(2017, 1, 5, 9, 30), datetime(2017, 1, 5, 15, 31)))
            ax.plot(index[len(index) - 30:len(index) - 1:1], store_price[len(index) - 30:len(index) - 1:1])
            # plt.set_title("%s" % code)
            plt.pause(0.01)

        time_sleep.sleep(1)
        print data['time'], data['nowPrice']


main()
