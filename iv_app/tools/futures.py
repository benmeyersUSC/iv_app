import pandas as pd
import matplotlib.pyplot as plt
import numpy as np



notes = {'ZT': {'term': 2, 'par': 200000, 'multiplier': ((1/8)*(.01/32)), 'tick': 7.8125, 'increment':32*8},
         'ZF': {'term': 5, 'par': 100000, 'multiplier': ((1/4)*(.01/32)), 'tick': 7.8125, 'increment':32*4},
         'ZN': {'term': 10, 'par': 100000, 'multiplier': ((1/2)*(.01/32)), 'tick': 15.625, 'increment':32*2},
         'ZB': {'term': 15, 'par': 100000, 'multiplier': (.01/32), 'tick': 31.25, 'increment':32},
         'UB': {'term': 30, 'par': 100000, 'multiplier': (.01/32), 'tick': 31.25, 'increment':32} }

class BondFuture():
    def __init__(self, symbol, price):
        self.symbol = symbol.upper()
        self.units = 1000 if self.symbol != 'ZT' else 2000
        self.price_str = price
        self.spot = float(price[:price.index('-')])
        thirty_seconds = price[price.index('-')+1:]
        if len(thirty_seconds) < 3:
            thirty_seconds += '0'
        self.num32s = float(thirty_seconds[:-1])
        if self.symbol == 'ZN':
            self.num32s += float(thirty_seconds[-1])/10
        elif self.symbol == 'ZF':
            tic_dic = {'0':0, '2':.25, '5':.5, '7':.75}
            self.num32s += tic_dic[thirty_seconds[-1]]
        elif self.symbol == 'ZT':
            self.num32s += float(thirty_seconds[-1])/8
        self.points_dec = self.spot + self.num32s/32
        self.info = notes[self.symbol]
        self.term = self.info['term']
        self.par = self.info['par']
        self.multiplier = self.info['multiplier']
        self.tick = self.info['tick']
        self.increment = float(self.info['increment'])
        self.notional_per_future = self.units * self.points_dec
        self.name = f"U.S. Treasury Note, {self.term} years"

class BondFuturePosition(BondFuture):
    def __init__(self, contract, quantity, dtd):
        super().__init__(contract.symbol, contract.price_str)
        self.quantity = int(quantity)
        self.dtd = int(dtd)
        self.notional = round(self.notional_per_future * self.quantity, 4)

def main():

    future = BondFuture('ZN', '107-280')
    fut = BondFuture('ZT', '101-188')

    long_zn = BondFuturePosition(future, 6, 45)
    short_zt = BondFuturePosition(fut, 3, 45)

    print(long_zn.notional, short_zt.notional)

if __name__ == '__main__':
    main()