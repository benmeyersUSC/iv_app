import pandas as pd
import matplotlib.pyplot as plt
import numpy as np



notes = {'ZT': {'term': 2, 'par': 200000, 'multiplier': ((1/8)*(.01/32)), 'tick': 7.8125, 'increment':32*8},
         'ZF': {'term': 5, 'par': 100000, 'multiplier': ((1/4)*(.01/32)), 'tick': 7.8125, 'increment':32*4},
         'ZN': {'term': 10, 'par': 100000, 'multiplier': ((1/2)*(.01/32)), 'tick': 15.625, 'increment':32*2},
         'ZB': {'term': 15, 'par': 100000, 'multiplier': (.01/32), 'tick': 31.25, 'increment':32},
         'TWE': {'term': 20, 'par': 100000, 'multiplier': (.01/32), 'tick': 31.25, 'increment':32},
         'UB': {'term': 30, 'par': 100000, 'multiplier': (.01/32), 'tick': 31.25, 'increment':32} }

class BondFuture():
    def __init__(self, symbol, price):
        self.units = 1000 if symbol != 'ZT' else 2000
        self.price = price
        self.whole_units = float(price[:price.index('-')])
        self.fractional_units = float(price[price.index('-')+1:])
        self.symbol = symbol
        self.info = notes[self.symbol]
        self.term = self.info['term']
        self.par = self.info['par']
        self.multiplier = self.info['multiplier']
        self.tick = self.info['tick']
        self.increment = float(self.info['increment'])
        self.notional_per_contract = round(self.units * (self.whole_units + (self.fractional_units/self.increment)), 4)
        self.name = f"U.S. Treasury Note, {self.term} years"
class BondFuturePosition(BondFuture):
    def __init__(self, contract, quantity, dtd):
        super().__init__(contract.symbol, contract.price)
        self.quantity = int(quantity)
        self.dtd = int(dtd)
        self.notional = round(self.notional_per_contract * self.quantity, 4)

future = BondFuture('UB', '109-30')
long_zn = BondFuturePosition(future, 1, 50)
print(long_zn.name, '-->', long_zn.notional)