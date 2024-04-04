import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import os
import math

class EarningStock:
    def __init__(self, ticker):
        self.ticker = ticker.strip().upper()
        if self.ticker != 'SPY':
            self.iv = round(self.get_implied_volatility(), 4)
        else:
            self.vix_date = datetime.today()
            # Define the start date (30 days back)
            self.vix_start_date = (self.vix_date - timedelta(days=45)).strftime('%Y-%m-%d')
            start_date = self.vix_start_date
            # # Download VIX data
            vix_data = yf.download('^VIX', start=self.vix_start_date, end=self.vix_date)
            #
            # # Convert DataFrame rows into list of lists
            vix_data_list = vix_data.values.tolist()
            self.vix_prices = []
            for l in vix_data_list:
                if len(l) > 2:
                    self.vix_prices.append(l[-3])

            self.last_vix = self.vix_prices[-1]

            self.iv = round(self.last_vix / 100, 5)


        self.yf_ticker = yf.Ticker(ticker)
        # check if bid and ask keys are accessible on yfinance right now...
        # print(self.yf_ticker.info.keys())
        if 'bid' in self.yf_ticker.info.keys():
            self.underlying_bid = self.yf_ticker.info['bid']
            self.underlying_ask = self.yf_ticker.info['ask']
            # midprice
            self.price = round(((self.underlying_ask + self.underlying_ask) * .5), 2)
        else:
            # if no live midprice, we take previous close
            self.price = self.yf_ticker.info['previousClose']

        self.name = self.yf_ticker.info['shortName'] if 'shortName' in self.yf_ticker.info.keys() else self.ticker

        self.implied_month_move = (self.price * ((1+self.iv) ** (1/4.5927))) - self.price
        self.move_ratio = self.implied_month_move/self.price
        self.implied_45_move = (self.price * ((1+self.iv) ** (1/4))) - self.price

        self.one_sd_up = self.price + self.implied_month_move
        self.one_sd_down = self.price - self.implied_month_move
        self.two_sd_up = self.one_sd_up + self.implied_month_move
        self.two_sd_down = self.one_sd_down - self.implied_month_move


        self.past_36 = self.get_past_36_days_closing_prices()

        self.today = str(datetime.today()).split()[0]
        day = self.today[-2:]
        self.today = self.today[:-2] + str(int(day)-1)

        self.brownianStock = self.brownianStock()




    def get_past_36_days_closing_prices(self):
        date = datetime.today()
        # Define the start date (30 days back)
        start_date = (date - timedelta(days=54)).strftime('%Y-%m-%d')

        # Fetch historical data for the specified ticker and date range
        stock_data = yf.download(self.ticker, start=start_date, end=date)

        # Extract the closing prices for the past 30 days
        closing_prices = [round(x, 2) for x in stock_data['Close'].tolist()]

        return closing_prices

    def get_implied_volatility(self):
        try:
            # Get the options chain for the next expiration cycle
            stock = yf.Ticker(self.ticker)
            expirations = stock.options
            expiration_dates = [datetime.strptime(exp, '%Y-%m-%d') for exp in expirations]
            today = datetime.now().date()

            next_expiration = min(expiration_dates, key=lambda x: abs(x.date() - today))
            next_expiration = expiration_dates[3]
            options = stock.option_chain(next_expiration.strftime('%Y-%m-%d'))

            # Extract at-the-money call option
            calls = options.calls
            atm_call = calls.iloc[(calls['strike'] - stock.history(period="1d").iloc[-1]['Close']).abs().argsort()[:1]]

            # Retrieve implied volatility for the at-the-money call option
            implied_volatility = atm_call['impliedVolatility'].values[0]

            return implied_volatility
        except Exception as e:
            print("Error:", e)
            return None

    def graph_iv(self):
        # print('ticker is', self.ticker)
        if self.ticker != 'SPY':
            prices = self.past_36+[self.price, self.price, self.price, self.price, self.price, self.price, self.price,
                                   self.price, self.price, self.price, self.price, self.price, self.price, self.price,
                                   self.price, self.price]
            if self.past_36[-1] > self.past_36[0]:
                color = 'g'
            else:
                color = 'r'

            up = []
            up2 = []
            down = []
            down2 = []
            for x in range(len(prices)):
                if prices[x] == self.price:
                    up.append(self.one_sd_up)
                    down.append(self.one_sd_down)
                    up2.append(self.two_sd_up)
                    down2.append(self.two_sd_down)
                else:
                    up.append(None)
                    down.append(None)
                    up2.append(None)
                    down2.append(None)

            prices = prices[:-16]
            prices = prices + [self.price, None, None, None, None, None, None, None, None, None, None, None, None, None,
                               None, None]

            plt.plot(prices, label=f'${self.ticker}: ${self.price:.2f}', color=color, marker='>', linestyle='-')

            #adjust stdev lines to make clean
            up[-16] = self.price
            down[-16] = self.price
            up2[-16] = self.price
            down2[-16] = self.price

            plt.plot(up2, label=f'2 StDev up: ${self.two_sd_up:.2f}', color='g', linewidth=2.7)
            plt.plot(up, label=f'1 StDev up: ${self.one_sd_up:.2f}', color='g', linewidth=5.4)

            plt.plot(down, label=f'1 StDev down: ${self.one_sd_down:.2f}', color='r', linewidth=5.4)
            plt.plot(down2, label=f'2 StDev down: ${self.two_sd_down:.2f}', color='r', linewidth=2.7)


            plt.xticks([])
            plt.ylabel('Price')
            plt.title(f'{self.name} (${self.ticker})\nNext Month Volatility\nIVx: % {self.iv*100:.2f}, +/- '
                      f'${self.implied_month_move:.2f}')
            plt.grid(True)

            plt.legend()  # Show legend
            plt.tight_layout()
        else:
            prices = self.past_36 + [self.price, self.price, self.price, self.price, self.price, self.price, self.price,
                                     self.price, self.price, self.price, self.price, self.price, self.price, self.price,
                                     self.price, self.price]
            if self.past_36[-1] > self.past_36[0]:
                color = 'g'
            else:
                color = 'r'

            up = []
            up2 = []
            down = []
            down2 = []
            for x in range(len(prices)):
                if prices[x] == self.price:
                    up.append(self.one_sd_up)
                    down.append(self.one_sd_down)
                    up2.append(self.two_sd_up)
                    down2.append(self.two_sd_down)
                else:
                    up.append(None)
                    down.append(None)
                    up2.append(None)
                    down2.append(None)

            prices = prices[:-16]
            prices = prices + [None, None, None, None, None, None, None, None, None, None, None, None, None, None,
                               None, None]

            plt.plot(prices, label=f'${self.ticker}: ${self.price:.2f}', color=color, marker='>', linestyle='-')

            # adjust stdev lines to make clean
            up[-17] = self.price
            down[-17] = self.price
            up2[-17] = self.price
            down2[-17] = self.price
            up[-18] = None
            down[-18] = None
            up2[-18] = None
            down2[-18] = None


            plt.plot(up2, label=f'2 StDev up: ${self.two_sd_up:.2f}', color='g', linewidth=2.7)
            plt.plot(up, label=f'1 StDev up: ${self.one_sd_up:.2f}', color='g', linewidth=5.4)

            plt.plot(down, label=f'1 StDev down: ${self.one_sd_down:.2f}', color='r', linewidth=5.4)
            plt.plot(down2, label=f'2 StDev down: ${self.two_sd_down:.2f}', color='r', linewidth=2.7)

            plt.xticks([])
            plt.ylabel('Price')
            plt.title(f'{self.name} (${self.ticker})\nNext Month Volatility\nIVx: % {self.iv * 100:.2f}, +/- '
                      f'${self.implied_month_move:.2f}')
            plt.grid(True)

            plt.legend()  # Show legend
            plt.tight_layout()

        # Save plot to a directory
        save_dir = os.path.join(os.getcwd(), 'static', 'images')
        os.makedirs(save_dir, exist_ok=True)  # Create directory if it doesn't exist
        file_path = os.path.join(save_dir, 'yyy.png')

        # Save the plot
        plt.savefig(file_path)

        # Close the plot to release memory
        plt.close()

    def brownianStock(self):
        """
        Description: creates a BLACK-SCHOLES based random daily stock path (list)
        :param params: the output list from user input....only uses all but 0 and 2 (type and strike) all (floats)
        :return: (list) of length dte (params[3]), each a new daily stock price
        """
        days = int(45)

        x = np.random.normal(1, .5)  # this makes volatility normal distributed around 1
        # (should be .85 theoretically)
        newVol = (self.iv * x)
        dailyVolatility = float(newVol / math.sqrt(250))
        dailyRiskFree = 1 + float(.0525) / 365
        dailyPrices = [self.price]
        for x in range(days):
            x = np.random.normal(1, self.iv)  # this makes volatility normal distributed around 1
            # (should be .85 theoretically)
            newVol = (self.iv * x)
            dailyVolatility = float(newVol / math.sqrt(365))
            dailyReturn = dailyRiskFree + dailyVolatility * np.random.normal(0, 1)
            if dailyReturn >= 0:
                newVol *= 1 - dailyReturn * 5  # this branching is to make volatility change with price
            else:  # in most cases, the daily return is small so it stays as was
                newVol += 1 - dailyReturn * 5  # but, when things fall WAY out of place,
                #     it reflects real market (worse for us)
            nextPrice = dailyPrices[-1] * dailyReturn
            dailyPrices.append(round(nextPrice, 2))
        return dailyPrices

    def graphStPrices(self):
        xes = range(len(self.brownianStock))

        if self.brownianStock[-1] > self.brownianStock[0]:
            color = 'g'
        else:
            color = 'r'

        plt.plot(xes, self.brownianStock, label=f'${self.ticker} Price', color=color, marker='>', linestyle='-')
        up = []
        up2 = []
        down = []
        down2 = []
        for x in range(len(self.brownianStock)):
            up.append(self.one_sd_up)
            down.append(self.one_sd_down)
            up2.append(self.two_sd_up)
            down2.append(self.two_sd_down)


        plt.plot(up2, label=f'2 StDev up: ${self.two_sd_up:.2f}', color='g', linewidth=2.7)
        plt.plot(up, label=f'1 StDev up: ${self.one_sd_up:.2f}', color='g', linewidth=5.4)

        plt.plot(down, label=f'1 StDev down: ${self.one_sd_down:.2f}', color='r', linewidth=5.4)
        plt.plot(down2, label=f'2 StDev down: ${self.two_sd_down:.2f}', color='r', linewidth=2.7)

        plt.xlabel('Days From Now')
        plt.ylabel('Stock Price')
        plt.title(f'{self.name} (${self.ticker})\nGBM/Black-Scholes Random Stock Price\nIVx: % {self.iv*100:.2f}, +/- '
                  f'${self.implied_45_move:.2f}')

        plt.grid(True)

        plt.tight_layout()

        # Save plot to a directory
        save_dir = os.path.join(os.getcwd(), 'static', 'images')
        os.makedirs(save_dir, exist_ok=True)  # Create directory if it doesn't exist
        file_path = os.path.join(save_dir, 'zzz.png')

        # Save the plot
        plt.savefig(file_path)

        # Close the plot to release memory
        plt.close()

def main():
    EarningStock('AMZN')


if __name__ == '__main__':
    main()

