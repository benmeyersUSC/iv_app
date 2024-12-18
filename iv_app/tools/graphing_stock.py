import yfinance as yf
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import os
import pandas as pd



class StockSet:
    def __init__(self, tickers=None, category=None):
        self.tickers = tickers
        self.category = category
        for t in tickers:
            #graph each stock, place each in directory
            GraphStock(t, self.category)

        #assure directory has just the recent batch
        self.clean_directory()

    def clean_directory(self):
        # save to static/images/criteria
        save_dir = os.path.join(os.getcwd(), f'static/dynamic/images/{self.category}')

        if os.path.exists(save_dir) and os.path.isdir(save_dir):
            # Get the list of files in the directory
            files_in_dir = os.listdir(save_dir)

            # Iterate over files and remove those not corresponding to tickers
            for file_name in files_in_dir:
                # Check if the file corresponds to a ticker
                if file_name.endswith('.png') and file_name[:-4] not in self.tickers:
                    # If not, delete the file
                    os.remove(os.path.join(save_dir, file_name))


class GraphStock:
    def __init__(self, ticker, category):
        self.ticker = ticker.strip().upper()
        self.category = category
        self.iv = round(self.get_implied_volatility(), 4)


        self.yf_ticker = yf.Ticker(ticker)
        # check if bid and ask keys are accessible on yfinance right now...
        if 'bid' in self.yf_ticker.info.keys():
            self.underlying_bid = self.yf_ticker.info['bid']
            self.underlying_ask = self.yf_ticker.info['ask']
            # midprice
            self.price = round(((self.underlying_ask + self.underlying_ask) * .5), 2)
        else:
            # if no live midprice, we take previous close
            self.price = self.yf_ticker.info['previousClose']

        # Amazon.com, Inc., Apple Inc., etc.
        self.name = self.yf_ticker.info['shortName'] if 'shortName' in self.yf_ticker.info.keys() else self.ticker


        self.implied_month_move = (self.price * ((1+self.iv) ** (1/4.5927))) - self.price
        self.move_ratio = self.implied_month_move/self.price

        self.one_sd_up = self.price + self.implied_month_move
        self.one_sd_down = self.price - self.implied_month_move
        self.two_sd_up = self.one_sd_up + self.implied_month_move
        self.two_sd_down = self.one_sd_down - self.implied_month_move


        self.past_month = self.get_past_month_days_closing_prices()

        self.today = str(datetime.today()).split()[0]
        day = self.today[-2:]
        self.today = self.today[:-2] + str(int(day)-1)

        self.graph()





    def get_past_month_days_closing_prices(self):
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
            next_expiration = expiration_dates[2]
            options = stock.option_chain(next_expiration.strftime('%Y-%m-%d'))

            # Extract at-the-money call option
            calls = options.calls
            atm_call = calls.iloc[(calls['strike'] - stock.history(period="1d").iloc[-1]['Close']).abs().argsort()[:1]]

            # Retrieve implied volatility for the at-the-money call option
            implied_volatility = atm_call['impliedVolatility'].values[0]

            return implied_volatility
        except Exception as e:
            print("Error:", e)
            return 0

    def graph(self):

        prices = self.past_month+[self.price, self.price, self.price, self.price, self.price, self.price, self.price,
                               self.price, self.price, self.price, self.price, self.price, self.price, self.price,
                               self.price, self.price]
        if self.past_month[-1] > self.past_month[0]:
            color = 'g'
        else:
            color = 'r'

        up = []
        up2 = []
        down = []
        down2 = []

        for x in range(len(prices)):
            # all 'future' values of the stock line are initially made to be today's price
            if prices[x] == self.price:
                # so when we reach that, create our SD lines
                up.append(self.one_sd_up)
                down.append(self.one_sd_down)
                up2.append(self.two_sd_up)
                down2.append(self.two_sd_down)
            else:
                # until we reach the future, SD lines are to be blank
                up.append(None)
                down.append(None)
                up2.append(None)
                down2.append(None)

        # now go back, replace repeated prices with None so SD lines can shine
        prices = prices[:-16]
        prices = prices + [self.price, None, None, None, None, None, None, None, None, None, None, None, None, None,
                           None, None]
        today = datetime.now().date()
        days_ago = today - timedelta(days=54)

        dates = pd.date_range(start=days_ago, periods=40, freq='B')

        dates_list = [date.strftime('%Y-%m-%d') for date in dates]
        display_dates = [dates_list[0], dates_list[18], today]

        # Calculate the corresponding indices
        display_indices = [0, 19, 36]


        # plot stock price
        plt.plot(prices, label=f'${self.ticker}: ${self.price:.2f}', color=color, marker='>', linestyle='-')

        #adjust SD lines to make clean
        up[-16] = self.price
        down[-16] = self.price
        up2[-16] = self.price
        down2[-16] = self.price
        up[-17] = None
        down[-17] = None
        up2[-17] = None
        down2[-17] = None

        #plot SD lines
        plt.plot(up2, label=f'2 StDev up: ${self.two_sd_up:.2f}', color='g', linewidth=2.7)
        plt.plot(up, label=f'1 StDev up: ${self.one_sd_up:.2f}', color='g', linewidth=5.4)

        plt.plot(down, label=f'1 StDev down: ${self.one_sd_down:.2f}', color='r', linewidth=5.4)
        plt.plot(down2, label=f'2 StDev down: ${self.two_sd_down:.2f}', color='r', linewidth=2.7)

        #remove x axis
        plt.xticks(display_indices, display_dates, rotation=45)

        # FIGURE OUT HOW TO GET THE X AXIS RIGHT

        plt.ylabel(f'{self.name} Price')
        plt.title(f'{self.name} (${self.ticker})\nNext Month Volatility\nIVx: % {self.iv*100:.2f}, +/- '
                  f'${self.implied_month_move:.2f}')
        plt.grid(True)

        if prices[-18] >= prices[0]:
            plt.legend(loc='upper left', fontsize='small')
        else:
            plt.legend(loc='lower left', fontsize='small')
        plt.tight_layout() # clean space

        # Save plot to a directory
        save_dir = os.path.join(os.getcwd(), f'dynamic/images/{self.category}')

        #save_dir = os.path.join(os.getcwd(), f'earnings_plots-{self.today}')
        os.makedirs(save_dir, exist_ok=True)  # Create directory if it doesn't exist
        plt.savefig(os.path.join(save_dir, f'{self.ticker}.png'))

        #close dis bih
        plt.close()


def main():
    #poop test
    StockSet(['AAPL', 'RDDT', 'SMCI', 'COIN', 'NVDA'], 'poop')

if __name__ == '__main__':
    main()