import yfinance as yf
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import os


class Ind_Week:
    def __init__(self, tickers=None):
        self.tickers = tickers
        for t in tickers:
            OI_Stock(t)

        self.clean_directory()

    def clean_directory(self):
        save_dir = os.path.join(os.getcwd(), 'static/images/indices')

        if os.path.exists(save_dir) and os.path.isdir(save_dir):
            # Get the list of files in the directory
            files_in_dir = os.listdir(save_dir)

            # Iterate over files and remove those not corresponding to tickers
            for file_name in files_in_dir:
                # Check if the file corresponds to a ticker
                if file_name.endswith('.png') and file_name[:-4] not in self.tickers:
                    # If not, delete the file
                    os.remove(os.path.join(save_dir, file_name))


class OI_Stock:
    def __init__(self, ticker):
        self.ticker = ticker.strip().upper()
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

        self.implied_month_move = (self.price * ((1+self.iv) ** (1/4.5927))) - self.price
        self.move_ratio = self.implied_month_move/self.price

        self.one_sd_up = self.price + self.implied_month_move
        self.one_sd_down = self.price - self.implied_month_move
        self.two_sd_up = self.one_sd_up + self.implied_month_move
        self.two_sd_down = self.one_sd_down - self.implied_month_move


        self.past_36 = self.get_past_36_days_closing_prices()

        self.today = str(datetime.today()).split()[0]
        day = self.today[-2:]
        self.today = self.today[:-2] + str(int(day)-1)

        self.graph()





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
            return None

    def graph(self):
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
        up[-17] = None
        down[-17] = None
        up2[-17] = None
        down2[-17] = None

        plt.plot(up2, label=f'2 StDev up: ${self.two_sd_up:.2f}', color='g', linewidth=2.7)
        plt.plot(up, label=f'1 StDev up: ${self.one_sd_up:.2f}', color='g', linewidth=5.4)

        plt.plot(down, label=f'1 StDev down: ${self.one_sd_down:.2f}', color='r', linewidth=5.4)
        plt.plot(down2, label=f'2 StDev down: ${self.two_sd_down:.2f}', color='r', linewidth=2.7)


        plt.xticks([])
        plt.ylabel('Price')
        plt.title(f'${self.ticker} Next Month Volatility\nIVx: % {self.iv*100:.2f}, +/- '
                  f'${self.implied_month_move:.2f}')
        plt.grid(True)

        plt.legend()  # Show legend
        plt.tight_layout()

        # Save plot to a directory

        save_dir = os.path.join(os.getcwd(), f'static/images/indices')


        #save_dir = os.path.join(os.getcwd(), f'earnings_plots-{self.today}')
        os.makedirs(save_dir, exist_ok=True)  # Create directory if it doesn't exist
        plt.savefig(os.path.join(save_dir, f'{self.ticker}.png'))

        plt.close()


        # plt.show()



def main():


    Ind_Week(['AAPL', 'RDDT', 'SMCI', 'COIN', 'NVDA'])


if __name__ == '__main__':
    main()