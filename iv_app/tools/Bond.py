import numpy as np
import matplotlib.pyplot as plt
import os

class Bond:
    def __init__(self, par, cr, ttm, price):
        self.par = par
        self.cr = cr
        self.ttm = ttm
        self.price = price
        self.ytm = self.calculate_ytm()
    def calculate_ytm(self, years_to_mat=None, price=None):
        if price == None:
            price = self.price
        tolerance = 0.0001
        ytm_range = [i / 1000 for i in range(1, 100000)]  # Define a range of YTM values (0.001 to 99.999%)
        closest_ytm = None
        min_diff = float('inf')
        if not years_to_mat:
            ttm = self.ttm
        else:
            ttm = years_to_mat

        for ytm in ytm_range:
            # Calculate the present value of bond's cash flows using the current YTM
            pv = 0
            for t in range(1, ttm + 1):
                pv += self.cr * self.par / ((1 + ytm) ** t)
            pv += self.par / ((1 + ytm) ** ttm)

            # Calculate bond price using the current YTM
            calculated_bond_price = pv

            # Check if the difference between the calculated bond price and actual bond price is within tolerance
            diff = abs(calculated_bond_price - price)
            if diff < min_diff:
                min_diff = diff
                closest_ytm = ytm
        return closest_ytm
    def calculate_fair_price(self, alternative_ytm, par=None, cr=None, ttm=None):
        # Calculate the present value of bond's cash flows using the alternative YTM
        if not par:
            par = self.par
        if not ttm:
            ttm = self.ttm
        if not cr:
            cr = self.cr
        pv = 0
        for t in range(1, ttm + 1):
            pv += cr * par / ((1 + alternative_ytm) ** t)
        pv += par / ((1 + alternative_ytm) ** ttm)
        return round(pv, 4)

class BondTrading:
    def __init__(self, bond):
        # init the bond
        self.bond = bond
        self.par = bond.par
        # extract starting Coupon Rate
        self.cr = self.bond.cr
        # starting Maturity
        self.years = bond.ttm
        self.price = self.bond.price

        # used in bond_trading_message(), updated in display_round()
        self.average_ytm = 0

        #cumulative arrays for graphing
        self.prices = [self.price]
        self.yields = [self.bond.calculate_ytm()]
        self.profits = [0]
        self.volatility = [0]
        self.yield_volatility = [0]
        self.position_array = [0]
        self.pos_bar_colors = ['g']

        self.trades = []
        self.transactions = {}

        # seen/used/updated in get_realized(), get_avg_cost(), get_asset_liab(), bond_trading_message(),
        #   get_ready_for_next_period(), and display_round()
        self.position = 0

        # seen/used/updated in get_realized(), get_ready_for_next_period()
        self.inflows = 0
        self.outflows = 0

        self.repaid_principal = 0

    def to_dict(self):
        return {
                'par':self.bond.par, 'cr':self.bond.cr, 'years':self.years, 'price':self.bond.price,
                'average_ytm':self.average_ytm, 'prices':self.prices,
                'yields':self.yields, 'profits':self.profits, 'volatility':self.volatility,
                'yield_volatility':self.yield_volatility, 'position_array':self.position_array,
                'pos_bar_colors':self.pos_bar_colors,  'trades':self.trades, 'position':self.position, 'inflows':self.inflows,
                'outflows':self.outflows, 'repaid_principal':self.repaid_principal, 'profit':self.get_net_prof(),
                    'transactions':self.transactions}
    def from_dict(self, data):
        self.bond.cr = data['cr']
        self.years = data['years']
        self.bond.price = data['price']
        self.average_ytm = data['average_ytm']
        self.prices = data['prices']
        self.yields = data['yields']
        self.profits = data['profits']
        self.volatility = data['volatility']
        self.yield_volatility = data['yield_volatility']
        self.position_array = data['position_array']
        self.pos_bar_colors = data['pos_bar_colors']
        self.trades = data['trades']
        self.position = data['position']
        self.inflows = data['inflows']
        self.outflows = data['outflows']
        self.repaid_principal = data['repaid_principal']
        self.transactions = data['transactions']



    # gets input, returns user choice of 'trade', ie how many shares they want to buy (- for sell), used in display_round() if not maturity
    def get_trade(self):
        return input("\nTrade ('1', '-1, '0'): ")

    # returns average cost of open position (if any), used in bond_trading_message(), and display_round()
    def get_avg_cost(self):
        total_cost = 0
        total_quantity = 0

        # Calculate total cost and total quantity
        for quantity, price in self.trades:
            total_cost += quantity * price
            total_quantity += quantity

        # Avoid division by zero
        if total_quantity == 0:
            return 0

        # Calculate average cost per share
        avg_cost = abs(total_cost) / abs(total_quantity)

        return avg_cost

    # returns overall net profit, used in display_round()
    def get_net_prof(self):
        # print('profit', self.get_realized() + self.get_asset_liab())
        # print('position', self.position)
        return self.get_realized() + self.get_asset_liab()

    # calculates and returns realized profit used in get_net_prof()
    def get_realized(self):
        if self.years == 0:
            if self.position == 0:
                pass
            else:
                if self.position > 0 and self.repaid_principal < 1:
                    self.inflows += 1000 * self.position * (self.bond.par * (1 + self.bond.cr))
                    self.repaid_principal += 1
                    # print('end inflow:', 1000 * self.position * (self.bond.par * (1 + self.bond.cr)))
                    self.transactions['t'+str(len(self.transactions.keys()))] = ('principal settlement', self.position,
                                                                                 self.bond.par * (1 + self.bond.cr),
                                                            f'${1000 * self.position * (self.bond.par * (1 + self.bond.cr)):,.2f}')

                elif self.position < 1 and self.repaid_principal < 1:
                    self.outflows += -1000 * self.position * (self.bond.par * (1 + self.bond.cr))
                    # print('end outflow:', -1000 * self.position * (self.bond.par * (1 + self.bond.cr)))
                    self.repaid_principal += 1
                    self.transactions['t'+str(len(self.transactions.keys()))] = ('principal settlement', self.position,
                                                                                 self.bond.par * (1 + self.bond.cr),
                                                            f'${1000 * self.position * (self.bond.par * (1 + self.bond.cr)):,.2f}')

        return self.inflows - self.outflows

    # calculates and returns equivalent of open profit used in bond_trading_message(), get_avg_cost(), and get_net_prof()
    def get_asset_liab(self):
        if self.position == 0:
            return 0
        if self.years == 0:
            return 0
        return self.position * self.bond.price * 1000

    # no return, builds graph used in display_round()
    def graph_bond_price(self):
        save_dir = os.path.join(os.getcwd(), 'static', 'images', 'bond_trading')
        file_path = os.path.join(save_dir, 'recent_bond_graph.png')
        if os.path.exists(file_path):
            print('file exists')
            os.remove(file_path)
        if os.path.exists(file_path):
            print('file still here :(')


        fig, ax = plt.subplots(2, 2, sharex=True)

        color = 'g' if self.prices[-1] > self.prices[0] else 'r'
        ax[0, 0].plot(self.prices, marker='*', color=color)
        ax[0, 0].plot([self.bond.par] * len(self.prices), marker='*', color='b')
        ax[0, 0].set(title=f"Bond Price = ${self.prices[-1]:,}", xlabel='Years', ylabel='Price')

        color2 = 'g' if self.yields[-1] < self.cr else 'r'
        ax[1, 0].plot(self.yields, marker='*', color=color2, label=f"Yield ({self.yields[-1]*100:,.2f}%)")
        ax[1, 0].plot([self.cr] * len(self.yields), marker='*', color='b', label=f"Coupon ({self.cr*100:.2f}%)")
        ax[1, 0].set(title=f"Yield ({self.yields[-1]*100:,.2f}%) and Coupon ({self.cr*100:.2f}%)", xlabel='Years', ylabel='Rate')
        ax[1, 0].legend(loc='upper left')

        color3 = 'g' if self.profits[-1] > 0 else 'r'
        ax[0, 1].plot(self.profits, marker='*', color=color3, label=f"Profit: ${self.profits[-1]:,.2f}")
        ax[0, 1].set(title=f"Cumulative Profit = ${self.profits[-1]:,.2f}", xlabel='Years', ylabel='Profit')
        ax[0, 1].legend(loc='upper left')

        ax[1, 1].bar(range(len(self.position_array)), self.position_array, color=self.pos_bar_colors)
        ax[1, 1].set(title=f"Position Size: {self.position_array[-1] if len(self.position_array) > 0 else 0}", xlabel='Years', ylabel='Position')

        plt.suptitle(f"{self.years} years to maturity")
        plt.tight_layout()  # Add this line to adjust subplot parameters for better layout

        # Save plot to a directory
        save_dir = os.path.join(os.getcwd(), 'static', 'images', 'bond_trading')
        os.makedirs(save_dir, exist_ok=True)  # Create directory if it doesn't exist
        file_path = os.path.join(save_dir, 'recent_bond_graph.png')

        # Save the plot
        plt.savefig(file_path)
        plt.close()



        # plt.show()

    # returns annualized vol strictly on an array...not needed as method
    def annualized_volatility(self, array):
        # Calculate the logarithmic returns
        returns = np.diff(np.log(array))  # Calculate the logarithmic returns

        # Calculate the standard deviation of returns
        std_dev = np.std(returns)

        # Calculate the annualized volatility
        annualized_volatility = std_dev   # Assuming 252 trading days in a year

        return annualized_volatility

    # returns trade message used in display_round()
    def bond_trading_message(self, price, yld):
        if self.position > 0:
            open_profit = f"{(self.bond.price - self.get_avg_cost())*self.position*1000:,.2f}"
        elif self.position < 0:
            open_profit = f"{(self.get_avg_cost() - self.bond.price)*self.position*-1000:,.2f}"
        else:
            open_profit = '0'
        return (f"{self.position}",
                f"{self.get_avg_cost():.2f}",
                open_profit,
                f"{self.average_ytm * 100:,.2f}",
                f"{self.prices[-1]:.2f}",
                f"{self.yields[-1] * 100:.2f}",
                f"{self.years}",
                f"{self.volatility[-1] * 100:,.2f}",
                 f"{self.yield_volatility[-1] * 100:,.2f}")

    # in-place implements trade made by user
    def get_ready_for_next_period(self, trade):
        trade = 0 if len(trade) < 1 else int(trade)
        old_yld = self.bond.ytm

        if trade == 0:
            self.years -= 1
            # Generate a new random yield for the next period
            new_yield = old_yld + np.random.normal(loc=0, scale=0.01)
            # print(new_yield)
            self.bond = Bond(self.bond.par, self.cr, self.years,
                             self.bond.calculate_fair_price(new_yield, cr=self.cr, ttm=self.years))
            return None
        elif trade > 0:
            self.position += trade
            self.outflows += trade * self.bond.price * 1000
            self.years -= 1
            # print('outflow', trade * self.bond.price * 1000)
        else:
            self.position += trade
            self.inflows += abs(trade) * self.bond.price * 1000
            self.years -= 1
            # print('inflow', abs(trade) * self.bond.price * 1000)

        self.trades.append((trade, self.bond.price))
        self.transactions['t'+str(len(self.transactions.keys()))] = ('trade', f'buy {trade}', f'price: ${self.bond.price}', f'yield: {self.bond.ytm*100:.2f}%', f'${-1*self.bond.price*trade*1000:,.2f}')
        # Generate a new random yield for the next period
        new_yield = old_yld + np.random.normal(loc=0, scale=0.01)
        #re init self.bond
        self.bond = Bond(self.bond.par, self.cr, self.years,
                         self.bond.calculate_fair_price(new_yield, cr=self.cr, ttm=self.years))



    # in-place display of graph and new trade conditions, trade entry (if time left)
    def display_round(self, ind):
        price, yld = self.bond.price, self.bond.ytm
        prof = self.get_net_prof()
        if ind != 0:
            self.prices.append(price)
            self.yields.append(yld)
            self.volatility.append(self.annualized_volatility(self.prices))
            self.yield_volatility.append(self.annualized_volatility(self.yields))
            self.position_array.append(self.position)
            self.profits.append(prof)
            if self.position_array[-1] < 0:
                self.pos_bar_colors.append('r')
            else:
                self.pos_bar_colors.append('g')
        else:
            self.prices = [price]
            self.yields = [yld]
            self.volatility = [0]
            self.yield_volatility = [0]
            self.position_array = [self.position]
            self.profits = [0]
            if self.position_array[-1] < 0:
                self.pos_bar_colors.append('r')
            else:
                self.pos_bar_colors.append('g')
        if self.years == 0:
            self.prices[-1] = self.bond.par
            self.yields[-1] = self.bond.cr
            # self.inflows += self.position * 1000 * self.bond.par * self.bond.cr

        if self.position == 0:
            self.average_ytm = 0
        else:
            self.average_ytm = self.bond.calculate_ytm(years_to_mat=self.years, price=self.get_avg_cost())
            if self.years == 0:
                self.average_ytm = 0


        #
        # print('\n\ngraphing')
        # for x, y in self.to_dict().items():
        #     print(x, y)
        self.graph_bond_price()


        # self.next_year(price, yld)


    def next_year(self, price, yld):
        if self.years != 0:

            trade = self.get_trade()
            self.get_ready_for_next_period(trade)
            return self.bond_trading_message(price, yld)


    def trade_bond(self):
        for ind in range(self.bond.ttm+1):

            # each year (except last, cuz that's handled in get_realized()),
            # add inflow/outflow of actual coupon payment if any position


            self.display_round(ind)



def main():
    # Create an instance of Bond
    bond_instance = Bond(100, 0.075, 5, 100)
    BondTrading(bond_instance).trade_bond()

    # Call calculate_ytm method on the instance
    # print(bond_instance.ytm, f"${bond_instance.calculate_fair_price(.15):,}")


if __name__ == '__main__':
    main()
