import numpy as np
import matplotlib.pyplot as plt

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
        return round(pv, 2)

class BondTrading:
    def __init__(self, bond):
        self.bond = bond
        self.cr = bond.cr
        self.years = bond.ttm

        self.net_cost = 0
        self.average_ytm = 0
        # self.position = 0

        self.prices = [self.bond.price]
        self.yields = [self.bond.ytm]
        self.profits = [0]
        self.volatility = [0]
        self.yield_volatility = [0]
        self.position = []
        self.pos_bar_colors = []

        self.inflows = 0
        self.outflows = 0
        # self.rlz_prof = self.inflows - self.outflows

        self.trade_bond()

    def graph_bond_price(self):
        fig, ax = plt.subplots(2, 2, sharex=True)

        color = 'g' if self.prices[-1] > self.prices[0] else 'r'
        ax[0, 0].plot(self.prices, marker='*', color=color)
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


        ax[1, 1].bar(range(len(self.position)), self.position, color=self.pos_bar_colors)
        ax[1, 1].set(title=f"Position Size: {self.position[-1]}", xlabel='Years', ylabel='Position')

        plt.suptitle(f"{self.years} years to maturity")
        plt.tight_layout()  # Add this line to adjust subplot parameters for better layout
        plt.show()

    def annualized_volatility(self, array):
        # Calculate the logarithmic returns
        returns = np.diff(np.log(array))  # Calculate the logarithmic returns

        # Calculate the standard deviation of returns
        std_dev = np.std(returns)

        # Calculate the annualized volatility
        annualized_volatility = std_dev * np.sqrt(252)  # Assuming 252 trading days in a year

        return annualized_volatility

    # def trade_bond(self):
    #     trade_dict = {'trades':[]}
    #     for ind in range(self.bond.ttm):
    #         price, yld = self.bond.price, self.bond.ytm
    #         self.prices.append(price)
    #         self.yields.append(yld)
    #
    #         # this comprises 'open' profit
    #         position = sum([x[2] for x in trade_dict['trades']])
    #         avg_cost = abs(self.net_cost/position) if position != 0 else 0
    #
    #         if position > 0 or position < 0:
    #             profit = round(position * (price - avg_cost), 2)
    #         else:
    #             profit = 0
    #         self.profits.append(self.profits[-1] + profit + self.rlz_prof)
    #
    #         self.graph_bond_price()
    #
    #         print(f"\nPosition: {position} tranches, @ ${avg_cost:.2f}...paper profit = ${profit:,}")
    #         print(f"Bond Price: ${price:.2f}, yield: {yld*100:.2f}%, years to maturity: {self.years}")
    #         trade = int(input("Trade ('1', '-1, '0'): "))
    #         if trade > 0 or trade < 0:
    #             trade_dict['trades'].append((self.bond.ttm-ind, price, trade))
    #             self.years -= 1
    #             new_yield = self.bond.cr + np.random.normal(loc=0, scale=.015)
    #             self.bond = Bond(self.bond.par, self.cr, self.years, self.bond.calculate_fair_price(new_yield, cr=self.cr, ttm=self.years))
    #             self.net_cost += price * trade
    #             if trade > 0:
    #                 self.outflows += price * trade
    #             else:
    #                 self.inflows += price * abs(trade)
    #         else:
    #             new_yield = self.bond.cr + np.random.normal(loc=0, scale=.015)
    #             self.bond = Bond(self.bond.par, self.cr, self.years,
    #                              self.bond.calculate_fair_price(new_yield, cr=self.cr, ttm=self.years))
    #             self.net_cost += price * trade
    #             self.years -= 1

    def trade_bond(self):
        trade_dict = {'trades': []}
        avg_cost = 0  # Initialize average cost per open position
        position = 0  # Initialize position
        realized_profit = 0  # Initialize realized profit

        for ind in range(self.bond.ttm+1):
            price, yld = self.bond.price, self.bond.ytm
            self.prices.append(price)
            self.yields.append(yld)
            if self.years == 0:
                self.prices[-1] = self.bond.par
                self.yields[-1] = self.bond.cr
            self.volatility.append(self.annualized_volatility(self.prices))
            self.yield_volatility.append(self.annualized_volatility(self.yields))
            self.position.append(position)
            if self.position[-1] < 0:
                self.pos_bar_colors.append('r')
            else:
                self.pos_bar_colors.append('g')



            # Calculate open profit
            open_profit = position * (price - avg_cost)

            # Calculate total profit (realized profit + open profit)
            total_profit = realized_profit + open_profit
            self.profits.append(1000*total_profit)
            if self.years == 0:
                if position != 0:
                    self.profits[-1] += 100*(-1*position*avg_cost + (self.bond.par + self.bond.par*self.bond.cr) * position)
                    self.position[-1] = 0

            if position == 0:
                self.average_ytm = 0
            else:
                print('position is not 0')
                self.average_ytm = self.bond.calculate_ytm(years_to_mat=self.years, price=avg_cost)

            self.graph_bond_price()

            if self.years != 0:
                print(f"\nPosition: {position} tranches, @ ${avg_cost:.2f}...open profit = ${100*open_profit:,.2f}")
                print(f"Net lending (long)/borrowing (short): {self.average_ytm*100:,.2f}%")
                print(f"Bond Price: ${price:.2f}, yield: {yld * 100:.2f}%, years to maturity: {self.years}")
                print(f"Bond Volatility: {self.volatility[-1]*100:,.2f}%, Yield Volatility: {self.yield_volatility[-1]*100:,.2f}%")
                trade = input("\nTrade ('1', '-1, '0'): ")
                trade = 0 if len(trade) < 1 else int(trade)

                if trade != 0 or position != 0:  # Update realized profit whenever position changes
                    realized_profit += open_profit  # Update realized profit

                if trade != 0:
                    trade_dict['trades'].append((self.bond.ttm - ind, price, trade))
                    self.years -= 1
                    position += trade  # Update position
                    if position != 0:  # Update average cost per open position
                        avg_cost = (avg_cost * (position - trade) + price * trade) / position
                    else:
                        avg_cost = 0  # Reset average cost to zero when position is closed
                    self.net_cost += price * trade
                    if trade > 0:
                        self.outflows += price * trade
                    else:
                        self.inflows += price * abs(trade)
                else:
                    self.years -= 1

                # Generate a new random yield for the next period
                new_yield = self.cr + np.random.normal(loc=0, scale=0.005)
                self.bond = Bond(self.bond.par, self.cr, self.years,
                                 self.bond.calculate_fair_price(new_yield, cr=self.cr, ttm=self.years))






def main():
    # Create an instance of Bond
    bond_instance = Bond(100, 0.075, 5, 100)
    trade_game = BondTrading(bond_instance)

    # Call calculate_ytm method on the instance
    # print(bond_instance.ytm, f"${bond_instance.calculate_fair_price(.15):,}")


if __name__ == '__main__':
    main()
