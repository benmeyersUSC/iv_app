import math
import numpy as np
import matplotlib.pyplot as plt
import os
from functools import reduce
from scipy.stats import norm



def norm_cdf(x):
    return (1 + math.erf(x / math.sqrt(2))) / 2

def n_prime(x):
    rat = 1/math.sqrt(2*math.pi)
    et = math.e ** ((-x)/2)
    return rat * et

class black_scholes_sim:
    def __init__(self, iv, spot, strike, r, dte):
        self.iv = float(iv)
        self.spot = float(spot)
        self.strike = float(strike)
        self.r = float(r)
        self.dte = int(float(dte))


        self.brownian_stock = self.brownianStock()

        self.daily_delta_array = self.daily_dict(self.dte)['delta']
        # self.weekly_delta_array = self.daily_dict(84)['delta']

        self.bs_price_array = self.daily_dict(self.dte)['bs_price']

        self.daily_theta_array = self.daily_dict(self.dte)['theta_prem']
        self.daily_gamma_array = self.daily_dict(self.dte)['gamma']
        self.daily_vega_array = self.daily_dict(self.dte)['vega']
        # self.daily_b_array = self.daily_dict(84)['b']

        # self.ag_sims_daily = self.ag_sims_daily()
        # self.ag_sims_weekly = self.ag_sims_weekly()

        self.total_costs_over_time_d = self.daily_dict(self.dte)['total']
        self.total_costs_over_time_w = self.weekly_dict(self.dte)['total']

        self.graphOption_list = [
            self.brownian_stock, self.bs_price_array, self.daily_delta_array, self.daily_theta_array,
            self.daily_vega_array, self.daily_gamma_array
        ]
        self.attributes_names = [
            f'GBM/Black-Scholes Random Stock Price\n𝛔: % {self.iv * 100:.2f}', f'Call Price\n𝛔: % {self.iv*100:.2f}',
            f'Delta\n𝛔: % {self.iv * 100:.2f}', f'Theta (time premium)\n𝛔: % {self.iv*100:.2f}',
            f'Vega\n𝛔: % {self.iv * 100:.2f}', f'Gamma\n𝛔: % {self.iv*100:.2f}'
        ]
        self.attribute_labels = [
            'Random Stock Price', 'Black-Scholes Call Price', 'Delta', 'Theta', 'Vega', 'Gamma'
        ]

    def graphemall(self):
        self.graphStPrices()
        self.graphBSPrice()
        self.graphDelta()
        self.graphTheta()
        self.graphGamma()
        self.graphVega()
        self.graphDailyCosts()
        self.graphWeeklyCosts()

    def graphOption(self):
        fig, ax = plt.subplots(2, 3)
        i = 0
        for row in range(2):
            for col in range(3):
                ax[row, col].plot(range(len(self.graphOption_list[i])), self.graphOption_list[i], marker='>')
                ax[row, col].set(title=self.attributes_names[i], xlabel='Days', ylabel=self.attribute_labels[i])
                i += 1
        fig.suptitle('Black Scholes Option Simulation')
        fig.tight_layout()
        plt.savefig('meh.png')
        plt.show()


    def ag_sims_daily(self, x=100):
        bs_price_start = self.bs_price_array[0]
        bs_total_exp = np.empty(x + 1)
        bs_total_exp[0] = self.daily_dict(self.dte)['total_cost']

        for run in range(1, x + 1):
            stock = self.brownianStock()
            info_dict = self.daily_dict(self.dte, stock)
            bs_total_exp[run] = info_dict['total_cost']

        avg_start = bs_price_start
        avg_total = np.mean(bs_total_exp)
        abs_diff = abs(avg_start - avg_total)
        abs_diff_p = abs_diff / avg_start

        return (avg_start, round(avg_total, 4), abs_diff, round(100 * abs_diff_p, 3))


    def ag_sims_weekly(self, x=100):
        bs_price_start = self.bs_price_array[0]
        bs_total_exp = np.empty(x + 1)
        bs_total_exp[0] = self.weekly_dict(self.dte)['total_cost']

        for run in range(1, x + 1):
            stock = self.brownianStock()
            info_dict = self.weekly_dict(self.dte, stock)
            bs_total_exp[run] = info_dict['total_cost']

        avg_start = bs_price_start
        avg_total = np.mean(bs_total_exp)
        abs_diff = abs(avg_start - avg_total)
        abs_diff_p = abs_diff / avg_start
                #avg_start,
        return (avg_start, round(avg_total, 4), abs_diff, round(100 * abs_diff_p, 3))




    def brownianStock(self) -> list:
        """
        Description: creates a BLACK-SCHOLES based random daily stock path (list)
        :return: (list) of length dte (params[3]), each a new daily stock price
        """
        days = int(self.dte)

        x = np.random.normal(1, .5)  # this makes volatility normal distributed around 1
        # (should be .85 theoretically)
        newVol = (self.iv * x)
        dailyVolatility = float(newVol / math.sqrt(365))
        dailyRiskFree = 1 + (float(self.r)-.01) / 365
        dailyPrices = [self.spot]
        for x in range(days):
            x = np.random.normal(1, self.iv)  # this makes volatility normal distributed around 1
            # (should be .85 theoretically)
            newVol = (self.iv * x)
            dailyVolatility = float(newVol / math.sqrt(365))
            dailyReturn = dailyRiskFree + dailyVolatility * np.random.normal(0, 1)
            nextPrice = dailyPrices[-1] * dailyReturn
            dailyPrices.append(round(nextPrice, 2))
        return dailyPrices

    def graphStPrices(self):
        xes = range(len(self.brownian_stock))


        if self.brownian_stock[-1] > self.brownian_stock[0]:
            color = 'g'
            colorr = 'r'
        else:
            color = 'r'
            colorr = 'g'

        plt.plot(xes, self.brownian_stock, label=f'$S Price', color=color, marker='>', linestyle='-')
        plt.plot(xes, [self.strike for _ in range(int(self.dte)+1)], label='Strike', color=colorr, linestyle='-', linewidth=5.4)



        plt.xlabel('Days From Now')
        plt.ylabel('Stock Price')
        plt.title(f'GBM/Black-Scholes Random Stock Price\nVolatility: % {self.iv*100:.2f}')

        plt.grid(True)

        plt.tight_layout()

        # Save plot to a directory
        save_dir = os.path.join(os.getcwd(), 'static', 'images', 'bsm_sim')
        os.makedirs(save_dir, exist_ok=True)  # Create directory if it doesn't exist
        file_path = os.path.join(save_dir, 'bsm_sim.png')

        # Save the plot
        plt.savefig(file_path)

        # Close the plot to release memory
        plt.close()


    def graphDelta(self):
        xes = range(len(self.daily_delta_array))

        if self.daily_delta_array[-2] > self.daily_delta_array[0]:
            color = 'g'
            colorr = 'r'
        else:
            color = 'r'
            colorr = 'g'

        plt.plot(xes, self.daily_delta_array, label=f'Delta', color=color, marker='>', linestyle='-')
        # plt.plot(xes, self.daily_b_array, label=f'Bt', color=colorr, marker='>', linestyle='-')



        plt.xlabel('Days From Now')
        plt.ylabel('Delta')
        plt.title(f'Delta\nVolatility: % {self.iv*100:.2f}')

        plt.grid(True)

        plt.tight_layout()

        # Save plot to a directory
        save_dir = os.path.join(os.getcwd(), 'static', 'images', 'bsm_sim')
        os.makedirs(save_dir, exist_ok=True)  # Create directory if it doesn't exist
        file_path = os.path.join(save_dir, 'bsm_sim_delta.png')

        # Save the plot
        plt.savefig(file_path)

        # Close the plot to release memory
        plt.close()

    def graphTheta(self):
        xes = range(len(self.daily_theta_array))[:-1]

        if self.daily_theta_array[-1] > self.daily_theta_array[0]:
            color = 'g'
            colorr = 'r'
        else:
            color = 'r'
            colorr = 'g'

        plt.plot(xes, self.daily_theta_array[:-1], label=f'Theta (time premium)', color=color, marker='>', linestyle='-')



        plt.xlabel('Days From Now')
        plt.ylabel('Theta (time premium)')
        plt.title(f'Theta (time premium)\nVolatility: % {self.iv*100:.2f}')

        plt.grid(True)

        plt.tight_layout()

        # Save plot to a directory
        save_dir = os.path.join(os.getcwd(), 'static', 'images', 'bsm_sim')
        os.makedirs(save_dir, exist_ok=True)  # Create directory if it doesn't exist
        file_path = os.path.join(save_dir, 'bsm_sim_theta.png')

        # Save the plot
        plt.savefig(file_path)

        # Close the plot to release memory
        plt.close()

    def graphGamma(self):
        xes = range(len(self.daily_gamma_array))[:-1]

        if self.daily_gamma_array[-2] > self.daily_gamma_array[0]:
            color = 'g'
            colorr = 'r'
        else:
            color = 'r'
            colorr = 'g'

        plt.plot(xes, self.daily_gamma_array[:-1], label=f'BS Price', color=color, marker='>', linestyle='-')



        plt.xlabel('Days From Now')
        plt.ylabel('Gamma')
        plt.title(f'Gamma\nVolatility: % {self.iv*100:.2f}')

        plt.grid(True)

        plt.tight_layout()

        # Save plot to a directory
        save_dir = os.path.join(os.getcwd(), 'static', 'images', 'bsm_sim')
        os.makedirs(save_dir, exist_ok=True)  # Create directory if it doesn't exist
        file_path = os.path.join(save_dir, 'bsm_sim_gamma.png')

        # Save the plot
        plt.savefig(file_path)

        # Close the plot to release memory
        plt.close()

    def graphVega(self):
        xes = range(len(self.daily_vega_array))[:-1]

        if self.daily_vega_array[-2] > self.daily_vega_array[0]:
            color = 'g'
            colorr = 'r'
        else:
            color = 'r'
            colorr = 'g'

        plt.plot(xes, self.daily_vega_array[:-1], label=f'Vega', color=color, marker='>', linestyle='-')



        plt.xlabel('Days From Now')
        plt.ylabel('Vega')
        plt.title(f'Vega\nVolatility: % {self.iv*100:.2f}')

        plt.grid(True)

        plt.tight_layout()

        # Save plot to a directory
        save_dir = os.path.join(os.getcwd(), 'static', 'images', 'bsm_sim')
        os.makedirs(save_dir, exist_ok=True)  # Create directory if it doesn't exist
        file_path = os.path.join(save_dir, 'bsm_sim_vega.png')

        # Save the plot
        plt.savefig(file_path)

        # Close the plot to release memory
        plt.close()

    def graphBSPrice(self):
        xes = range(len(self.bs_price_array))


        if self.bs_price_array[-1] > self.bs_price_array[0]:
            color = 'g'
            colorr = 'r'
        else:
            color = 'r'
            colorr = 'g'

        plt.plot(xes, self.bs_price_array, label=f'BS Call Price', color=color, marker='>', linestyle='-')



        plt.xlabel('Days From Now')
        plt.ylabel('BS Price')
        plt.title(f'Call Price\nVolatility: % {self.iv*100:.2f}')

        plt.grid(True)

        plt.tight_layout()

        # Save plot to a directory
        save_dir = os.path.join(os.getcwd(), 'static', 'images', 'bsm_sim')
        os.makedirs(save_dir, exist_ok=True)  # Create directory if it doesn't exist
        file_path = os.path.join(save_dir, 'bsm_sim_bs_price.png')

        # Save the plot
        plt.savefig(file_path)

        # Close the plot to release memory
        plt.close()

    def graphDailyCosts(self):
        xes = range(len(self.total_costs_over_time_d))

        colors = ['r' if cost > 0 else 'g' for cost in self.total_costs_over_time_d]
        # plt.plot(xes, self.total_costs_over_time_d, label=f'Rebalancing Costs', color='b', marker='>', linestyle='-')
        plt.bar(xes, self.total_costs_over_time_d, label='Rebalancing Costs', color=colors)

        plt.xlabel('Days')
        plt.ylabel('Rebalancing Costs')
        plt.title(f'Rebalacing Costs (Daily)\nVolatility: % {self.iv * 100:.2f}')

        plt.grid(True)

        plt.tight_layout()

        # Save plot to a directory
        save_dir = os.path.join(os.getcwd(), 'static', 'images', 'bsm_sim')
        os.makedirs(save_dir, exist_ok=True)  # Create directory if it doesn't exist
        file_path = os.path.join(save_dir, 'bsm_costs_d.png')

        # Save the plot
        plt.savefig(file_path)

        # Close the plot to release memory
        plt.close()


    def graphWeeklyCosts(self):
        xes = range(len(self.total_costs_over_time_w))

        colors = ['r' if cost > 0 else 'g' for cost in self.total_costs_over_time_w]

        # plt.plot(xes, self.total_costs_over_time_w, label=f'Rebalancing Costs', color='b', marker='>', linestyle='-')
        plt.bar(xes, self.total_costs_over_time_w, label='Rebalancing Costs', color=colors)

        plt.xlabel('Weeks')
        plt.ylabel('Rebalancing Costs')
        plt.title(f'Rebalacing Costs (Weekly)\nVolatility: % {self.iv * 100:.2f}')

        plt.grid(True)

        plt.tight_layout()

        # Save plot to a directory
        save_dir = os.path.join(os.getcwd(), 'static', 'images', 'bsm_sim')
        os.makedirs(save_dir, exist_ok=True)  # Create directory if it doesn't exist
        file_path = os.path.join(save_dir, 'bsm_costs_w.png')

        # Save the plot
        plt.savefig(file_path)

        # Close the plot to release memory
        plt.close()




    def black_scholes_call(self, S, dte):
        if dte > 0:
            K = self.strike
            r = self.r
            sigma = self.iv
            T = dte / 365
            # K*= math.e ** -(r*T)
            # print(K)
            d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
            # if S - K >3 and dte > 45:
                # print('d1', d1)
            d2 = d1 - sigma * math.sqrt(T)
            call_price = S * norm_cdf(d1) - K * math.exp(-r * T) * norm_cdf(d2)
            return call_price, norm_cdf(d1), norm_cdf(d2), d1, d2
        else:
            if S > self.strike:
                return (S - self.strike), 0, 0, 0, 0
            else:
                return 0, 0, 0, 0, 0


    def delta(self, spot, dte):

        return self.black_scholes_call(spot, dte)[1]

    def b(self, spot, dte):
        return self.black_scholes_call(spot, dte)[2]

    def theta(self, spot, dte):
        # print('dte', dte)
        top_n = spot * n_prime(self.black_scholes_call(spot, dte)[3]) * self.iv
        bot = 2 * math.sqrt(dte/365)
        rk = self.r * self.strike * norm.cdf(-self.black_scholes_call(spot, dte)[4])

        diff = self.black_scholes_call(spot, dte)[0] - self.black_scholes_call(spot, .01)[0]
        if bot != 0:
            return -1*(rk - (top_n/bot)), diff
        return 0, 0

    def gamma(self, spot, dte):
        top = norm.cdf(self.black_scholes_call(spot, dte)[3])
        bot = spot * self.iv * math.sqrt(dte/365)
        if bot != 0:
            return top/bot
        return 0

    def vega(self, spot, dte):

        return spot*math.sqrt(dte/365) * norm.cdf(self.black_scholes_call(spot, dte)[3])



    def daily_dict(self, dte, stock=None):

        # turn ['total'] into a bar graph to show the costs over the life of the trade

        if not stock:
            stock = self.brownian_stock
        info_dict = {'bs_price':[round(self.black_scholes_call(self.spot, dte)[0], 2)],
                     'til_exp':[x for x in range(int(dte))][::-1],
                                        #.5777        .5493
                     'stock_price':[], 'delta':[round(self.delta(self.spot, dte), 4)],
                     'b':[round(self.b(self.spot, dte), 4)],
                     'd-b':[round(100*(self.delta(self.spot, dte) - self.b(self.spot, dte)), 2)],
                     'theta':[round(self.theta(self.spot, dte)[0], 4)],
                     'gamma':[round(self.gamma(self.spot, dte), 4)],
                     'vega': [round(self.vega(self.spot, dte), 4)],
                     'theta_prem':[round(self.theta(self.spot, dte)[1], 4)],
                     'stock_costs':[], 'bond_costs':[], 'total':[]}


        for i in range(len(stock)):
            s = stock[i]
            k = self.strike
            dtee = dte - i
            info_dict['stock_price'].append(s)
            info_dict['delta'].append(round(self.delta(s, dtee), 4))
            info_dict['b'].append(round(self.b(s, dtee), 4))
            info_dict['theta'].append(round(self.theta(s, dtee)[0], 4))
            info_dict['theta_prem'].append(round(self.theta(s, dtee)[1], 4))
            info_dict['gamma'].append(round(self.gamma(s, dtee), 4))
            info_dict['vega'].append(round(self.vega(s, dtee), 4))
            info_dict['d-b'].append(round(100 * (info_dict['delta'][-1] - info_dict['b'][-1]), 2))

            if i != 0:
                # if dte == 0:
                #     print('delta, delta -2', info_dict['delta'][-2], info_dict['delta'][-3])
                #     print('s', s, 'k', k)
                #     print(info_dict['delta'])

                if dtee != 0:
                    info_dict['stock_costs'].append(
                        round(((info_dict['delta'][-1] - info_dict['delta'][-2])) * s, 4)
                    )
                    info_dict['bond_costs'].append(
                        round(((info_dict['b'][-1] - info_dict['b'][-2])) * -k, 2)
                    )
                    info_dict['bs_price'].append(round(self.black_scholes_call(s, dtee)[0], 2))
                else:
                    info_dict['stock_costs'].append(
                        round(((info_dict['delta'][-2] - info_dict['delta'][-3])) * s, 4)
                    )
                    info_dict['bond_costs'].append(
                        round(((info_dict['b'][-2] - info_dict['b'][-3])) * -k, 2)
                    )
                    info_dict['bs_price'].append(round(self.black_scholes_call(s, dtee)[0], 2))



            #     if i == 85:
            #         info_dict['stock_costs'][-1] = round(((info_dict['delta'][-1] - info_dict['delta'][-2])) * s, 4)
            else:
                info_dict['stock_costs'].append(
                    round((info_dict['delta'][0]) * s, 2)
                )
                info_dict['bond_costs'].append(
                    round(((info_dict['b'][0])) * -k, 2)
                )

            info_dict['total'].append(
                round(info_dict['stock_costs'][-1] + info_dict['bond_costs'][-1], 2)
            )


            info_dict['total_cost'] = round(reduce(lambda x,y: x+y, info_dict['total']), 2)
            info_dict['bs-real'] = round(abs(info_dict['bs_price'][0] - info_dict['total_cost']), 3)
            info_dict['bs-real_p'] = round(100*info_dict['bs-real'] / info_dict['bs_price'][0], 4)
        del info_dict['delta'][-1]

        if self.brownian_stock[-1] >= self.strike:
            info_dict['delta'][-1] = 1.0
        else:
            info_dict['delta'][-1] = 0.0
        return info_dict



    def weekly_dict(self, dte, stock=None):

        if not stock:
            stock = self.brownian_stock
        info_dict = {'bs_price':[round(self.black_scholes_call(self.spot, dte)[0], 2)],
                     'til_exp':[i for i in range(int(dte)//7)][::-1],
                     'stock_price':[], 'delta':[round(self.delta(self.spot, dte), 4)],
                     'b':[round(self.b(self.spot, dte), 4)],
                     'd-b':[round(100*(self.delta(self.spot, dte) - self.b(self.spot, dte)), 2)],
                     'stock_costs':[], 'bond_costs':[], 'total':[]}
        for i in range(len(stock)):
            if i in [0, 7, 14, 21, 28, 35, 42, 49, 56, 63, 70, 77, 84, 91, 98, 105, 112, 119, 126, 133]:
                s = stock[i]
                k = self.strike
                dtee = dte - i
                info_dict['stock_price'].append(s)
                info_dict['delta'].append(round(self.delta(s, dtee), 2))
                info_dict['b'].append(round(self.b(s, dtee), 2))
                info_dict['d-b'].append(round(100*(info_dict['delta'][-1] - info_dict['b'][-1]), 2))

                if i != 0:
                    # if dte == 0:
                    #     print('delta, delta -2', info_dict['delta'][-2], info_dict['delta'][-3])
                    #     print('s', s, 'k', k)
                    #     print(info_dict['delta'])
                    if dtee != 0:
                        info_dict['stock_costs'].append(
                            round(((info_dict['delta'][-1] - info_dict['delta'][-2])) * s, 2)
                        )
                        info_dict['bond_costs'].append(
                            round(((info_dict['b'][-1] - info_dict['b'][-2])) * -k, 2)
                        )
                        info_dict['bs_price'].append(round(self.black_scholes_call(self.spot, dtee)[0], 2))
                    else:
                        info_dict['stock_costs'].append(
                            round(((info_dict['delta'][-2] - info_dict['delta'][-3])) * s, 2)
                        )
                        info_dict['bond_costs'].append(
                            round(((info_dict['b'][-2] - info_dict['b'][-3])) * -k, 2)
                        )
                        info_dict['bs_price'].append(round(self.black_scholes_call(self.spot, dtee)[0], 2))
                else:
                    info_dict['stock_costs'].append(
                        round((info_dict['delta'][0]) * s, 2)
                    )
                    info_dict['bond_costs'].append(
                        round(((info_dict['b'][0])) * -k, 2)
                    )
                info_dict['total'].append(
                    round(info_dict['stock_costs'][-1] + info_dict['bond_costs'][-1], 2)
                )

                info_dict['total_cost'] = round(reduce(lambda x,y: x+y, info_dict['total']), 2)
                info_dict['bs-real'] = round(abs(info_dict['bs_price'][0] - info_dict['total_cost']), 3)
                info_dict['bs-real_p'] = round(100*info_dict['bs-real'] / info_dict['bs_price'][0], 4)
        del info_dict['delta'][-1]
        if self.brownian_stock[-1] >= self.strike:
            info_dict['delta'][-1] = 1.0
        else:
            info_dict['delta'][-1] = 0.0

        return info_dict


