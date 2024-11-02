import os

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor

class spy_vix_frame():
    def __init__(self):
        # reading in the dataset
        df = pd.read_csv("csv_files/good_spy_vix.csv")

        # removing rows of data where the observed temp is null
        df = df[df["spy"].notnull()]
        df = df[df["vix"].notnull()]
        df = df[df["date"].notnull()]
        df = df[df["sp20vol"].notnull()]

        # making a column for year: allows us to easily get the last 10 years
        df["year"] = df["date"].str[-4:]
        df["month_day"] = df["date"].str[:-5]

        df = df[['date', 'year', 'month_day', 'spy', 'vix', 'sp20vol']]

        self.years_list = list(df["year"].unique())
        self.month_list = ["J", "F", "M", "A", "M", "J", "J", "A", "S", "O", "N", "D"]
        self.long_month_list = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
        self.ref_dict = {'2007':0, '2008':1, '2009':2, '2010':3, '2011':4, '2012':5, '2013':6, '2014':7,
                         '2015':8, '2016':9, '2017':10, '2018':11, '2019':12, '2020':13, '2021':14, '2022':15,
                         '2023':16, '2024':17}

        # for 20-day 'vix', divide by sqrt(252/20)
        df["next_20_vix"] = df["vix"].div(12.6 ** .5)
        # take 20-day 'vix' and multiply by SPY to get next 20 day exp
        df["next_20_move"] = df["next_20_vix"].mul(df["spy"]).div(100)
        # diff between today and 20 days ago SPY
        df["last_20_move"] = df['spy'].shift(20).sub(df["spy"]).abs()
        # difference between implied and realized moves, pos is understated, neg is overstated
        df['imp_v_hist'] = df["last_20_move"].sub(df['next_20_move'].shift(20))
        # percentage
        df['vol_diff_p'] = df['imp_v_hist'].div(df['next_20_move'].shift(20)).mul(100)

        df['spy_50_ma'] = df['spy'].rolling(window=50).mean()
        df['spy_v_ma'] = df['spy'] / df['spy_50_ma']



        self.df = df[df['month_day'] != '02-29']  # drop leap years
        self.df.to_csv('csv_files/cleaned_spy_vix.csv', index=False)  # Export to CSV
        self.df_grouped_by_yr = self.df.groupby('year')




        self.is_understated = list(np.where(self.df['vol_diff_p'] > 0, True, False))
        self.understated = sum(self.is_understated)
        self.overstated = len(self.is_understated) - self.understated

        self.understated_p = self.understated / len(self.is_understated)
        self.overstated_p = self.overstated / len(self.is_understated)



    def knn_vix_bins(self, start='2007', years=None):
        if not years:
            years = 2024 - int(start)

        merged_df = self.get_years_group(start, years)[0]

        bins = [0, 10, 15, 20, 35, float('inf')]  # Bins: [0-10), [10-20), [20-35), [35-inf)

        # Labels for the bins
        labels = [0, 1, 2, 3, 4]

        bin_dict = {0:0, 1:10, 2:15, 3:20, 4:35}

        # Create a new column with the bin labels based on VIX readings
        merged_df['vix_bin'] = pd.cut(merged_df['vix'], bins=bins, labels=labels, right=False)

        features = merged_df[["spy", "sp20vol"]].values
        target = merged_df['vix_bin'].values

        X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.25, random_state=0)

        knn = KNeighborsClassifier(n_neighbors=5)
        knn.fit(X_train, y_train)

        y_pred = knn.predict(X_test)

        full_pred = knn.predict(features)

        bin_predictions = [bin_dict[x] for x in full_pred]
        # print(bin_predictions)

        return knn.score(X_test, y_test), bin_predictions

    def vix_reg(self, start='2007', years=None):
        # returns the average absolute percent difference between predicted VIX and real VIX numbers
        if not years:
            years = 2024 - int(start)

        merged_df = self.get_years_group(start, years)[0]

        merged_df.dropna(inplace=True)

        features = merged_df[["spy", "sp20vol", "spy_v_ma"]].values
        target = merged_df['vix'].values

        X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.25, random_state=0)

        knn = KNeighborsRegressor(n_neighbors=3)
        knn.fit(X_train, y_train)

        simple_test = knn.predict([[453.24, .4950, .68]])

        y_pred = knn.predict(X_test)

        full_pred = knn.predict(features)
        avg = []
        for i in range(len(y_pred)):
            avg.append(abs(y_pred[i] - y_test[i]) / y_test[i])
        return sum(avg) / len(avg), full_pred

    def knn_vix_understatement(self, start='2007', years=None):
        if not years:
            years = 2024 - int(start)

        merged_df = self.get_years_group(start, years)[0]

        merged_df['vix_discrep'] = merged_df['vol_diff_p'] > 0

        merged_df.dropna(inplace=True)

        features = merged_df[["spy", "vix", "spy_v_ma"]].values
        target = merged_df['vix_discrep'].values

        X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=0)

        knn = KNeighborsClassifier(n_neighbors=5)
        knn.fit(X_train, y_train)

        y_pred = knn.predict(X_test)

        full_pred = knn.predict(features)
        # print(full_pred)

        return knn.score(X_test, y_test), full_pred



    def get_years_group(self, start='2007', years=None):
        if not years:
            years = 2024 - int(start)

        # raw years (2007, 2008, 2009)
        list_of_years = self.years_list[self.ref_dict[start]:self.ref_dict[start] + years]
        # list of frames of each year
        frame_years = [self.df_grouped_by_yr.get_group(x) for x in list_of_years]
        # dataframe with year frames merged in
        merged_df = pd.concat(frame_years, ignore_index=True)

        return merged_df, list_of_years


    def graph_year_spy_vix(self, start='2007', years=None):

        if not years:
            years = 2024 - int(start)
        year_group = self.get_years_group(start, years)
        merged_df = year_group[0]
        list_of_years = year_group[1]

        knn_bin = self.knn_vix_bins(start, years)
        knn_bin_accuracy = knn_bin[0]
        knn_bin_preds = knn_bin[1]

        reg_vix = self.vix_reg(start, years)
        reg_vix_accuracy = reg_vix[0]
        reg_vix_preds = reg_vix[1]

        if start == '2007':
            diff1 = len(merged_df['date']) - len(knn_bin_preds)
            prep = [0] * diff1
            new_pred = prep
            for x in knn_bin_preds:
                new_pred.append(x)
            knn_bin_preds = new_pred


            diff2 = len(merged_df['date']) - len(reg_vix_preds)
            prep2 = [0] * diff2
            new_preds2 = prep2
            for x in reg_vix_preds:
                new_preds2.append(x)
            reg_vix_preds = new_preds2

        height_ratios = [4, 6]  # Bottom subplot is 60% (0.6) of the figure, top subplot is 40% (0.4)

        # Create the subplots with specified height ratios
        fig, ax = plt.subplots(2, 1, gridspec_kw={'height_ratios': height_ratios})


        # fig, ax = plt.subplots(2, 1)

        # 0
        ax[0].plot(merged_df['date'], merged_df['spy'], label=f'SPY ({start} - {int(start) + years})', color='green')
        ax[0].set(title='SPY Price', xlabel='Year', ylabel='SPY')
        ax[0].legend()

        ax[0].grid(True)

        # 1
        ax[1].plot(merged_df['date'], merged_df['vix'], label=f'VIX ({start} - {int(start) + years})', color='blue')
        ax[1].plot(merged_df['date'], merged_df['sp20vol'].mul(100),
                   label=f'Spy Trailing Vol ({start} - {int(start) + years})', color='red', alpha=0.5)
        ax[1].plot(merged_df['date'], knn_bin_preds, label=f'KNN VIX Region ({100*knn_bin_accuracy:,.2f}% accurate', color='orange', alpha=0.27)
        ax[1].plot(merged_df['date'], reg_vix_preds, label=f'KNN VIX ({100*reg_vix_accuracy:,.2f}% avg error) ', color='green', alpha=0.5)
        ax[1].set(title='VIX / Trailing Vol', xlabel='Year', ylabel='VIX price')
        # ax[1].legend()
        ax[1].legend(fontsize='small')

        ax[1].grid(True)

        if years < 10:
            if years > 2:
                ax[0].xaxis.set_ticks(np.arange(0, 251 * years, 251), list_of_years)
                ax[1].xaxis.set_ticks(np.arange(0, 251 * years, 251), list_of_years)
            elif years < 3:
                if years == 1:
                    ax[0].xaxis.set_ticks(np.arange(0, 251 * years, 251 // 12)[:-years], self.long_month_list * years)
                    ax[1].xaxis.set_ticks(np.arange(0, 251 * years, 251 // 12)[:-years], self.long_month_list * years)
                else:
                    ax[0].xaxis.set_ticks(np.arange(0, 251 * years, 251 // 12)[:-years], self.month_list * years)
                    ax[1].xaxis.set_ticks(np.arange(0, 251 * years, 251 // 12)[:-years], self.month_list * years)
        else:
            ax[0].xaxis.set_ticks(np.arange(0, 251 * years, 251 * 2), list_of_years[::2])
            ax[1].xaxis.set_ticks(np.arange(0, 251 * years, 251 * 2), list_of_years[::2])

        plt.tight_layout()

        directory = f'static/dynamic/images/spy_vix_stuff/yearly_charts'
        if not os.path.exists(directory):
            os.makedirs(directory)
            print('created directory')


        plt.savefig(f'static/dynamic/images/spy_vix_stuff/yearly_charts/{start}-{int(start)+years}prices.png')
        print('image saved', f'static/dynamic/images/spy_vix_stuff/yearly_charts/{start}-{int(start)+years}prices.png')

        plt.close()
        # plt.show()

    def graph_year_iv_disc(self, start='2007', years=None):


        if not years:
            years = 2024 - int(start)

        year_group = self.get_years_group(start, years)

        merged_df = year_group[0]
        list_of_years = year_group[1]

        knn_binary = self.knn_vix_understatement(start, years)
        accuracy = knn_binary[0]
        predictions = knn_binary[1]
        bars = []
        for p in predictions:
            if p:
                bars.append(100)
            else:
                bars.append(-100)

        if start == '2007':
            diff = len(merged_df['date']) - len(bars)
            prep = [0] * diff
            new_bars = prep
            for x in bars:
                new_bars.append(x)
            bars = new_bars

        fig, ax = plt.subplots(2, 1, figsize=(8, 6), gridspec_kw={'height_ratios': [6, 4]})  # 60% and 40%

        # 2
        colors = list(np.where(merged_df['vol_diff_p'] > 0, 'green', 'red'))
        under_over = list(map(lambda x: x == 'green', colors))
        ax[0].scatter(merged_df['date'], merged_df['vol_diff_p'], c=colors, s=5)
        ax[0].plot(merged_df['date'], merged_df['vol_diff_p'], label=f'vol diff ({start} - {int(start) + years})',
                   c='gray', alpha=.5)
        ax[0].bar(merged_df['date'], bars, color='blue', alpha=0.45, label=f'KNN under/overstatement ({100*accuracy:,.2f}% accurate)')
        ax[0].set(title='Realized/Implied Volatility', xlabel='Year', ylabel='Realized/Implied Volatility')
        ax[0].legend()
        ax[0].grid(True)

        understated = sum(under_over)
        understated_p = understated / len(under_over)
        overstated = len(under_over) - understated
        overstated_p = overstated / len(under_over)

        ax[1].bar("Understated", understated, color='green', label=f'{understated_p * 100:.2f}% understated')
        ax[1].bar("Overstated", overstated, color='red', label=f'{overstated_p * 100:.2f}% overstated')

        if years < 10:
            if years > 2:
                ax[0].xaxis.set_ticks(np.arange(0, 251 * years, 251), list_of_years)
            elif years < 3:
                if years == 1:
                    ax[0].xaxis.set_ticks(np.arange(0, 251 * years, 251 // 12)[:-years], self.long_month_list * years)
                else:
                    ax[0].xaxis.set_ticks(np.arange(0, 251 * years, 251 // 12)[:-years], self.long_month_list * years)
        else:
            ax[0].xaxis.set_ticks(np.arange(0, 251 * years, 251 * 2), list_of_years[::2])

        ax[1].set(title='Over/Under-stated Volatility', ylabel='Observations')
        ax[1].legend()

        plt.tight_layout()


        plt.savefig(f'static/dynamic/images/spy_vix_stuff/yearly_charts/{start}-{int(start)+years}volatility.png')
        print('image saved', f'static/dynamic/images/spy_vix_stuff/yearly_charts/{start}-{int(start)+years}volatility.png')

        plt.close()
        # plt.show()


def main():
    g = spy_vix_frame()
    # g.graph_year_spy_vix()
    # g.graph_year_iv_disc()
    # g.knn_vix()
if __name__ == '__main__':
    main()







