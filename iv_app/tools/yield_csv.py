import matplotlib.pyplot as plt
import pandas as pd
import os

class YieldCurve:
    def __init__(self, csv_file_path='tools/combined_data.csv'):


        # Read the CSV file into a DataFrame
        df = pd.read_csv(csv_file_path)

        # Convert the 'Date' column to datetime format
        df['Date'] = pd.to_datetime(df['Date'])

        # Extract year and month from the 'Date' column
        df['Year'] = df['Date'].dt.year
        df['Month'] = df['Date'].dt.month

        # Group by year and month, then select the first row of each group
        result_df = df.groupby(['Year', 'Month']).first().reset_index()

        result_df = result_df.fillna(0)
        self.df = result_df
        self.n90 = self.get_year_df('1990')
        self.n91 = self.get_year_df('1991')
        self.n92 = self.get_year_df('1992')
        self.n93 = self.get_year_df('1993')
        self.n94 = self.get_year_df('1994')
        self.n95 = self.get_year_df('1995')
        self.n96 = self.get_year_df('1996')
        self.n97 = self.get_year_df('1997')
        self.n98 = self.get_year_df('1998')
        self.n99 = self.get_year_df('1999')
        self.t00 = self.get_year_df('2000')
        self.t01 = self.get_year_df('2001')
        self.t02 = self.get_year_df('2002')
        self.t03 = self.get_year_df('2003')
        self.t04 = self.get_year_df('2004')
        self.t05 = self.get_year_df('2005')
        self.t06 = self.get_year_df('2006')
        self.t07 = self.get_year_df('2007')
        self.t08 = self.get_year_df('2008')
        self.t09 = self.get_year_df('2009')
        self.t10 = self.get_year_df('2010')
        self.t11 = self.get_year_df('2011')
        self.t12 = self.get_year_df('2012')
        self.t13 = self.get_year_df('2013')
        self.t14 = self.get_year_df('2014')
        self.t15 = self.get_year_df('2015')
        self.t16 = self.get_year_df('2016')
        self.t17 = self.get_year_df('2017')
        self.t18 = self.get_year_df('2018')
        self.t19 = self.get_year_df('2019')
        self.t20 = self.get_year_df('2020')
        self.t21 = self.get_year_df('2021')
        self.t22 = self.get_year_df('2022')
        self.t23 = self.get_year_df('2023')
        self.t24 = self.get_year_df('2024')

        self.maturities = ['3mo', '6mo', '1yr', '2yr', '3yr', '5yr', '7yr', '10yr', '30yr']

        self.long_month_list = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


    def get_year_df(self, year='2020'):
        values = self.df[self.df['Year'] == int(year)]
        return values

    def graph_now(self):


        df = self.t24

        fig, ax = plt.subplots(1, 1)

        filtered_row = df[df["Month"] == df["Month"].max()].values.tolist()[0]
        row_values = filtered_row

        ax.plot(self.maturities, row_values[3:], marker='.')
        ax.set(title=f"{self.long_month_list[row_values[1]-1]} 2024", ylabel='Yield %')

        for x, y in zip(self.maturities, row_values[3:]):
            ax.annotate(f'{x}', xy=(x, y), xytext=(x, y), fontsize=10)

        plt.tight_layout()

        # Save plot to a directory
        save_dir = os.path.join(os.getcwd(), 'static', 'images', 'yield_curve')
        os.makedirs(save_dir, exist_ok=True)  # Create directory if it doesn't exist
        file_path = os.path.join(save_dir, 'recent.png')

        # Save the plot
        plt.savefig(file_path)

        # Close the plot to release memory
        plt.close()
        # plt.show()


    def graph_curve_monthly(self, year='2020'):
        if int(year) >= 2000:
            name = 't' + year[-2:]
        else:
            name = 'n' + year[-2:]

        df = getattr(self, name)

        fig, axs = plt.subplots(4, 3, figsize=(12, 16))

        for i in range(1, 13):
            row = (i - 1) // 3  # Calculates the row index
            col = (i - 1) % 3
            month = self.long_month_list[i-1]
            filtered_row = df[df["Month"] == i]
            row_values = filtered_row.values.tolist()[0]

            axs[row,col].plot(self.maturities, row_values[3:], marker='.')
            axs[row,col].set(title=f"{month} {year}", ylabel='Yield %')

            for x, y in zip(self.maturities, row_values[3:]):
                axs[row, col].annotate(f'{x}', xy=(x, y), xytext=(x, y), fontsize=10)

        plt.tight_layout()

        # Save plot to a directory
        save_dir = os.path.join(os.getcwd(), 'static', 'images', 'yield_curve')
        os.makedirs(save_dir, exist_ok=True)  # Create directory if it doesn't exist
        file_path = os.path.join(save_dir, 'recent.png')

        # Save the plot
        plt.savefig(file_path)

        # Close the plot to release memory
        plt.close()

    def graph_curve_qtly(self, year='2020'):
        if int(year) >= 2000:
            name = 't' + year[-2:]
        else:
            name = 'n' + year[-2:]
        df = getattr(self, name)

        fig, axs = plt.subplots(2, 2, figsize=(8, 8))
        for i in range(3, 13, 3):
            if i == 3:
                row, col = 0, 0
            elif i == 6:
                row, col = 0, 1
            elif i == 9:
                row, col = 1, 0
            elif i == 12:
                row, col = 1, 1

            month = self.long_month_list[i - 1]

            filtered_row = df[df["Month"] == int(i)]

            row_values = filtered_row.values.tolist()[0]

            axs[row,col].plot(self.maturities, row_values[3:], marker='.')
            axs[row,col].set(title=f"{month} {year}", ylabel='Yield %')

            for x, y in zip(self.maturities, row_values[3:]):
                axs[row, col].annotate(f'{x}', xy=(x, y), xytext=(x, y), fontsize=10)

        plt.tight_layout()

        # Save plot to a directory
        save_dir = os.path.join(os.getcwd(), 'static', 'images', 'yield_curve')
        os.makedirs(save_dir, exist_ok=True)  # Create directory if it doesn't exist
        file_path = os.path.join(save_dir, 'recent.png')

        # Save the plot
        plt.savefig(file_path)

        # Close the plot to release memory
        plt.close()

YieldCurve().graph_now()