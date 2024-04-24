import math
import random
from bs4 import BeautifulSoup as bs
import urllib.request
import ssl
import os
import numpy as np

# <div class="text-5xl/9 font-bold text-[#232526] md:text-[42px] md:leading-[60px] bg-positive-light" data-test="instrument-price-last">2,385.70</div>

# <span data-test="instrument-price-change">-28.05</span>
# <span data-test="instrument-price-change-percent">(-1.15%)</span>

# <h1 class="mb-2.5 text-left text-xl font-bold leading-7 text-[#232526] md:mb-2 md:text-3xl md:leading-8 rtl:soft-ltr">Gold Futures - Jun 24 (GCM4)</h1>

def get_security_context():
    # Ignore certificate errors
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    return ctx
def store_webpage(url, ctx, fn):
    page = urllib.request.urlopen(url, context=ctx)
    soup = bs(page.read(), 'html.parser')
    f = open(fn, 'w', encoding='utf-8')
    print(soup, file=f)
    f.close()
def load_webpage(url, ctx):
    page = urllib.request.urlopen(url, context=ctx)
    return bs(page.read(), 'html.parser')
def get_webpage(url, ctx, file_name):
    ctx = get_security_context()
    web_url = 'https://www.investing.com/commodities/gold'
    file_name = '../gold_futures.html'

    # only read page from net iff the file doesn't exist or is empty
    if not os.path.exists(file_name) or os.path.getsize(file_name) == 0:
        store_webpage(web_url, ctx, file_name)
    file_url = 'file:///' + os.path.abspath(file_name)

    # reading from a local file
    soup = load_webpage(file_url, ctx)
    return soup
    #
    # # Find all div tags with a data-test attribute of 'single-storyline'
    # story_wraps = soup.find_all('div', {'data-test': 'single-storyline'})
    #
    #
    #
    # story_wraps = soup.find_all('div', datatest="single-storyline")
    # for story in story_wraps:
    #     print(story)
    #     # print(story.h2.text)
    #     # print(story.a['href'])

class Future:
    def __init__(self, futures_price=157.0, spot_price=100.0, ttd=4.0, rfr=0.05):
        self.f_price = float(futures_price)
        self.s_price = float(spot_price)
        self.ttd = float(ttd)
        self.rfr = float(rfr)

        self.basis_dollars = self.get_basis_dollars()
        self.basis_percent = self.get_basis_percent()

        # self.annualized_basis = self.get_basis_annualized()

    def get_basis_dollars(self):
        return self.f_price - self.s_price

    def get_basis_percent(self):
        return self.f_price/self.s_price - 1

    def get_basis_annualized(self):
        raw_basis = self.get_basis_percent()
        amt_at_deliv = 1 + raw_basis

        if self.ttd < 1:
            whole_amt_1yr = raw_basis + amt_at_deliv ** (1 - self.ttd)

        else:
            whole_amt_1yr = amt_at_deliv ** (1/self.ttd)

        annualized_basis = whole_amt_1yr - 1
        return annualized_basis

    def get_risk_basis(self):
        return self.get_basis_annualized() - self.rfr

    def find_years_for_no_risk(self):
        diff = self.get_risk_basis()
        if diff < -.01:
            x = self.ttd + random.random() * abs(diff)
            print('start x', x)
        elif diff > .01:
            x = self.ttd + random.random() * -diff
            print('start x', x)
        else:
            return self.ttd
        error = 1
        runs = 0
        while abs(error) > 0.000001 and runs < 10000:
            f = Future(self.f_price, self.s_price, x, self.rfr)
            diff = f.get_risk_basis()
            if diff < -.0001:
                print('hi')
                x = f.ttd + random.random() * (abs(diff))
                print(x)
            elif diff > .0001:
                print('ho')
                x = f.ttd + random.random()* (abs(diff))
                print(x)
            error = f.get_risk_basis()
            runs += 1
        return x

    def years_risk_free(self):
        return math.log((1+self.basis_percent), (1+self.rfr))

    def __str__(self):
        return (f'F($):{f.f_price}, S($):{f.s_price}, yrs:{f.ttd:,.3f}, rfr:{100 * f.rfr:.2f}%'
                f'\nbasis {100 * f.get_basis_annualized():,.2f}%, risk {100 * f.get_risk_basis():,.2f}%')

t = 56/365
f = Future(futures_price=2619.75, spot_price=2306.31, ttd=t, rfr=.052)
print(f, '\n')

print(f.years_risk_free())

