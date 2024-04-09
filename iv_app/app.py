from flask import Flask, redirect, render_template, request, session, url_for
import os
import iv_app.tools.ticker_functionality as fcc
from datetime import datetime
from iv_app.tools import black_scholes_sandbox as BS
from iv_app.tools.graphing_stock import StockSet


# globals (eek, I know) for today's date.....if any global is valid, its this
today = str(datetime.today()).split()[0]


# Initialize the Flask application
app = Flask(__name__)
app.secret_key = os.urandom(12)

with open('tickers_available', 'r') as fn:
    tickers_set = set([tick for tick in fn.read().split('\n') if tick])
bsm_sim_ag_prices_d = {}
bsm_sim_ag_prices_w = {}


# Define a route for the homepage
@app.route('/')
def home():
    """
    Checks whether the user is logged in and returns appropriately.

    :return: renders login.html if not logged in,
                redirects to client otherwise.
    """
    g = fcc.EarningStock('SPY')
    g.graph_iv()
    g.graphStPrices()
    g.graph_ticker_options()
    return redirect(url_for("client"))


# client endpoint
# renders appropriate template (admin or user)
@app.route("/client")
def client():
    """
    Renders appropriate template (admin or user)
    NOTE: Should only come to /client if /login'ed successfully, i.e. (db_check_creds() returned True)

    :return: redirects home if not logged in,
                renders admin.html if logged in as admin,
                user.html otherwise
    """
    return render_template("user.html")


@app.route("/ticker/<symbol>")
def ticker(symbol='SPY'):
    return render_template("ticker.html", ticker=symbol, ticker_name=fcc.EarningStock(symbol).name)


@app.route("/bsm")
def bsm_page():
    return render_template("bsm.html")


@app.route("/bsm/sim/<iv>/<spot>/<strike>/<r>/<dte>/<period>", methods=["GET", "POST"])
def bsm_sim(iv=None, spot=None, strike=None, r=None, dte=None, period=None):
    if iv == None:
        iv = .2
    if spot == None:
        spot = 100
    if strike == None:
        strike = 100
    if r == None:
        r = .05
    if dte == None:
        dte = 50
    if period == None:
        period = 'weekly'

    iv = float(iv) if is_float(iv) and float(iv) <= 1.3 else 0.2
    spot = float(spot) if is_float(spot) and float(spot) <= 1000 else 100.0
    strike = float(strike) if is_float(strike) and float(strike) <= 1000 else 100.0
    r = float(r) if is_float(r) and float(r) <= .5 else 0.05
    dte = float(dte) if is_float(dte) and float(dte) <= 120 else 50
    if 'weekly' in period:
        period = 'weekly'
    if 'daily' in period:
        period = 'daily'

    go = BS.black_scholes_sim(iv, spot, strike, r, dte)
    go.graphemall()
    sed = (iv, spot, strike, r, dte)
    # if sed not in bsm_sim_ag_prices_d.keys():
    #     ag = [go.ag_sims_daily(), go.ag_sims_weekly()]
    #     bsm_sim_ag_prices_d[sed] = ag[0]
    #     bsm_sim_ag_prices_w[sed] = ag[1]
    # else:
    #     ag = [bsm_sim_ag_prices_d[sed], bsm_sim_ag_prices_w[sed]]
    ag = [0, 0, 0, 0]
    if period == 'weekly':
        data = go.weekly_dict(dte)
        total = data['total_cost']
    elif period == 'daily':
        data = go.daily_dict(dte)
        total = data['total_cost']
    return render_template("bsm_sim.html", data=data, total=total, iv=iv, spot=spot,
                           strike=strike, r=r, dte=dte, period=period, ag=ag)

@app.route("/bsm/newparams", methods=["POST"])
def bsm_new_params():
    iv = request.form['iv']
    spot = request.form['spot']
    strike = request.form['strike']
    r = request.form['r']
    dte = request.form['dte']
    return redirect(url_for("bsm_sim", iv=f'{iv}', spot=f'{spot}', strike=f'{strike}', r=f'{r}',
                            dte=f'{dte}', period='weekly'))




@app.route('/top/<criteria>')
def top(criteria):
    if criteria == 'earnings':
        stocks = ['DAL', 'BLK', 'C', 'JPM', 'PGR', 'STT', 'WFC', 'FAST', 'KMX', 'STZ']
        StockSet(stocks, criteria)
        tail = "This Week's Top Earnings"
    elif criteria == 'oi':
        stocks = ['AAPL', 'NVDA', 'COIN', "SPY", 'QQQ', 'TSLA', 'AMZN', 'EEM', 'HYG']
        StockSet(stocks, criteria)
        tail = "Top Open Interest Underlyings"
    elif criteria == 'mag7':
        stocks = ['AAPL', 'MSFT', 'META', 'AMZN', 'NVDA', "GOOG", 'TSLA']
        StockSet(stocks, criteria)
        tail = "the Magnificent 7"
    else:
        criteria = 'indices'
        stocks = ['SPY', 'QQQ', 'IWM', 'DIA', 'TLT', 'GLD']
        StockSet(stocks, criteria)
        tail = "the Top Market Indices"

    # find directory with wanted images based on criteria
    image_dir = f'./static/images/{criteria}'
    #list the filenames
    image_files = [f for f in os.listdir(image_dir) if os.path.isfile(os.path.join(image_dir, f))]
    #create general message with personalized tail
    message = f'Past Month Price, IV & Expected Move of {tail}'

    return render_template('top.html', image_files=image_files,
                           criteria=criteria, message=message, today=today)




# set food endpoint (user only)
# updates user food, then re-renders user template
@app.route("/action/setticker", methods=["POST", "GET"])
def set_ticker():
    """
    Gets called from user.html form submit
    Updates user food by calling db_set_food, then re-renders user template

    :return: redirects to home if user not logged in,
                re-renders user.html otherwise
    """
    if not ticker_check(request.form['set_ticker'].upper()):
        #return redirect(url_for('ticker', symbol='SPY'))
        tick = 'SPY'
    else:
        tick = request.form['set_ticker'].upper()
    g = fcc.EarningStock(tick)
    g.graph_iv()
    g.graphStPrices()
    g.graph_ticker_options()
    # then reload ticker
    return redirect(url_for('ticker', symbol=tick))



def ticker_check(ticker):
    return ticker in tickers_set

def is_float(s):
    try:
        float(s)
        return True
    except ValueError:
        return False


# Run the Flask application
if __name__ == '__main__':
    app.run(debug=True)
