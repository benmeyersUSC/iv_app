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
    session['ticker'] = 'SPY'
    fcc.EarningStock(session['ticker']).graph_iv()
    fcc.EarningStock(session['ticker']).graphStPrices()
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
    session['ticker'] = 'SPY'
    return render_template("user.html")


@app.route("/ticker/<symbol>")
def ticker(symbol='SPY'):
    return render_template("ticker.html", ticker=symbol, ticker_name=fcc.EarningStock(symbol).name)


@app.route("/bsm")
def bsm_page():
    return render_template("bsm.html")


@app.route("/bsm/sim/<iv>/<spot>/<strike>/<r>/<period>", methods=["GET", "POST"])
def bsm_sim(iv=None, spot=None, strike=None, r=None, period=None):
    if iv == None:
        iv = .15
    if spot == None:
        spot = 50
    if strike == None:
        strike = 50
    if r == None:
        r = .05
    if period == None:
        period = 'weekly'

    iv = float(iv) if is_float(iv) else 0.15
    spot = float(spot) if is_float(spot) else 50.0
    strike = float(strike) if is_float(strike) else 50.0
    r = float(r) if is_float(r) else 0.05
    if 'weekly' in period:
        period = 'weekly'
    if 'daily' in period:
        period = 'daily'

    go = BS.black_scholes_sim(iv, spot, strike, r)
    go.graphemall()
    sed = (iv, spot, strike, r)
    if sed not in bsm_sim_ag_prices_d.keys():
        ag = [go.ag_sims_daily(), go.ag_sims_weekly()]
        bsm_sim_ag_prices_d[(iv, spot, strike, r)] = ag[0]
        bsm_sim_ag_prices_w[(iv, spot, strike, r)] = ag[1]
    else:
        ag = [bsm_sim_ag_prices_d[(iv, spot, strike, r)], bsm_sim_ag_prices_w[(iv, spot, strike, r)]]
    if period == 'weekly':
        data = go.weekly_dict()
        total = data['total_cost']
    elif period == 'daily':
        data = go.daily_dict(85)
        total = data['total_cost']
    return render_template("bsm_sim.html", data=data, total=total, iv=iv, spot=spot,
                           strike=strike, r=r, period=period, ag=ag)



@app.route('/top/<criteria>')
def top(criteria):
    if criteria == 'earnings':
        stocks = ['DAL', 'BLK', 'C', 'JPM', 'PGR', 'STT', 'WFC', 'FAST', 'KMX', 'STZ']
        StockSet(stocks, criteria)
        tail = "This Week's Top Earnings"
    elif criteria == 'oi':
        stocks = ['RDDT', 'SMCI', 'COIN', "SPY", 'QQQ']
        StockSet(stocks, criteria)
        tail = "Top Open Interest Underlyings"
    elif criteria == 'mag7':
        stocks = ['AAPL', 'MSFT', 'META', 'AMZN', 'NVDA', "GOOG", 'TSLA']
        StockSet(stocks, criteria)
        tail = "the Magnificent 7"
    else:
        criteria = 'indices'
        stocks = ['SPY', 'QQQ', 'IWM', 'DIA', 'GLD']
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
        return redirect(url_for('ticker'))
    else:
        tick = request.form['set_ticker'].upper()
        fcc.EarningStock(tick).graph_iv()
        fcc.EarningStock(tick).graphStPrices()
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
