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


@app.route("/ticker")
def ticker():
    user_ticker = session.get('ticker', 'SPY')
    return render_template("ticker.html",
                           ticker=user_ticker, ticker_name=fcc.EarningStock(user_ticker).name)


@app.route("/bsm")
def bsm_page():
    return render_template("bsm.html")



@app.route("/bsm/sim<x>", methods=["GET", "POST"])
def bsm_sim(x=None):
    if x == 'y':
        # y is first time arriving
        session['iv'] = .15
        session['spot'] = 50
        session['strike'] = 50
        session['r'] = .05
        session['period'] = 'weekly'
        iv = session['iv']
        spot = session['spot']
        strike = session['strike']
        r = session['r']
        period = session['period']

        go = BS.black_scholes_sim(iv, spot, strike, r)
        go.graphemall()
        # go.graphOption()
        ag = [go.ag_sims_daily, go.ag_sims_weekly]
        if period == 'weekly':
            data = go.weekly_dict()
            total = data['total_cost']
        elif period == 'daily':
            data = go.daily_dict(85)
            total = data['total_cost']
        return render_template("bsm_sim.html", data=data, total=total, iv=iv, spot=spot,
                               strike=strike, r=r, ag=ag)
    if x == 'x':
        # x new shitttt
        session['iv'] = float(request.form.get('iv'))
        session['spot'] = float(request.form.get('spot'))
        session['strike'] = float(request.form.get('strike'))
        session['r'] = float(request.form.get('r'))
    if 'w' == x:
        # w is changing to weekly
        session['period'] = 'weekly'
    if 'd' == x:
        # d is changing to daily
        session['period'] = 'daily'

    #otherwise itll be simr, which just rerenders

    iv = session['iv']
    spot = session['spot']
    strike = session['strike']
    r = session['r']
    period = session['period']
    go = BS.black_scholes_sim(iv, spot, strike, r)
    go.graphemall()
    ag = [go.ag_sims_daily, go.ag_sims_weekly]
    if period == 'weekly':
        data = go.weekly_dict()
        total = data['total_cost']
    elif period == 'daily':
        data = go.daily_dict(85)
        total = data['total_cost']
    return render_template("bsm_sim.html", data=data, total=total, iv=iv, spot=spot,
                           strike=strike, r=r, ag=ag)

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
        session['ticker'] = 'SPY'
        print('ticker check failed')
        return redirect(url_for('ticker'))
    else:
        session['ticker'] = request.form['set_ticker'].upper()
        print('ticker check succeed')
        print(session['ticker'])
        fcc.EarningStock(session['ticker']).graph_iv()
        fcc.EarningStock(session['ticker']).graphStPrices()
        # then reload ticker
        return redirect(url_for('ticker'))



def ticker_check(ticker):
    return ticker in tickers_set


# Run the Flask application
if __name__ == '__main__':
    app.run(debug=True)
