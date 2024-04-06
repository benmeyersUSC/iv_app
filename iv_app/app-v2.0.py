from flask import Flask, redirect, render_template, request, session, url_for
import os
import iv_app.tools.ticker_functionality as fcc
import sqlite3 as sl
from datetime import datetime
from iv_app.tools import black_scholes_sandbox as BS
from iv_app.tools.graphing_stock import StockSet


# globals (eek, I know) for today's date.....if any global is valid, its this
today = str(datetime.today()).split()[0]
# day = today[-2:]
# today = today[:-2] + str(int(day)-1)
# print(today)


# Initialize the Flask application
app = Flask(__name__)
db = "iv_app.db"



# Define a route for the homepage
@app.route('/')
def home():
    """
    Checks whether the user is logged in and returns appropriately.

    :return: renders login.html if not logged in,
                redirects to client otherwise.
    """
    # checking to see if logged_in is even in session dict yet (wasn't for the first time)
    if 'logged_in' not in session.keys():
        # send to login
        return render_template("login.html", message='Please log in to continue')
    else:
        # if not logged in
        if not session["logged_in"]:
            # log em in
            return render_template("login.html", message='Please log in to continue')
        else:
            # otherwise, send em to client
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
    # if logged out
    if not session["logged_in"]:
        # send em home, which will then send them to login cuz they're not
        return redirect(url_for("home"))
    elif session["username"] == 'admin':
        # if logged in as admin, send to admin page
        return render_template('admin.html',
                               username=session['username'].upper(),
                            message='Welcome back!', result=db_get_user_list())
    else:
        # otherwise, the user page
        return render_template("user.html", username=session['username'].upper())


@app.route("/ticker")
def ticker():
    if not session["logged_in"]:
        return redirect(url_for('home'))
    else:
        return render_template("ticker.html", username=session['username'].upper(),
                               ticker=session['ticker'], ticker_name=fcc.EarningStock(session['ticker']).name)


@app.route("/bsm")
def bsm_page():
    if not session["logged_in"]:
        return redirect(url_for('home'))
    else:
        return render_template("bsm.html")



@app.route("/bsm/sim<x>", methods=["GET", "POST"])
def bsm_sim(x=None):
    if not session["logged_in"]:
        return redirect(url_for('home'))
    else:
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
    if not session["logged_in"]:
        return redirect(url_for('home'))
    else:
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




# create user endpoint (admin only)
# adds new user to db, then re-renders admin template
@app.route("/action/createuser", methods=["POST", "GET"])
def create_user():
    """
    Gets called from admin.html form submit
    Adds a new user to db by calling db_create_user, then re-renders admin template

    :return: redirects to home if user not logged in,
                re-renders admin.html otherwise
    """
    if not session["logged_in"]:
        return redirect(url_for('home'))
    else:
        # call create_user
        db_create_user(request.form['username'], request.form['password'])
        # then reload client
        return redirect(url_for('client'))


# remove user endpoint (admin only)
# removes user from db, then re-renders admin template
@app.route("/action/removeuser", methods=["POST", "GET"])
def remove_user():
    """
    Gets called from admin.html form submit
    Removes user from the db by calling db_remove_user, then re-renders admin template.

    :return: redirects to home if user not logged in,
                re-renders admin.html otherwise
    """
    if not session["logged_in"]:
        return redirect(url_for('home'))
    else:
        # call remove_user
        db_remove_user(request.form['username'])
        # then reload client
        return redirect(url_for('client'))

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
    if not session["logged_in"]:
        return redirect(url_for('home'))
    else:
        if not ticker_check(request.form['set_ticker'].upper()):
            return redirect(url_for('ticker'))
        else:
            session['ticker'] = request.form['set_ticker'].upper()
            fcc.EarningStock(session['ticker']).graph_iv()
            fcc.EarningStock(session['ticker']).graphStPrices()
            # then reload ticker
            return redirect(url_for('ticker'))


# login endpoint
# allows client to log in (checks creds)
@app.route("/login", methods=["POST", "GET"])
def login():
    """
    Allows client to log in
    Calls db_check_creds to see if supplied username and password are correct

    :return: redirects to client if login correct,
                redirects back to home otherwise
    """
    if request.method == "POST":
        # check_creds is boolean
        if db_check_creds(request.form["username"],
                              request.form["password"]):
            # create k:v for username in session
            session["username"] = request.form["username"]
            # create k:v for logged_in in session
            session["logged_in"] = True
    # then send em home, which will pass them through to client or admin :)
    return redirect(url_for("home"))


# logout endpoint
@app.route("/logout", methods=["POST", "GET"])
def logout():
    """
    Logs client out, then routes to login
    Remove the user from the session
    :return: redirects back to home
    """
    if request.method == "POST":
        # change logged_in status
        session["logged_in"] = False
        session.pop("username", None)
    # then send home, logged out
    session["logged_in"] = False
    return redirect(url_for("home"))


def db_get_user_list() -> dict:
    """
    Queries the DB's userfoods table to get a list
    of all the user and their corresponding favorite food for display on admin.html.
    Called to render admin.html template.

    :return: a dictionary with username as key and their favorite food as value
                thblacis is what populates the 'result' variable in the admin.html template
    """
    info_dict = {}
    conn = sl.connect(db)
    curs = conn.cursor()
    stmt = "SELECT * FROM credentials"
    results = curs.execute(stmt)
    for result in results:
        info_dict[result[0]] = result[1]
    conn.close()
    return info_dict



def db_create_user(un: str, pw: str) -> None:
    """
    Add provided user and password to the credentials table
    Add provided user to the userfoods table
    and sets their favorite food to "not set yet".
    Called from create_user() view function.

    :param un: username to create
    :param pw: password to create
    :return: None
    """
    if un == '' or pw == '':
        return
    conn = sl.connect(db)
    curs = conn.cursor()
    v = (un, pw,)
    stmt = "INSERT INTO credentials (username, password) VALUES (?, ?)"
    curs.execute(stmt, v)

    conn.commit()
    conn.close()



def db_remove_user(un: str) -> None:
    """
    Removes provided user from all DB tables.
    Called from remove_user() view function.

    :param un: username to remove from DB
    :return: None
    """
    conn = sl.connect(db)
    curs = conn.cursor()
    v = (un,)
    stmt = "DELETE FROM credentials WHERE username=?"
    curs.execute(stmt, v)


    conn.commit()
    conn.close()


# database function
# connects to db and checks cred param (all clients)
def db_check_creds(un, pw):
    """
    Checks to see if supplied username and password are in the DB's credentials table.
    Called from login() view function.

    :param un: username to check
    :param pw: password to check
    :return: True if both username and password are correct, False otherwise.
    """
    conn = sl.connect(db)
    curs = conn.cursor()
    v = (un,)
    stmt = "SELECT password FROM credentials WHERE username=?"
    results = curs.execute(stmt, v)  # results is an iterable cursor of tuples (result set)
                                     # (password,)
    correct_pw = ''
    for result in results:
        correct_pw = result[0]  # each result is a tuple of 1, so grab the first thing in it
    conn.close()
    if correct_pw == pw and pw != '':
        return True
    return False

def ticker_check(ticker):
    conn = sl.connect(db)
    curs = conn.cursor()
    v = (ticker,)
    stmt = "SELECT * FROM tickers WHERE ticker=?"
    results = curs.execute(stmt, v)  # results is an iterable cursor of tuples (result set)
    results = list(results)
    conn.close()
    if len(results) > 0:
        return True
    return False


# Run the Flask application
if __name__ == '__main__':
    app.secret_key = os.urandom(12)
    app.run(debug=True)
