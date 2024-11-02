from flask import Flask, redirect, render_template, request, session, url_for
import os
import iv_app.tools.ticker_functionality as fcc
import sqlite3 as sl
from datetime import datetime
from iv_app.tools import black_scholes_sandbox as BS
# from iv_app.tools.earnings import Week
# from iv_app.tools.oi import OI_Week
# from iv_app.tools.mag7 import Mag7_Week
# from iv_app.tools.indices import Ind_Week


today = str(datetime.today()).split()[0]
day = today[-2:]
today = today[:-2] + str(int(day)-1)


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
                finance.html otherwise
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
        return render_template("finance.html", username=session['username'].upper())

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
            go.graphStPrices()
            go.graphBSPrice()
            go.graphDelta()
            go.graphTheta()
            go.graphGamma()
            go.graphVega()
            go.graphDailyCosts()
            go.graphWeeklyCosts()
            ag = [go.ag_sims_daily, go.ag_sims_weekly]
            if period == 'weekly':
                data = go.weekly_dict()
                total = data['total_cost']
            elif period == 'daily':
                data = go.daily_dict(85)
                total = data['total_cost']
            return render_template("bsm_sim.html", data=data, total=total, iv=iv, spot=spot,
                                   strike=strike, r=r, ag=ag)
        if 'x' in x:
            if x == 'x':
                session['iv'] = float(request.form.get('iv'))
                session['spot'] = float(request.form.get('spot'))
                session['strike'] = float(request.form.get('strike'))
                session['r'] = float(request.form.get('r'))
            if 'w' in x:
                session['period'] = 'weekly'
            if 'd' in x:
                session['period'] = 'daily'
        iv = session['iv']
        spot = session['spot']
        strike = session['strike']
        r = session['r']
        period = session['period']
        go = BS.black_scholes_sim(iv, spot, strike, r)
        go.graphStPrices()
        go.graphBSPrice()
        go.graphDelta()
        go.graphTheta()
        go.graphGamma()
        go.graphVega()
        go.graphDailyCosts()
        go.graphWeeklyCosts()
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
    print(criteria, type(criteria))
    if not session["logged_in"]:
        return redirect(url_for('home'))
    else:
        if criteria == 'earnings':
            earnings_list = [
                'PAYX', 'CAG', 'LW', 'TGT', 'ROST', 'CPB', 'KR', "AVGO", 'COST'
            ]
            Week(earnings_list)
            image_dir = '../static/dynamic/images/earnings'
            image_files = [f for f in os.listdir(image_dir) if os.path.isfile(os.path.join(image_dir, f))]
            message = "IV of This Week's Top Earnings"
        elif criteria == 'oi':
            oi_list = [
                'RDDT', 'SMCI', 'COIN', "SPY", 'QQQ'
            ]
            OI_Week(oi_list)
            image_dir = '../static/dynamic/images/oi'
            image_files = [f for f in os.listdir(image_dir) if os.path.isfile(os.path.join(image_dir, f))]
            message = "IV & Expected Move of Top Open Interest Underlyings"
        elif criteria == 'mag7':
            mag = [
                'AAPL', 'MSFT', 'META', 'AMZN', 'NVDA', "GOOG", 'TSLA'
            ]
            Mag7_Week(mag)
            image_dir = '../static/dynamic/images/mag7'
            image_files = [f for f in os.listdir(image_dir) if os.path.isfile(os.path.join(image_dir, f))]
            message = "IV & Expected Move of the Magnificent 7"
        elif criteria == 'indices':
            ind = [
                'SPY', 'QQQ', 'IWM', 'DIA'
            ]
            Ind_Week(ind)
            image_dir = '../static/dynamic/images/indices'
            image_files = [f for f in os.listdir(image_dir) if os.path.isfile(os.path.join(image_dir, f))]
            message = "IV & Expected Move of the Top Market Indices"


        return render_template('top.html', image_files=image_files,
                               criteria=criteria, message=message)




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
    Gets called from finance.html form submit
    Updates user food by calling db_set_food, then re-renders user template

    :return: redirects to home if user not logged in,
                re-renders finance.html otherwise
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
            # then reload client
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
