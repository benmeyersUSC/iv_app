import numpy as np
from flask import Flask, redirect, render_template, request, session, url_for, jsonify
import os
import random
from matplotlib import pyplot as plt
import iv_app.tools.ticker_functionality as fcc
from datetime import datetime
from iv_app.tools import black_scholes_sandbox as BS
from iv_app.tools.graphing_stock import StockSet
from iv_app.tools import spy_vix_unpacker as spv
from iv_app.tools import yield_csv as yld
from iv_app.tools import Bond as bnd
from iv_app.tools.lambdaCalculus import lambda_calculus_interpreter as LI
import iv_app.tools.waves.PrimitiveFT as ft
from iv_app.tools.NeuralNetworks.MNIST_NN import MNIST_NN_3layer as mnn
import iv_app.tools.turing.TuringMachine as tm
import json


file_path = 'starters.json'
if os.path.exists(file_path):
    mode = 'w'  # If file exists, overwrite
else:
    mode = 'x'  # If file doesn't exist, create

# Write JSON string to the file
with open(file_path, mode) as json_file:
    json_file.write('')



file_path = 'session_repl.json'
if os.path.exists(file_path):
    mode = 'w'  # If file exists, overwrite
else:
    mode = 'x'  # If file doesn't exist, create

# Write JSON string to the file
with open(file_path, mode) as json_file:
    json_file.write('')





# globals (eek, I know) for today's date.....if any global is valid, its this
today = str(datetime.today()).split()[0]


# Initialize the Flask application
app = Flask(__name__)
app.secret_key = os.urandom(12)

with open('tickers_available', 'r') as fn:
    tickers_set = set([tick for tick in fn.read().split('\n') if tick])
# bsm_sim_ag_prices_d = {}
# bsm_sim_ag_prices_w = {}



# Error handling for 404 Not Found
@app.errorhandler(404)
def page_not_found(e):
    print('not found')
    return redirect(url_for('home'))

# Error handling for Internal Server Error (500)
@app.errorhandler(500)
def internal_server_error(e):
    print('internal server error')
    return redirect(url_for('home'))


# Define a route for the homepage
@app.route('/')
def home():
    """
    Checks whether the user is logged in and returns appropriately.

    :return: renders login.html if not logged in,
                redirects to client otherwise.
    """

    return redirect(url_for("client"))


# finance endpoint
# renders appropriate template (admin or user)
@app.route("/finance")
def finance():
    g = fcc.EarningStock('SPY')
    g.graph_iv()
    g.graphStPrices()
    g.graph_ticker_options()
    return render_template("finance.html")



# lambdaCalculus calc endpoint
# renders appropriate template (admin or user)
@app.route("/lambda", methods=["POST", "GET"])
def lambdaCalc():

    return render_template("lambdaCalc.html")

# lambdaCalculus what is endpong
# renders appropriate template (admin or user)
@app.route("/whatIsLambda", methods=["POST", "GET"])
def whatIsLambdaCalc():

    return render_template("whatIsLambda.html")

# lambdaCalculus calc endpoint
# renders appropriate template (admin or user)
@app.route("/runLambda", methods=["POST", "GET"])
def runLambdaCalc():
    code = request.form.get('inputText')
    if code == None:
        with open("./tools/lambdaCalculus/userStarter.lambda", 'r') as fn:
            code = fn.read()
        return render_template("lambdaCalc.html", starter_code=code)

    with open('./tools/lambdaCalculus/userCode.lambda', 'w') as fn:
        fn.write(code)
    exprs, evaluated = LI.lambda_interpret_file_viz(filename='./tools/lambdaCalculus/userCode.lambda')
    with open("./tools/lambdaCalculus/userCode_parsed_tree.txt", 'r') as fn:
        tree_text = fn.read()
    program = ">>" + "\n>>".join([LI.format_church_numeral_output(str(evaluated[x])) for x in range(len(evaluated))])
    return render_template("lambdaCalc.html", starter_code=code,
                           output_text=program,
                           file_content=tree_text)



# neural networks endpoint
# renders appropriate template (admin or user)
@app.route("/neural")
def neural():
    return render_template("neural.html")

@app.route("/mnistDraw")
def mnistDraw():
    return render_template("mnist_draw.html")



@app.route('/mnistPredict', methods=['POST'])
def mnistPredict():
    data = request.json['image']
    image = np.array(data, dtype=np.float32)
    image = image.reshape(1, 784) * 255
    image = np.array(image, dtype=np.int64)
    # print(image)
    # plt.imshow(image.reshape(28, 28), cmap='gray')
    # plt.show()

    nn = mnn.NeuralNetwork_MNIST(use_pretrained=True, json_file='tools/NeuralNetworks/MNIST_NN/three_layer_MNIST_W&B.json')
    pred = nn.predict(image)
    prediction = pred[0]
    dist = pred[1]
    # print("Prediction:", prediction)

    return jsonify({
        'prediction': int([prediction][0]),
        'distr': float([dist][0])
    })







# turing  endpoint
# renders appropriate template (admin or user)
@app.route("/turing")
def turing():
    basic_prog = ("q1 - S0 - S2 - R - q1;\n\n\n#########\nSCAFF_TURING_PROGRAM\n\nThese are comments, keep them below the 9 hashmarks."
                  "\nThis program is very simple, it will only move to the right and print 1s until the end...\n"
                  "Click one of the above options for a present program which will do some cooler things!")

    with open("./tools/turing/art.javaturing", 'r') as fn:
        art = fn.read().strip()

    with open("./tools/turing/sqrt2.javaturing", 'r') as fn:
        sqrt2 = fn.read().strip()

    with open("./tools/turing/isOdd.javaturing", 'r') as fn:
        isOdd = fn.read().strip()

    with open("./tools/turing/counting.javaturing", 'r') as fn:
        counting = fn.read().strip()

    with open("./tools/turing/doubling.javaturing", 'r') as fn:
        doubling = fn.read().strip()
    return render_template("turing.html", starter_code=basic_prog, basic_program=basic_prog,
                           art=art, isOdd=isOdd, counting=counting, doubling=doubling, sqrt2=sqrt2, show_uns=False)

@app.route("/runTuring", methods=["GET", "POST"])
def runTuring():
    basic_prog = ("q1 - S0 - S2 - R - q1;\n\n\n#########\n\nThese are comments, keep them below the 9 hashmarks."
                  "\nThis program is very simple, it will only move to the right and print 1s until the end...\n"
                  "Click one of the above options for a present program which will do some cooler things!")

    user_prog = request.form.get('inputText')

    show_uns = not ("ART_TURING_PROGRAM" in user_prog or "SCAFF_TURING_PROGRAM" in user_prog)

    sl = 2727 if "SQRT_2_TURING_PROGRAM" not in user_prog else 216
    machine = tm.TuringMachine(tm.Tape(), description=user_prog, sizeLimit=sl)
    theRun = machine.run(saveFirst=101)

    unary = machine.printUnary(tape=False) if "DOUBLING_TURING_PROGRAM" in user_prog or "COUNTING_TURING_PROGRAM" in user_prog else machine.getTape()

    with open("./tools/turing/art.javaturing", 'r') as fn:
        art = fn.read().strip()

    with open("./tools/turing/sqrt2.javaturing", 'r') as fn:
        sqrt2 = fn.read().strip()

    with open("./tools/turing/isOdd.javaturing", 'r') as fn:
        isOdd = fn.read().strip()

    with open("./tools/turing/counting.javaturing", 'r') as fn:
        counting = fn.read().strip()

    with open("./tools/turing/doubling.javaturing", 'r') as fn:
        doubling = fn.read().strip()



    return render_template("turing.html", output_text=unary, file_content=theRun,
                           starter_code=user_prog, basic_program=basic_prog,
                           art=art, isOdd=isOdd, counting=counting, doubling=doubling, sqrt2=sqrt2,
                           show_uns=show_uns)




# sound waves endpoint
# renders appropriate template (admin or user)
@app.route("/waves", methods=["GET", "POST"])
def waves():
    return render_template("waves.html", show_image=False, show_descr=True)

# sound waves endpoint
# renders appropriate template (admin or user)
@app.route("/runWaves", methods=["GET", "POST"])
def runWaves():
    frm = request.form
    numFreqs = int(frm["numFreqs"])
    freqs = []
    for i in range(1, 6):
        if f"freq{i}" in frm:
            freqs.append(int(frm[f"freq{i}"]))


    ft.runNewWave(numFreqs, freqs)
    return render_template("waves.html", show_image=True, last_value=numFreqs,
                           show_descr=False)



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
    return render_template("main.html")


@app.route("/ticker/<symbol>")
def ticker(symbol='SPY'):
    object = fcc.EarningStock(symbol)
    object.graph_iv()
    object.graphStPrices()
    object.graph_ticker_options()
    return render_template("ticker.html", ticker=symbol, ticker_name=object.name)


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
    elif 'daily' in period:
        period = 'daily'
    else:
        period = 'weekly'


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
        stocks = ['PLTR', 'DIS', 'LYFT', 'UBER', 'OXY', 'RIVN', 'MARA', 'SHOP', 'ABNB', 'HOOD']
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
    elif criteria == 'reddit':
        stocks = ['TLRY', 'AMZN', 'SPY', 'NVDA', 'PYPL', 'AAPL', 'MSOS', 'AMD']
        StockSet(stocks, criteria)
        tail = "r/WallStBets most mentioned stocks"
    elif criteria == 'macro':
        stocks = ['GLD', 'USO', 'UNG', 'SLV', 'DBA', 'TLT', 'IEF', 'SHY', 'BND', 'VTI', 'BITO']
        StockSet(stocks, criteria)
        tail = "major commodity/macroeconomic indices"
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
    Gets called from finance.html form submit
    Updates user food by calling db_set_food, then re-renders user template

    :return: redirects to home if user not logged in,
                re-renders finance.html otherwise
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

@app.route("/ticker/redirect", methods=["POST", "GET"])
def ticker_redirect():
    ticker = request.form['newtick']
    return redirect(url_for("ticker", symbol=ticker))

@app.route("/spyvixhome", methods=["POST", "GET"])
def spyvixhome():
    return render_template('spy_vix_home.html')

@app.route("/spyvix/<year>", methods=["POST", "GET"])
def spyvix(year):
    years = 1
    load = spv.spy_vix_frame()
    load.graph_year_spy_vix(start=year, years=years)
    load.graph_year_iv_disc(start=year, years=years)
    # pred = round(load.knn_vix_bins(start=year, years=1)[0] * 100, 2)
    # pred2 = round(load.knn_vix_understatement(start=year, years=1)[0] * 100, 2)
    # pred3 = round(load.vix_reg(start=year, years=1)[0] * 100, 2)
    year = str(year)+'-'+str(int(year)+years)
    # , pred = pred, pred2 = pred2, pred3 = pred3
    return render_template('spy_vix_year.html', year=year)

@app.route("/spyvix/range<years>/<year>", methods=["POST", "GET"])
def spyvixrange(years, year):
    years = int(years)
    if int(year) + years > 2024:
        years = 2024 - int(year)
    load = spv.spy_vix_frame()
    load.graph_year_spy_vix(start=year, years=years)
    load.graph_year_iv_disc(start=year, years=years)
    # pred = round(load.knn_vix_bins(start=year, years=years)[0]*100, 2)
    # pred2 = round(load.knn_vix_understatement(start=year, years=years)[0] * 100, 2)
    # pred3 = round(load.vix_reg(start=year, years=years)[0] * 100, 2)
    #
    # , pred = pred, pred2 = pred2,
    # pred3 = pred3
    return render_template('spy_vix_year.html', year=f'{year}-{int(year)+years}')

@app.route("/spyvix/custom", methods=["POST", "GET"])
def spyvixcustom():
    years = request.form['years']
    year = request.form['start']
    return redirect(url_for('spyvixrange', years=years, year=year))

@app.route("/yieldcurve/<year>/<period>", methods=["POST", "GET"])
def yieldcurve(year, period):
    year = year
    if period.upper() == 'MONTHLY':
        period = 'm'
        yld.YieldCurve().graph_curve_monthly(year)
    elif period.upper() == 'QUARTERLY':
        period = 'q'
        yld.YieldCurve().graph_curve_qtly(year)

    if year == '2024':
        yld.YieldCurve().graph_now()

    return render_template('yield_curve.html', year=year, period=period)

@app.route("/yieldcurve/set", methods=["POST", "GET"])
def setyieldcurve():
    year = request.form['year']
    period = request.form['period']
    return redirect(url_for('yieldcurve', year=year, period=period))



@app.route("/bondtrading/switch/<amt>", methods=["POST", "GET"])
def bond_trading_switch(amt):

    if 'ind' in session:
        session['ind'] += 1
    else:
        session['ind'] = 1
    # Retrieve the serialized game object from session

    with open('session_repl.json', 'r') as json_file:
        json_string = json_file.read()

    # Convert JSON string back to dictionary
    game_json = json.loads(json_string)

    # game_json = session.get('game', None)

    # game_data = json.loads(game_json)

    bond = bnd.Bond(game_json['par'], game_json['cr'], game_json['years'], game_json['price'])
    game = bnd.BondTrading(bond)
    game.from_dict(game_json)

    if amt == 'x':
        trade = request.form['trade']
    elif amt == 'clear':
        trade = str(game.position * -1)
    elif amt == 'reverse':
        trade = str(game.position * -2)
    elif amt == 'double':
        trade = str(game.position)
    else:
        trade = amt


    game.get_ready_for_next_period(trade)
    # print('after switch', game.bond.price)

    # print(f">>> position: {game.position}")
    if game.years != 0 and game.position != 0:
        # game.inflows += game.position * 1000 * game.bond.cr * game.bond.par
        game.cash += game.position * 1000 * game.bond.cr * game.bond.par
        game.transactions['t' + str(len(game.transactions.keys()))] = ( 'interest on holdings',
                                                                        game.position,
                                                                        f'{100*game.bond.cr:,.2f}%',
                                                                        f'${game.bond.par * game.bond.cr:,.2f}',
                                                                        f'${1000 * game.position * game.bond.par * game.bond.cr:,.2f}')

    """
    sitting on whether to do interest on cash at rfr or not
    """
    # if game.years != 0 and game.profits[-1] > 0:
        # game.inflows += game.profits[-1] * game.bond.ytm

        # game.cash += game.profits[-1] * game.bond.ytm
        # game.transactions['t' + str(len(game.transactions.keys()))] = ('interest on profit',
        #                                                                f'${game.profits[-1]:,.2f}',
        #                                                                f'RFR: {100*game.bond.ytm:,.2f}%',
        #                                                                f'${game.profits[-1] * game.bond.ytm:,.2f}')

    # print('inflow', game.position * 1000 * game.bond.cr * game.bond.par)

    game.display_round(session['ind'])

    g_dict = game.to_dict()
    # print('GAME DICT POST TRADE', g_dict)
    # session['game'] = json.dumps(g_dict)

    file_path = 'session_repl.json'
    # Check if the file exists
    # Convert dictionary to JSON string
    json_string = json.dumps(g_dict)
    # Write JSON string to the file
    with open(file_path, 'w') as json_file:
        json_file.write(json_string)

    return redirect(url_for('bond_trading_year'))


@app.route("/bondtrading", methods=["POST", "GET"])
def bond_trading_year():



    with open('session_repl.json', 'r') as json_file:
        json_string = json_file.read()

    # Convert JSON string back to dictionary
    try:
        game_json = json.loads(json_string)
    except Exception as e:
        return redirect(url_for('bond_trading_home'))


    # game_data = json.loads(game_json)
    bond = bnd.Bond(game_json['par'], game_json['cr'], game_json['years'], game_json['price'])

    game = bnd.BondTrading(bond)
    game.from_dict(game_json)

    print(f'CASH ${game.cash:,.2f}, ASSETS(LIAB) ${game.get_asset_liab():,.2f}, NETLIQ ${game.get_portfolio_value():,.2f}')


    tb_message1, tb_message2, tb_message3, tb_message4, tb_message5, tb_message6 = game.graph_bond_price()


    # game.display_round(session['ind'])
    g_dict = game.to_dict()
    session['game'] = json.dumps(g_dict)

    file_path = 'session_repl.json'

    # # Check if the file exists
    # if os.path.exists(file_path):
    #     mode = 'w'  # If file exists, overwrite
    # else:
    #     mode = 'x'  # If file doesn't exist, create

    # Convert dictionary to JSON string
    json_string = json.dumps(g_dict)

    # Write JSON string to the file
    with open(file_path, 'w') as json_file:
        json_file.write(json_string)
    if game.years >= 0:
        if game.position >= 0:
            close_color = '#ff0000'
        else:
            close_color = '#008000'
        return render_template('bond_trading.html',
                           trade_message=game.bond_trading_message(game.bond.price, game.bond.ytm),
                               transactions=game.transactions, close_color=close_color,
                               tb_message1=tb_message1, tb_message2=tb_message2, tb_message3=tb_message3,
                               tb_message4=tb_message4, tb_message5=tb_message5, tb_message6=tb_message6)
    else:
        if 'ind' in session:
            del session['ind']
        # del session['game']
        if 'trade' in session:
            del session['trade']
        if os.path.exists("iv_app/static/images/bond_trading"):
            # Remove the file
            os.remove("iv_app/static/images/bond_trading/recent_bond_graph.png")
        return redirect(url_for('bond_trading_home'))


@app.route("/bondtrading/first/<rerenderings>", methods=["POST", "GET"])
def bond_trading_first(rerenderings=None):
    if 'ind' in session:
        del session['ind']
    if 'game' in session:
        del session['game']
    if 'trade' in session:
        del session['trade']

    if len(rerenderings) < 3:
        par = float(request.form['par'])
        cr = float(request.form['coupon_rate'])
        maturity = int(request.form['maturity_period'])
        price = float(request.form['price'])
    else:
        rerenderings = rerenderings.split('-')
        par, cr, maturity, price = float(rerenderings[0]), float(rerenderings[1]), int(rerenderings[2]), float(rerenderings[3])

    starters = {'START_par':par, 'START_cr':cr, 'START_maturity':maturity, 'START_price':price}
    file_path = 'starters.json'
    if os.path.exists(file_path):
        mode = 'w'  # If file exists, overwrite
    else:
        mode = 'x'  # If file doesn't exist, create

    # Convert dictionary to JSON string
    json_string = json.dumps(starters)

    # Write JSON string to the file
    with open(file_path, mode) as json_file:
        json_file.write(json_string)

    #
    # session['START_par'] = par
    # session['START_cr'] = cr
    # session['START_maturity'] = maturity
    # session['START_price'] = price

    # Create a new game object
    bond = bnd.Bond(par, cr, maturity, price)
    game = bnd.BondTrading(bond)
    g_dict = game.to_dict()
    # Serialize the game object and store it in session

    # session['game'] = json.dumps(g_dict)

    file_path = 'session_repl.json'

    # Check if the file exists
    if os.path.exists(file_path):
        mode = 'w'  # If file exists, overwrite
    else:
        mode = 'x'  # If file doesn't exist, create

    # Convert dictionary to JSON string
    json_string = json.dumps(g_dict)

    # Write JSON string to the file
    with open(file_path, mode) as json_file:
        json_file.write(json_string)


    return redirect(url_for('bond_trading_year'))



@app.route("/bondtrading/home", methods=["POST", "GET"])
def bond_trading_home():
    return render_template('bond_trading_home.html')

@app.route("/cleantrading", methods=["POST", "GET"])
def clear_client():
    if 'ind' in session:
        del session['ind']
    if 'game' in session:
        del session['game']
    if 'trade' in session:
        del session['trade']
    if os.path.exists("iv_app/static/images/bond_trading/recent_bond_graph.png"):
        # Remove the file
        os.remove("iv_app/static/images/bond_trading/recent_bond_graph.png")
    return redirect(url_for('home'))

@app.route("/cleantradingstart", methods=["POST", "GET"])
def clear_trading():
    if 'ind' in session:
        del session['ind']
    if 'game' in session:
        del session['game']
    if 'trade' in session:
        del session['trade']
    if os.path.exists("iv_app/static/images/bond_trading/recent_bond_graph.png"):
        # Remove the file
        os.remove("iv_app/static/images/bond_trading/recent_bond_graph.png")

    return redirect(url_for('bond_trading_home'))


@app.route("/bondtrading/restart", methods=["POST", "GET"])
def bond_trading_restart():
    with open('starters.json', 'r') as json_file:
        json_string = json_file.read()

        # Convert JSON string back to dictionary
        game_json = json.loads(json_string)

    old_params = f'{game_json['START_par']}-{game_json['START_cr']}-{game_json['START_maturity']}-{game_json['START_price']}'
    return redirect(url_for('bond_trading_first', rerenderings=old_params))


@app.route("/cantorsquare", methods=["POST", "GET"])
def cantor_square():
    return render_template("cantor_square.html")

@app.route("/cantorsphere", methods=["POST", "GET"])
def cantor_sphere():
    return render_template("cantor_sphere.html")

@app.route("/mathviz", methods=["POST", "GET"])
def math_viz():
    return render_template("math_viz.html")

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
