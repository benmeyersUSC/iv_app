IV-APP is a Flask webapp implemented in Python, HTML, CSS that explores my love of options! While I continue to tirelessly improve the app to add functionality, as it is now, I believe it can provide valuable information about real-time data (stock prices, option prices, Implied Volatility, forward looking price ranges) as well as some intuitive education on options pricing in parlence I have yet to see in such a dense subject. The mathematical complexity and elegance of the Black Scholes formula should not be overlooked, however with such a barrier of understanidng, intuitive explanations and interactive tools can greatly inform at least on the 'music' the intuition behind options pricing, if not diving deep into Ito Calculus ;)

Click this [link](https://iv-app-1408b00f8e09.herokuapp.com/)! 


Pages:
- Home page gives the user a dashboard of choices to explore
  
- Research Ticker
-   this page allows the user to input any optionable underlying symbol to get a quick picture of recent price action as well as the current IV metrics that can inform its near future

- Black Scholes Model
-   the first click takes you to a static page that walks through the Black Scholes Formula and hedging strategy
-   from this page, the user can then simulate the Black Scholes hedging strategy with their own parameters, as many times as they want!

- Stock Group Info
-   the final several tabs on the home page allow the user to look at different 'watchlists' or portfolios of underlyings
-   This week's earnings names, the current top open interest underlyings, the Mag 7, and the market's top indices

