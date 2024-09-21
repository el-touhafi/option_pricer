import pandas as pd
import yfinance as yf
from datetime import *
import numpy as np
import requests
from bs4 import BeautifulSoup as bs
import urllib3
import re

urllib3.disable_warnings()


class OptionMother:
    """
    This class generates approximately all the variables needed in the computation of the underlying option's price.
    I will introduce the ambigious variables in this part:
    underlying_ticker : it's the stock ticker of any firm that exists in yahoo finance website or api (eg 'apple stock' : 'AAPL')
    steps_number : number of stepsin the tree (we can find convergence to the Black and Scholes vanilla option pricing model in 50 steps)
    option_type : 'Call' or 'Put'
    expiry_date : this date will serve to customize you time to maturity "%Y-%m-%d"
    which_contract_number : if you want listed contracts in yahoo finance options you can chose by number [0,1,...,15]
    NB : The last close price of the unerlying price will be automatically added as an attribute of the class
    """

    def __init__(
        self,
        strike: float,
        underlying_ticker: str,
        steps_number: int,
        option_type: str,
        expiry_date: str,
        dividend: float,
        interest_rate: float,
        volatility: float,
        ):
        self.strike = strike
        self.underlying_ticker = underlying_ticker
        self.interest_rate = interest_rate
        self.dividend = dividend
        self.volatility = volatility
        self.steps_number = steps_number
        self.option_type = option_type
        self.expiry_date = expiry_date
        self.underlying_price_df = pd.DataFrame()
        self.url = str()
        self.expiry_date_to_maturity = date

    def load_underlying_price(self):
        """
        This fonction will serve to import underlying prices dataframe and fill self.underlying_price_df
        Note: if you encountred any problem in importing the dataframe probably the cause will be the api
        and we will try to fix the problem as soon as possible
        """
        ticker = self.data
        underlying_price_df = ticker.history(period="1y")["Close"]
        underlying_price_df = pd.DataFrame(
            np.array(underlying_price_df.iloc[:]),
            columns=[self.underlying_ticker],
            index=np.array(underlying_price_df.index.strftime("%Y-%m-%d")),
        )
        self.underlying_price_df = underlying_price_df

    def load_dividend(self):
        """
        This fonction will serve to import underlying dividend of this year and fill self.dividend
        Note: if you encountred any problem in importing the dataframe probably the cause will be the api
        and we will try to fix the problem as soon as possible
        """
        if self.dividend == 0:
            ticker = yf.Ticker(self.underlying_ticker)
            dividend = float(
                sum(ticker.history(period="1y")["Dividends"])
                / self.underlying_price_df.iloc[-1, 0]
            )
            self.dividend = dividend

    def calculate_volatility(self):
        """
        Based on self.underlying_price_df we will compute the volatility
        """
        if self.volatility == 0:
            underlying_price_df_shift = self.underlying_price_df.pct_change()
            underlying_price_df_shift = underlying_price_df_shift.dropna()
            self.volatility = np.std(np.array(underlying_price_df_shift)) * np.sqrt(260)

    def load_expiries(self):
        """
        This method will surve to scrape the expiry dates that exists in yahoo finance options you can check it in there plateform
        Note: if you encountred any problem in importing the list of expiries probably the cause will be the get request and
        we will try to fix the problem as soon as possible
        """
        self.number_of_days = (
            (datetime.strptime(self.expiry_date, "%Y-%m-%d").date() - date.today()).days
            / 365
            if datetime.strptime(self.expiry_date, "%Y-%m-%d").date() > date.today()
            else None
        )

    def compute_option_contract(self):
        """
        this method will serve in computing all the methods needed to run the process and give the option price
        """
        self.data = yf.Ticker(self.underlying_ticker)
        self.load_underlying_price()
        self.load_dividend()
        self.load_expiries()
        self.calculate_volatility()
