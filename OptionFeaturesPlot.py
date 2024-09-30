import numpy as np
import sys
import matplotlib.pyplot as plt

sys.path.append("..")
from OptionMother import OptionMother
from scipy.stats import norm


class OptionFeaturesPlot(OptionMother):
    """
    This class will serve to plot (2d and 3d) the features sensitivity to option [strike, underlying price, volatility ...].
    I will introduce the ambigious variables in this part:
    feature_in_x : you should enter one of this ["Stock Price","Strike","Maturity","Volatility","Interest rate","Dividend"]
    type_plot : you should enter one of this ["2d","3d"]
    x_min : is the minimum value of your chosen feature_in_x
    x_max : is the maximum value of your chosen feature_in_x
    if type_plot that you have entered is 2d :
        feature_in_y : you should enter one of this ["black scholes","delta","gamma","rho","theta","vega"]
    if type_plot that you have entered is 3d :
        feature_in_y : you should enter one of this ["Stock Price","Strike","Maturity","Volatility","Interest rate","Dividend"]
                       !!!! Note that: feature_in_y shouldn't be equal to feature_in_x
        y_min : is the minimum value of your chosen feature_in_y
        y_max : is the maximum value of your chosen feature_in_y
        feature_in_z : you should enter one of this ["black scholes","delta","gamma","rho","theta","vega"]
    NB : The last close price of the unerlying price will be automatically added as an attribute of the class
    """

    def __init__(
        self,
        strike: float,
        underlying_ticker: str,
        steps_number: int,
        option_type: str,
        expiry_date: str,
        interest_rate: float,
        dividend: float,
        volatility: float,
        x_min: float,
        x_max: float,
        feature_in_x: str,
        y_min: float,
        y_max: float,
        feature_in_y: str,
        feature_in_z: str,
        type_plot: str,
    ):
        super().__init__(
            strike,
            underlying_ticker,
            steps_number,
            option_type,
            expiry_date,
            dividend,
            interest_rate,
            volatility,
        )
        self.x_min = x_min
        self.y_min = y_min
        self.x_max = x_max
        self.y_max = y_max
        self.feature_in_x = feature_in_x
        self.feature_in_y = feature_in_y
        self.feature_in_z = feature_in_z
        self.type_plot = type_plot

    def black_scholes_plot(
        self,
        underlying_price: float,
        strike: float,
        maturity: float,
        volatility: float,
        interest_rate: float,
        option_type: str,
        dividend: float,
    ):
        """
        This method is like the black_scholes_model method of EuropeenOption class but it will be more flexible for academic and plot use.
        It isn't related to the yahoo finance api.
        """
        d1 = (1 / (volatility * np.sqrt(maturity))) * (
            np.log(underlying_price / strike)
            + ((interest_rate - dividend) + 0.5 * (volatility) ** 2) * maturity
        )
        d2 = d1 - volatility * np.sqrt(maturity)
        if option_type == "Call":
            return np.exp(dividend * maturity) * underlying_price * norm.cdf(
                d1
            ) - strike * np.exp(-interest_rate * maturity) * norm.cdf(d2)
        elif option_type == "Put":
            return -np.exp(dividend * maturity) * underlying_price * norm.cdf(
                -d1
            ) + strike * np.exp(-interest_rate * maturity) * norm.cdf(-d2)

    def delta(
        self,
        underlying_price: float,
        strike: float,
        maturity: float,
        volatility: float,
        interest_rate: float,
        option_type: str,
        dividend: float,
    ):
        """
        This method is like the delta method of EuropeenOption class but it will be more flexible for academic and plot use.
        It isn't related to the yahoo finance api.
        """
        d1 = (1 / (volatility * np.sqrt(maturity))) * (
            np.log(underlying_price / strike)
            + ((interest_rate - dividend) + 0.5 * (volatility) ** 2) * maturity
        )
        return (
            np.exp(-dividend * maturity) * norm.cdf(d1)
            if option_type == "Call"
            else -np.exp(-dividend * maturity) * norm.cdf(-d1)
        )

    def gamma(
        self,
        underlying_price: float,
        strike: float,
        maturity: float,
        volatility: float,
        interest_rate: float,
        dividend: float,
    ):
        """
        This method is like the gamma method of EuropeenOption class but it will be more flexible for academic and plot use.
        It isn't related to the yahoo finance api.
        """
        d1 = (1 / (volatility * np.sqrt(maturity))) * (
            np.log(underlying_price / strike)
            + ((interest_rate - dividend) + 0.5 * (volatility) ** 2) * maturity
        )
        return (
            np.exp(-dividend * maturity)
            * norm.pdf(d1)
            / (underlying_price * volatility * np.sqrt(maturity))
        )

    def theta(
        self,
        underlying_price: float,
        strike: float,
        maturity: float,
        volatility: float,
        interest_rate: float,
        option_type: str,
        dividend: float,
    ):
        """
        This method is like the theta method of EuropeenOption class but it will be more flexible for academic and plot use.
        It isn't related to the yahoo finance api.
        """
        d1 = (1 / (volatility * np.sqrt(maturity))) * (
            np.log(underlying_price / strike)
            + ((interest_rate - dividend) + 0.5 * (volatility) ** 2) * maturity
        )
        d2 = d1 - volatility * np.sqrt(maturity)
        call = (
            (
                -np.exp(-dividend * maturity)
                * underlying_price
                * norm.pdf(d1, 0, 1)
                * volatility
            )
            / (2 * np.sqrt(maturity))
            - interest_rate
            * strike
            * np.exp(-interest_rate * maturity)
            * norm.cdf(d2, 0, 1)
            + dividend
            * underlying_price
            * np.exp(-dividend * maturity)
            * norm.cdf(d1, 0, 1)
        )
        put = (
            (
                -np.exp(-dividend * maturity)
                * underlying_price
                * norm.pdf(d1, 0, 1)
                * volatility
            )
            / (2 * np.sqrt(maturity))
            + interest_rate * strike * np.exp(-interest_rate * maturity) * norm.cdf(-d2)
            - dividend
            * underlying_price
            * np.exp(-dividend * maturity)
            * norm.cdf(-d1, 0, 1)
        )
        return call if option_type == "Call" else put

    def vega(
        self,
        underlying_price: float,
        strike: float,
        maturity: float,
        volatility: float,
        interest_rate: float,
        dividend: float,
    ):
        """
        This method is like the vega method of EuropeenOption class but it will be more flexible for academic and plot use.
        It isn't related to the yahoo finance api.
        """
        d1 = (1 / (volatility * np.sqrt(maturity))) * (
            np.log(underlying_price / strike)
            + ((interest_rate - dividend) + 0.5 * (volatility) ** 2) * maturity
        )
        d2 = d1 - volatility * np.sqrt(maturity)
        return (
            underlying_price
            * np.exp(-dividend * maturity)
            * norm.pdf(d1)
            * np.sqrt(maturity)
        )

    def rho(
        self,
        underlying_price: float,
        strike: float,
        maturity: float,
        volatility: float,
        interest_rate: float,
        option_type: str,
        dividend: float,
    ):
        """
        This method is like the rho method of EuropeenOption class but it will be more flexible for academic and plot use.
        It isn't related to the yahoo finance api.
        """
        d1 = (1 / (volatility * np.sqrt(maturity))) * (
            np.log(underlying_price / strike)
            + ((interest_rate - dividend) + 0.5 * (volatility) ** 2) * maturity
        )
        d2 = d1 - volatility * np.sqrt(maturity)
        call = (
            strike * maturity * np.exp(-interest_rate * maturity) * norm.cdf(d2, 0, 1)
        )
        put = (
            -strike * maturity * np.exp(-interest_rate * maturity) * norm.cdf(-d2, 0, 1)
        )
        return call if option_type == "Call" else put

    def compute_option_contract(self):
        """
        this method will serve in computing all the methods needed to run the process and give the option price and plot.
        """
        super().compute_option_contract()
        if self.type_plot == "2d":
            self.option_features_2D()
        else:
            self.option_features_3D()

    def option_features_2D(self):
        """
        This method will serve to plot the 2d plot of x_features against y_option_features
        """
        x_min = int(self.x_min)
        x_max = int(self.x_max)

        x_features = [
            "Stock Price",
            "Strike",
            "Maturity",
            "Volatility",
            "Interest rate",
            "Dividend",
        ]
        x_features = np.array(x_features)
        matching_indices_x = np.where(x_features == self.feature_in_x)[0]
        y_option_features = ["black scholes", "delta", "gamma", "rho", "theta", "vega"]
        y_option_features = np.array(y_option_features)
        matching_indices_y = np.where(y_option_features == self.feature_in_y)[0]
        if x_max < 2:
            x_features = np.arange(x_min, x_max, 0.1)
            y_option_features = np.zeros((6, len(x_features)))
        else:
            x_features = np.arange(x_min, x_max)
            y_option_features = np.zeros((6, len(x_features)))
        match matching_indices_x:
            case 0:

                for y_vector, x_position in enumerate(x_features):
                    y_option_features[0, y_vector] = self.black_scholes_plot(
                        x_position,
                        self.strike,
                        self.number_of_days,
                        self.volatility,
                        self.interest_rate,
                        self.option_type,
                        self.dividend,
                    )
                    y_option_features[1, y_vector] = self.delta(
                        x_position,
                        self.strike,
                        self.number_of_days,
                        self.volatility,
                        self.interest_rate,
                        self.option_type,
                        self.dividend,
                    )
                    y_option_features[2, y_vector] = self.gamma(
                        x_position,
                        self.strike,
                        self.number_of_days,
                        self.volatility,
                        self.interest_rate,
                        self.dividend,
                    )
                    y_option_features[3, y_vector] = self.rho(
                        x_position,
                        self.strike,
                        self.number_of_days,
                        self.volatility,
                        self.interest_rate,
                        self.option_type,
                        self.dividend,
                    )
                    y_option_features[4, y_vector] = self.theta(
                        x_position,
                        self.strike,
                        self.number_of_days,
                        self.volatility,
                        self.interest_rate,
                        self.option_type,
                        self.dividend,
                    )
                    y_option_features[5, y_vector] = self.vega(
                        x_position,
                        self.strike,
                        self.number_of_days,
                        self.volatility,
                        self.interest_rate,
                        self.dividend,
                    )
            case 1:
                for y_vector, x_position in enumerate(x_features):
                    y_option_features[0, y_vector] = self.black_scholes_plot(
                        self.underlying_price_df.iloc[-1, 0],
                        x_position,
                        self.number_of_days,
                        self.volatility,
                        self.interest_rate,
                        self.option_type,
                        self.dividend,
                    )
                    y_option_features[1, y_vector] = self.delta(
                        self.underlying_price_df.iloc[-1, 0],
                        x_position,
                        self.number_of_days,
                        self.volatility,
                        self.interest_rate,
                        self.option_type,
                        self.dividend,
                    )
                    y_option_features[2, y_vector] = self.gamma(
                        self.underlying_price_df.iloc[-1, 0],
                        x_position,
                        self.number_of_days,
                        self.volatility,
                        self.interest_rate,
                        self.dividend,
                    )
                    y_option_features[3, y_vector] = self.rho(
                        self.underlying_price_df.iloc[-1, 0],
                        x_position,
                        self.number_of_days,
                        self.volatility,
                        self.interest_rate,
                        self.option_type,
                        self.dividend,
                    )
                    y_option_features[4, y_vector] = self.theta(
                        self.underlying_price_df.iloc[-1, 0],
                        x_position,
                        self.number_of_days,
                        self.volatility,
                        self.interest_rate,
                        self.option_type,
                        self.dividend,
                    )
                    y_option_features[5, y_vector] = self.vega(
                        self.underlying_price_df.iloc[-1, 0],
                        x_position,
                        self.number_of_days,
                        self.volatility,
                        self.interest_rate,
                        self.dividend,
                    )

            case 2:

                for y_vector, x_position in enumerate(x_features):
                    y_option_features[0, y_vector] = self.black_scholes_plot(
                        self.underlying_price_df.iloc[-1, 0],
                        self.strike,
                        x_position,
                        self.volatility,
                        self.interest_rate,
                        self.option_type,
                        self.dividend,
                    )
                    y_option_features[1, y_vector] = self.delta(
                        self.underlying_price_df.iloc[-1, 0],
                        self.strike,
                        x_position,
                        self.volatility,
                        self.interest_rate,
                        self.option_type,
                        self.dividend,
                    )
                    y_option_features[2, y_vector] = self.gamma(
                        self.underlying_price_df.iloc[-1, 0],
                        self.strike,
                        x_position,
                        self.volatility,
                        self.interest_rate,
                        self.dividend,
                    )
                    y_option_features[3, y_vector] = self.rho(
                        self.underlying_price_df.iloc[-1, 0],
                        self.strike,
                        x_position,
                        self.volatility,
                        self.interest_rate,
                        self.option_type,
                        self.dividend,
                    )
                    y_option_features[4, y_vector] = self.theta(
                        self.underlying_price_df.iloc[-1, 0],
                        self.strike,
                        x_position,
                        self.volatility,
                        self.interest_rate,
                        self.option_type,
                        self.dividend,
                    )
                    y_option_features[5, y_vector] = self.vega(
                        self.underlying_price_df.iloc[-1, 0],
                        self.strike,
                        x_position,
                        self.volatility,
                        self.interest_rate,
                        self.dividend,
                    )

            case 3:

                for y_vector, x_position in enumerate(x_features):
                    y_option_features[0, y_vector] = self.black_scholes_plot(
                        self.underlying_price_df.iloc[-1, 0],
                        self.strike,
                        self.number_of_days,
                        x_position,
                        self.interest_rate,
                        self.option_type,
                        self.dividend,
                    )
                    y_option_features[1, y_vector] = self.delta(
                        self.underlying_price_df.iloc[-1, 0],
                        self.strike,
                        self.number_of_days,
                        x_position,
                        self.interest_rate,
                        self.option_type,
                        self.dividend,
                    )
                    y_option_features[2, y_vector] = self.gamma(
                        self.underlying_price_df.iloc[-1, 0],
                        self.strike,
                        self.number_of_days,
                        x_position,
                        self.interest_rate,
                        self.dividend,
                    )
                    y_option_features[3, y_vector] = self.rho(
                        self.underlying_price_df.iloc[-1, 0],
                        self.strike,
                        self.number_of_days,
                        x_position,
                        self.interest_rate,
                        self.option_type,
                        self.dividend,
                    )
                    y_option_features[4, y_vector] = self.theta(
                        self.underlying_price_df.iloc[-1, 0],
                        self.strike,
                        self.number_of_days,
                        x_position,
                        self.interest_rate,
                        self.option_type,
                        self.dividend,
                    )
                    y_option_features[5, y_vector] = self.vega(
                        self.underlying_price_df.iloc[-1, 0],
                        self.strike,
                        self.number_of_days,
                        x_position,
                        self.interest_rate,
                        self.dividend,
                    )

            case 4:

                for y_vector, x_position in enumerate(x_features):
                    y_option_features[0, y_vector] = self.black_scholes_plot(
                        self.underlying_price_df.iloc[-1, 0],
                        self.strike,
                        self.number_of_days,
                        self.volatility,
                        x_position,
                        self.option_type,
                        self.dividend,
                    )
                    y_option_features[1, y_vector] = self.delta(
                        self.underlying_price_df.iloc[-1, 0],
                        self.strike,
                        self.number_of_days,
                        self.volatility,
                        x_position,
                        self.option_type,
                        self.dividend,
                    )
                    y_option_features[2, y_vector] = self.gamma(
                        self.underlying_price_df.iloc[-1, 0],
                        self.strike,
                        self.number_of_days,
                        self.volatility,
                        x_position,
                        self.dividend,
                    )
                    y_option_features[3, y_vector] = self.rho(
                        self.underlying_price_df.iloc[-1, 0],
                        self.strike,
                        self.number_of_days,
                        self.volatility,
                        x_position,
                        self.option_type,
                        self.dividend,
                    )
                    y_option_features[4, y_vector] = self.theta(
                        self.underlying_price_df.iloc[-1, 0],
                        self.strike,
                        self.number_of_days,
                        self.volatility,
                        x_position,
                        self.option_type,
                        self.dividend,
                    )
                    y_option_features[5, y_vector] = self.vega(
                        self.underlying_price_df.iloc[-1, 0],
                        self.strike,
                        self.number_of_days,
                        self.volatility,
                        x_position,
                        self.dividend,
                    )

            case 5:

                for y_vector, x_position in enumerate(x_features):
                    y_option_features[0, y_vector] = self.black_scholes_plot(
                        self.underlying_price_df.iloc[-1, 0],
                        self.strike,
                        self.number_of_days,
                        self.volatility,
                        self.interest_rate,
                        self.option_type,
                        x_position,
                    )
                    y_option_features[1, y_vector] = self.delta(
                        self.underlying_price_df.iloc[-1, 0],
                        self.strike,
                        self.number_of_days,
                        self.volatility,
                        self.interest_rate,
                        self.option_type,
                        x_position,
                    )
                    y_option_features[2, y_vector] = self.gamma(
                        self.underlying_price_df.iloc[-1, 0],
                        self.strike,
                        self.number_of_days,
                        self.volatility,
                        self.interest_rate,
                        x_position,
                    )
                    y_option_features[3, y_vector] = self.rho(
                        self.underlying_price_df.iloc[-1, 0],
                        self.strike,
                        self.number_of_days,
                        self.volatility,
                        self.interest_rate,
                        self.option_type,
                        x_position,
                    )
                    y_option_features[4, y_vector] = self.theta(
                        self.underlying_price_df.iloc[-1, 0],
                        self.strike,
                        self.number_of_days,
                        self.volatility,
                        self.interest_rate,
                        self.option_type,
                        x_position,
                    )
                    y_option_features[5, y_vector] = self.vega(
                        self.underlying_price_df.iloc[-1, 0],
                        self.strike,
                        self.number_of_days,
                        self.volatility,
                        self.interest_rate,
                        x_position,
                    )

            case _:
                print("error")
        plt.plot(x_features.T, y_option_features[matching_indices_y].T)
        plt.xlabel(f"{self.feature_in_x} feature")
        plt.ylabel(f"{self.feature_in_y} option")
        plt.title(
            f"Plot of how {self.feature_in_y} option behave if {self.feature_in_x} feature variates"
        )
        plt.show()
        self.image = x_features, y_option_features[matching_indices_y]

    def option_features_3D(self):
        """
        This method will serve to plot the 3d plot of x_features and y_feature against z_option_feature
        """
        x_features = [
            "Stock Price",
            "Strike",
            "Maturity",
            "Volatility",
            "Interest rate",
            "Dividend",
        ]
        x_features = np.array(x_features)
        matching_indices_x1 = int(np.where(x_features == self.feature_in_x)[0])
        matching_indices_x2 = int(np.where(x_features == self.feature_in_y)[0])
        matching_indices = str(matching_indices_x1) + str(matching_indices_x2)
        if (self.x_max - self.x_min) < (self.y_max - self.y_min):
            if self.x_max <= 10:
                x_features = np.arange(self.x_min, self.x_max, 0.1)
                y_feature = np.arange(
                    self.y_min, self.y_max, (self.y_max - self.y_min) / len(x_features)
                )
            else:
                x_features = np.arange(self.x_min, self.x_max)
                y_feature = np.arange(
                    self.y_min, self.y_max, (self.y_max - self.y_min) / len(x_features)
                )
        else:
            if self.y_max <= 10:
                y_feature = np.arange(self.y_min, self.y_max, 0.1)
                x_features = np.arange(
                    self.x_min, self.x_max, (self.x_max - self.x_min) / len(y_feature)
                )
            else:
                x_features = np.arange(self.x_min, self.x_max)
                y_feature = np.arange(
                    self.y_min, self.y_max, (self.y_max - self.y_min) / len(x_features)
                )
        z_option_feature = np.zeros((len(x_features), len(y_feature)))
        match matching_indices:
            case "01" | "10":
                if matching_indices == "10":
                    switch_1 = self.x_min
                    self.x_min = self.y_min
                    self.y_min = switch_1
                    switch_1 = self.x_max
                    self.x_max = self.y_max
                    self.y_max = switch_1
                    switch_2 = x_features
                    x_features = y_feature
                    y_feature = switch_2
                    switch_3 = self.feature_in_x
                    self.feature_in_x = self.feature_in_y
                    self.feature_in_y = switch_3

                for x_position, x_feature_value in enumerate(x_features):

                    for y_position, y_feature_value in enumerate(y_feature):
                        match self.feature_in_z:
                            case "black scholes":
                                z_option_feature[x_position, y_position] = (
                                    self.black_scholes_plot(
                                        x_feature_value,
                                        y_feature_value,
                                        self.number_of_days,
                                        self.volatility,
                                        self.interest_rate,
                                        self.option_type,
                                        self.dividend,
                                    )
                                )
                            case "delta":
                                z_option_feature[x_position, y_position] = self.delta(
                                    x_feature_value,
                                    y_feature_value,
                                    self.number_of_days,
                                    self.volatility,
                                    self.interest_rate,
                                    self.option_type,
                                    self.dividend,
                                )
                            case "gamma":
                                z_option_feature[x_position, y_position] = self.gamma(
                                    x_feature_value,
                                    y_feature_value,
                                    self.number_of_days,
                                    self.volatility,
                                    self.interest_rate,
                                    self.dividend,
                                )
                            case "rho":
                                z_option_feature[x_position, y_position] = self.rho(
                                    x_feature_value,
                                    y_feature_value,
                                    self.number_of_days,
                                    self.volatility,
                                    self.interest_rate,
                                    self.option_type,
                                    self.dividend,
                                )
                            case "theta":
                                z_option_feature[x_position, y_position] = self.theta(
                                    x_feature_value,
                                    y_feature_value,
                                    self.number_of_days,
                                    self.volatility,
                                    self.interest_rate,
                                    self.option_type,
                                    self.dividend,
                                )
                            case "vega":
                                z_option_feature[x_position, y_position] = self.vega(
                                    x_feature_value,
                                    y_feature_value,
                                    self.number_of_days,
                                    self.volatility,
                                    self.interest_rate,
                                    self.dividend,
                                )

            case "02" | "20":
                if matching_indices == "20":
                    switch_1 = self.x_min
                    self.x_min = self.y_min
                    self.y_min = switch_1
                    switch_1 = self.x_max
                    self.x_max = self.y_max
                    self.y_max = switch_1
                    switch_2 = x_features
                    x_features = y_feature
                    y_feature = switch_2
                    switch_3 = self.feature_in_x
                    self.feature_in_x = self.feature_in_y
                    self.feature_in_y = switch_3

                for x_position, x_feature_value in enumerate(x_features):

                    for y_position, y_feature_value in enumerate(y_feature):
                        match self.feature_in_z:
                            case "black scholes":
                                z_option_feature[x_position, y_position] = (
                                    self.black_scholes_plot(
                                        x_feature_value,
                                        self.strike,
                                        y_feature_value,
                                        self.volatility,
                                        self.interest_rate,
                                        self.option_type,
                                        self.dividend,
                                    )
                                )
                            case "delta":
                                z_option_feature[x_position, y_position] = self.delta(
                                    x_feature_value,
                                    self.strike,
                                    y_feature_value,
                                    self.volatility,
                                    self.interest_rate,
                                    self.option_type,
                                    self.dividend,
                                )
                            case "gamma":
                                z_option_feature[x_position, y_position] = self.gamma(
                                    x_feature_value,
                                    self.strike,
                                    y_feature_value,
                                    self.volatility,
                                    self.interest_rate,
                                    self.dividend,
                                )
                            case "rho":
                                z_option_feature[x_position, y_position] = self.rho(
                                    x_feature_value,
                                    self.strike,
                                    y_feature_value,
                                    self.volatility,
                                    self.interest_rate,
                                    self.option_type,
                                    self.dividend,
                                )
                            case "theta":
                                z_option_feature[x_position, y_position] = self.theta(
                                    x_feature_value,
                                    self.strike,
                                    y_feature_value,
                                    self.volatility,
                                    self.interest_rate,
                                    self.option_type,
                                    self.dividend,
                                )
                            case "vega":
                                z_option_feature[x_position, y_position] = self.vega(
                                    x_feature_value,
                                    self.strike,
                                    y_feature_value,
                                    self.volatility,
                                    self.interest_rate,
                                    self.dividend,
                                )

            case "03" | "30":
                if matching_indices == "30":
                    switch_1 = self.x_min
                    self.x_min = self.y_min
                    self.y_min = switch_1
                    switch_1 = self.x_max
                    self.x_max = self.y_max
                    self.y_max = switch_1
                    switch_2 = x_features
                    x_features = y_feature
                    y_feature = switch_2
                    switch_3 = self.feature_in_x
                    self.feature_in_x = self.feature_in_y
                    self.feature_in_y = switch_3

                for x_position, x_feature_value in enumerate(x_features):

                    for y_position, y_feature_value in enumerate(y_feature):
                        match self.feature_in_z:
                            case "black scholes":
                                z_option_feature[x_position, y_position] = (
                                    self.black_scholes_plot(
                                        x_feature_value,
                                        self.strike,
                                        self.number_of_days,
                                        y_feature_value,
                                        self.interest_rate,
                                        self.option_type,
                                        self.dividend,
                                    )
                                )
                            case "delta":
                                z_option_feature[x_position, y_position] = self.delta(
                                    x_feature_value,
                                    self.strike,
                                    self.number_of_days,
                                    y_feature_value,
                                    self.interest_rate,
                                    self.option_type,
                                    self.dividend,
                                )
                            case "gamma":
                                z_option_feature[x_position, y_position] = self.gamma(
                                    x_feature_value,
                                    self.strike,
                                    self.number_of_days,
                                    y_feature_value,
                                    self.interest_rate,
                                    self.dividend,
                                )
                            case "rho":
                                z_option_feature[x_position, y_position] = self.rho(
                                    x_feature_value,
                                    self.strike,
                                    self.number_of_days,
                                    y_feature_value,
                                    self.interest_rate,
                                    self.option_type,
                                    self.dividend,
                                )
                            case "theta":
                                z_option_feature[x_position, y_position] = self.theta(
                                    x_feature_value,
                                    self.strike,
                                    self.number_of_days,
                                    y_feature_value,
                                    self.interest_rate,
                                    self.option_type,
                                    self.dividend,
                                )
                            case "vega":
                                z_option_feature[x_position, y_position] = self.vega(
                                    x_feature_value,
                                    self.strike,
                                    self.number_of_days,
                                    y_feature_value,
                                    self.interest_rate,
                                    self.dividend,
                                )

            case "04" | "40":
                if matching_indices == "40":
                    switch_1 = self.x_min
                    self.x_min = self.y_min
                    self.y_min = switch_1
                    switch_1 = self.x_max
                    self.x_max = self.y_max
                    self.y_max = switch_1
                    switch_2 = x_features
                    x_features = y_feature
                    y_feature = switch_2
                    switch_3 = self.feature_in_x
                    self.feature_in_x = self.feature_in_y
                    self.feature_in_y = switch_3

                for x_position, x_feature_value in enumerate(x_features):

                    for y_position, y_feature_value in enumerate(y_feature):
                        match self.feature_in_z:
                            case "black scholes":
                                z_option_feature[x_position, y_position] = (
                                    self.black_scholes_plot(
                                        x_feature_value,
                                        self.strike,
                                        self.number_of_days,
                                        self.volatility,
                                        y_feature_value,
                                        self.option_type,
                                        self.dividend,
                                    )
                                )
                            case "delta":
                                z_option_feature[x_position, y_position] = self.delta(
                                    x_feature_value,
                                    self.strike,
                                    self.number_of_days,
                                    self.volatility,
                                    y_feature_value,
                                    self.option_type,
                                    self.dividend,
                                )
                            case "gamma":
                                z_option_feature[x_position, y_position] = self.gamma(
                                    x_feature_value,
                                    self.strike,
                                    self.number_of_days,
                                    self.volatility,
                                    y_feature_value,
                                    self.dividend,
                                )
                            case "rho":
                                z_option_feature[x_position, y_position] = self.rho(
                                    x_feature_value,
                                    self.strike,
                                    self.number_of_days,
                                    self.volatility,
                                    y_feature_value,
                                    self.option_type,
                                    self.dividend,
                                )
                            case "theta":
                                z_option_feature[x_position, y_position] = self.theta(
                                    x_feature_value,
                                    self.strike,
                                    self.number_of_days,
                                    self.volatility,
                                    y_feature_value,
                                    self.option_type,
                                    self.dividend,
                                )
                            case "vega":
                                z_option_feature[x_position, y_position] = self.vega(
                                    x_feature_value,
                                    self.strike,
                                    self.number_of_days,
                                    self.volatility,
                                    y_feature_value,
                                    self.dividend,
                                )

            case "05" | "50":
                if matching_indices == "50":
                    switch_1 = self.x_min
                    self.x_min = self.y_min
                    self.y_min = switch_1
                    switch_1 = self.x_max
                    self.x_max = self.y_max
                    self.y_max = switch_1
                    switch_2 = x_features
                    x_features = y_feature
                    y_feature = switch_2
                    switch_3 = self.feature_in_x
                    self.feature_in_x = self.feature_in_y
                    self.feature_in_y = switch_3

                for x_position, x_feature_value in enumerate(x_features):

                    for y_position, y_feature_value in enumerate(y_feature):
                        match self.feature_in_z:
                            case "black scholes":
                                z_option_feature[x_position, y_position] = (
                                    self.black_scholes_plot(
                                        x_feature_value,
                                        self.strike,
                                        self.number_of_days,
                                        self.volatility,
                                        self.interest_rate,
                                        self.option_type,
                                        y_feature_value,
                                    )
                                )
                            case "delta":
                                z_option_feature[x_position, y_position] = self.delta(
                                    x_feature_value,
                                    self.strike,
                                    self.number_of_days,
                                    self.volatility,
                                    self.interest_rate,
                                    self.option_type,
                                    y_feature_value,
                                )
                            case "gamma":
                                z_option_feature[x_position, y_position] = self.gamma(
                                    x_feature_value,
                                    self.strike,
                                    self.number_of_days,
                                    self.volatility,
                                    self.interest_rate,
                                    y_feature_value,
                                )
                            case "rho":
                                z_option_feature[x_position, y_position] = self.rho(
                                    x_feature_value,
                                    self.strike,
                                    self.number_of_days,
                                    self.volatility,
                                    self.interest_rate,
                                    self.option_type,
                                    y_feature_value,
                                )
                            case "theta":
                                z_option_feature[x_position, y_position] = self.theta(
                                    x_feature_value,
                                    self.strike,
                                    self.number_of_days,
                                    self.volatility,
                                    self.interest_rate,
                                    self.option_type,
                                    y_feature_value,
                                )
                            case "vega":
                                z_option_feature[x_position, y_position] = self.vega(
                                    x_feature_value,
                                    self.strike,
                                    self.number_of_days,
                                    self.volatility,
                                    self.interest_rate,
                                    y_feature_value,
                                )

            case "21" | "12":
                if matching_indices == "21":
                    switch_1 = self.x_min
                    self.x_min = self.y_min
                    self.y_min = switch_1
                    switch_1 = self.x_max
                    self.x_max = self.y_max
                    self.y_max = switch_1
                    switch_2 = x_features
                    x_features = y_feature
                    y_feature = switch_2
                    switch_3 = self.feature_in_x
                    self.feature_in_x = self.feature_in_y
                    self.feature_in_y = switch_3

                for x_position, x_feature_value in enumerate(x_features):

                    for y_position, y_feature_value in enumerate(y_feature):
                        match self.feature_in_z:
                            case "black scholes":
                                z_option_feature[x_position, y_position] = (
                                    self.black_scholes_plot(
                                        self.underlying_price_df.iloc[-1, 0],
                                        x_feature_value,
                                        y_feature_value,
                                        self.volatility,
                                        self.interest_rate,
                                        self.option_type,
                                        self.dividend,
                                    )
                                )
                            case "delta":
                                z_option_feature[x_position, y_position] = self.delta(
                                    self.underlying_price_df.iloc[-1, 0],
                                    x_feature_value,
                                    y_feature_value,
                                    self.volatility,
                                    self.interest_rate,
                                    self.option_type,
                                    self.dividend,
                                )
                            case "gamma":
                                z_option_feature[x_position, y_position] = self.gamma(
                                    self.underlying_price_df.iloc[-1, 0],
                                    x_feature_value,
                                    y_feature_value,
                                    self.volatility,
                                    self.interest_rate,
                                    self.dividend,
                                )
                            case "rho":
                                z_option_feature[x_position, y_position] = self.rho(
                                    self.underlying_price_df.iloc[-1, 0],
                                    x_feature_value,
                                    y_feature_value,
                                    self.volatility,
                                    self.interest_rate,
                                    self.option_type,
                                    self.dividend,
                                )
                            case "theta":
                                z_option_feature[x_position, y_position] = self.theta(
                                    self.underlying_price_df.iloc[-1, 0],
                                    x_feature_value,
                                    y_feature_value,
                                    self.volatility,
                                    self.interest_rate,
                                    self.option_type,
                                    self.dividend,
                                )
                            case "vega":
                                z_option_feature[x_position, y_position] = self.vega(
                                    self.underlying_price_df.iloc[-1, 0],
                                    x_feature_value,
                                    y_feature_value,
                                    self.volatility,
                                    self.interest_rate,
                                    self.dividend,
                                )

            case "31" | "13":
                if matching_indices == "31":
                    switch_1 = self.x_min
                    self.x_min = self.y_min
                    self.y_min = switch_1
                    switch_1 = self.x_max
                    self.x_max = self.y_max
                    self.y_max = switch_1
                    switch_2 = x_features
                    x_features = y_feature
                    y_feature = switch_2
                    switch_3 = self.feature_in_x
                    self.feature_in_x = self.feature_in_y
                    self.feature_in_y = switch_3

                for x_position, x_feature_value in enumerate(x_features):

                    for y_position, y_feature_value in enumerate(y_feature):
                        match self.feature_in_z:
                            case "black scholes":
                                z_option_feature[x_position, y_position] = (
                                    self.black_scholes_plot(
                                        self.underlying_price_df.iloc[-1, 0],
                                        x_feature_value,
                                        self.number_of_days,
                                        y_feature_value,
                                        self.interest_rate,
                                        self.option_type,
                                        self.dividend,
                                    )
                                )
                            case "delta":
                                z_option_feature[x_position, y_position] = self.delta(
                                    self.underlying_price_df.iloc[-1, 0],
                                    x_feature_value,
                                    self.number_of_days,
                                    y_feature_value,
                                    self.interest_rate,
                                    self.option_type,
                                    self.dividend,
                                )
                            case "gamma":
                                z_option_feature[x_position, y_position] = self.gamma(
                                    self.underlying_price_df.iloc[-1, 0],
                                    x_feature_value,
                                    self.number_of_days,
                                    y_feature_value,
                                    self.interest_rate,
                                    self.dividend,
                                )
                            case "rho":
                                z_option_feature[x_position, y_position] = self.rho(
                                    self.underlying_price_df.iloc[-1, 0],
                                    x_feature_value,
                                    self.number_of_days,
                                    y_feature_value,
                                    self.interest_rate,
                                    self.option_type,
                                    self.dividend,
                                )
                            case "theta":
                                z_option_feature[x_position, y_position] = self.theta(
                                    self.underlying_price_df.iloc[-1, 0],
                                    x_feature_value,
                                    self.number_of_days,
                                    y_feature_value,
                                    self.interest_rate,
                                    self.option_type,
                                    self.dividend,
                                )
                            case "vega":
                                z_option_feature[x_position, y_position] = self.vega(
                                    self.underlying_price_df.iloc[-1, 0],
                                    x_feature_value,
                                    self.number_of_days,
                                    y_feature_value,
                                    self.interest_rate,
                                    self.dividend,
                                )

            case "41" | "14":
                if matching_indices == "41":
                    switch_1 = self.x_min
                    self.x_min = self.y_min
                    self.y_min = switch_1
                    switch_1 = self.x_max
                    self.x_max = self.y_max
                    self.y_max = switch_1
                    switch_2 = x_features
                    x_features = y_feature
                    y_feature = switch_2
                    switch_3 = self.feature_in_x
                    self.feature_in_x = self.feature_in_y
                    self.feature_in_y = switch_3

                for x_position, x_feature_value in enumerate(x_features):

                    for y_position, y_feature_value in enumerate(y_feature):
                        match self.feature_in_z:
                            case "black scholes":
                                z_option_feature[x_position, y_position] = (
                                    self.black_scholes_plot(
                                        self.underlying_price_df.iloc[-1, 0],
                                        x_feature_value,
                                        self.number_of_days,
                                        self.volatility,
                                        y_feature_value,
                                        self.option_type,
                                        self.dividend,
                                    )
                                )
                            case "delta":
                                z_option_feature[x_position, y_position] = self.delta(
                                    self.underlying_price_df.iloc[-1, 0],
                                    x_feature_value,
                                    self.number_of_days,
                                    self.volatility,
                                    y_feature_value,
                                    self.option_type,
                                    self.dividend,
                                )
                            case "gamma":
                                z_option_feature[x_position, y_position] = self.gamma(
                                    self.underlying_price_df.iloc[-1, 0],
                                    x_feature_value,
                                    self.number_of_days,
                                    self.volatility,
                                    y_feature_value,
                                    self.dividend,
                                )
                            case "rho":
                                z_option_feature[x_position, y_position] = self.rho(
                                    self.underlying_price_df.iloc[-1, 0],
                                    x_feature_value,
                                    self.number_of_days,
                                    self.volatility,
                                    y_feature_value,
                                    self.option_type,
                                    self.dividend,
                                )
                            case "theta":
                                z_option_feature[x_position, y_position] = self.theta(
                                    self.underlying_price_df.iloc[-1, 0],
                                    x_feature_value,
                                    self.number_of_days,
                                    self.volatility,
                                    y_feature_value,
                                    self.option_type,
                                    self.dividend,
                                )
                            case "vega":
                                z_option_feature[x_position, y_position] = self.vega(
                                    self.underlying_price_df.iloc[-1, 0],
                                    x_feature_value,
                                    self.number_of_days,
                                    self.volatility,
                                    y_feature_value,
                                    self.dividend,
                                )

            case "51" | "15":
                if matching_indices == "51":
                    switch_1 = self.x_min
                    self.x_min = self.y_min
                    self.y_min = switch_1
                    switch_1 = self.x_max
                    self.x_max = self.y_max
                    self.y_max = switch_1
                    switch_2 = x_features
                    x_features = y_feature
                    y_feature = switch_2
                    switch_3 = self.feature_in_x
                    self.feature_in_x = self.feature_in_y
                    self.feature_in_y = switch_3
                for x_position, x_feature_value in enumerate(x_features):
                    for y_position, y_feature_value in enumerate(y_feature):
                        match self.feature_in_z:
                            case "black scholes":
                                z_option_feature[x_position, y_position] = (
                                    self.black_scholes_plot(
                                        self.underlying_price_df.iloc[-1, 0],
                                        x_feature_value,
                                        self.number_of_days,
                                        self.volatility,
                                        self.interest_rate,
                                        self.option_type,
                                        y_feature_value,
                                    )
                                )
                            case "delta":
                                z_option_feature[x_position, y_position] = self.delta(
                                    self.underlying_price_df.iloc[-1, 0],
                                    x_feature_value,
                                    self.number_of_days,
                                    self.volatility,
                                    self.interest_rate,
                                    self.option_type,
                                    y_feature_value,
                                )
                            case "gamma":
                                z_option_feature[x_position, y_position] = self.gamma(
                                    self.underlying_price_df.iloc[-1, 0],
                                    x_feature_value,
                                    self.number_of_days,
                                    self.volatility,
                                    self.interest_rate,
                                    y_feature_value,
                                )
                            case "rho":
                                z_option_feature[x_position, y_position] = self.rho(
                                    self.underlying_price_df.iloc[-1, 0],
                                    x_feature_value,
                                    self.number_of_days,
                                    self.volatility,
                                    self.interest_rate,
                                    self.option_type,
                                    y_feature_value,
                                )
                            case "theta":
                                z_option_feature[x_position, y_position] = self.theta(
                                    self.underlying_price_df.iloc[-1, 0],
                                    x_feature_value,
                                    self.number_of_days,
                                    self.volatility,
                                    self.interest_rate,
                                    self.option_type,
                                    y_feature_value,
                                )
                            case "vega":
                                z_option_feature[x_position, y_position] = self.vega(
                                    self.underlying_price_df.iloc[-1, 0],
                                    x_feature_value,
                                    self.number_of_days,
                                    self.volatility,
                                    self.interest_rate,
                                    y_feature_value,
                                )

            case "23" | "32":
                if matching_indices == "32":
                    switch_1 = self.x_min
                    self.x_min = self.y_min
                    self.y_min = switch_1
                    switch_1 = self.x_max
                    self.x_max = self.y_max
                    self.y_max = switch_1
                    switch_2 = x_features
                    x_features = y_feature
                    y_feature = switch_2
                    switch_3 = self.feature_in_x
                    self.feature_in_x = self.feature_in_y
                    self.feature_in_y = switch_3

                for x_position, x_feature_value in enumerate(x_features):

                    for y_position, y_feature_value in enumerate(y_feature):
                        match self.feature_in_z:
                            case "black scholes":
                                z_option_feature[x_position, y_position] = (
                                    self.black_scholes_plot(
                                        self.underlying_price_df.iloc[-1, 0],
                                        self.strike,
                                        x_feature_value,
                                        y_feature_value,
                                        self.interest_rate,
                                        self.option_type,
                                        self.dividend,
                                    )
                                )
                            case "delta":
                                z_option_feature[x_position, y_position] = self.delta(
                                    self.underlying_price_df.iloc[-1, 0],
                                    self.strike,
                                    x_feature_value,
                                    y_feature_value,
                                    self.interest_rate,
                                    self.option_type,
                                    self.dividend,
                                )
                            case "gamma":
                                z_option_feature[x_position, y_position] = self.gamma(
                                    self.underlying_price_df.iloc[-1, 0],
                                    self.strike,
                                    x_feature_value,
                                    y_feature_value,
                                    self.interest_rate,
                                    self.dividend,
                                )
                            case "rho":
                                z_option_feature[x_position, y_position] = self.rho(
                                    self.underlying_price_df.iloc[-1, 0],
                                    self.strike,
                                    x_feature_value,
                                    y_feature_value,
                                    self.interest_rate,
                                    self.option_type,
                                    self.dividend,
                                )
                            case "theta":
                                z_option_feature[x_position, y_position] = self.theta(
                                    self.underlying_price_df.iloc[-1, 0],
                                    self.strike,
                                    x_feature_value,
                                    y_feature_value,
                                    self.interest_rate,
                                    self.option_type,
                                    self.dividend,
                                )
                            case "vega":
                                z_option_feature[x_position, y_position] = self.vega(
                                    self.underlying_price_df.iloc[-1, 0],
                                    self.strike,
                                    x_feature_value,
                                    y_feature_value,
                                    self.interest_rate,
                                    self.dividend,
                                )

            case "42" | "24":
                if matching_indices == "42":
                    switch_1 = self.x_min
                    self.x_min = self.y_min
                    self.y_min = switch_1
                    switch_1 = self.x_max
                    self.x_max = self.y_max
                    self.y_max = switch_1
                    switch_2 = x_features
                    x_features = y_feature
                    y_feature = switch_2
                    switch_3 = self.feature_in_x
                    self.feature_in_x = self.feature_in_y
                    self.feature_in_y = switch_3

                for x_position, x_feature_value in enumerate(x_features):

                    for y_position, y_feature_value in enumerate(y_feature):
                        match self.feature_in_z:
                            case "black scholes":
                                z_option_feature[x_position, y_position] = (
                                    self.black_scholes_plot(
                                        self.underlying_price_df.iloc[-1, 0],
                                        self.strike,
                                        x_feature_value,
                                        self.volatility,
                                        y_feature_value,
                                        self.option_type,
                                        self.dividend,
                                    )
                                )
                            case "delta":
                                z_option_feature[x_position, y_position] = self.delta(
                                    self.underlying_price_df.iloc[-1, 0],
                                    self.strike,
                                    x_feature_value,
                                    self.volatility,
                                    y_feature_value,
                                    self.option_type,
                                    self.dividend,
                                )
                            case "gamma":
                                z_option_feature[x_position, y_position] = self.gamma(
                                    self.underlying_price_df.iloc[-1, 0],
                                    self.strike,
                                    x_feature_value,
                                    self.volatility,
                                    y_feature_value,
                                    self.dividend,
                                )
                            case "rho":
                                z_option_feature[x_position, y_position] = self.rho(
                                    self.underlying_price_df.iloc[-1, 0],
                                    self.strike,
                                    x_feature_value,
                                    self.volatility,
                                    y_feature_value,
                                    self.option_type,
                                    self.dividend,
                                )
                            case "theta":
                                z_option_feature[x_position, y_position] = self.theta(
                                    self.underlying_price_df.iloc[-1, 0],
                                    self.strike,
                                    x_feature_value,
                                    self.volatility,
                                    y_feature_value,
                                    self.option_type,
                                    self.dividend,
                                )
                            case "vega":
                                z_option_feature[x_position, y_position] = self.vega(
                                    self.underlying_price_df.iloc[-1, 0],
                                    self.strike,
                                    x_feature_value,
                                    self.volatility,
                                    y_feature_value,
                                    self.dividend,
                                )

            case "25" | "52":
                if matching_indices == "52":
                    switch_1 = self.x_min
                    self.x_min = self.y_min
                    self.y_min = switch_1
                    switch_1 = self.x_max
                    self.x_max = self.y_max
                    self.y_max = switch_1
                    switch_2 = x_features
                    x_features = y_feature
                    y_feature = switch_2
                    switch_3 = self.feature_in_x
                    self.feature_in_x = self.feature_in_y
                    self.feature_in_y = switch_3

                for x_position, x_feature_value in enumerate(x_features):

                    for y_position, y_feature_value in enumerate(y_feature):
                        match self.feature_in_z:
                            case "black scholes":
                                z_option_feature[x_position, y_position] = (
                                    self.black_scholes_plot(
                                        self.underlying_price_df.iloc[-1, 0],
                                        self.strike,
                                        x_feature_value,
                                        self.volatility,
                                        self.interest_rate,
                                        self.option_type,
                                        y_feature_value,
                                    )
                                )
                            case "delta":
                                z_option_feature[x_position, y_position] = self.delta(
                                    self.underlying_price_df.iloc[-1, 0],
                                    self.strike,
                                    x_feature_value,
                                    self.volatility,
                                    self.interest_rate,
                                    self.option_type,
                                    y_feature_value,
                                )
                            case "gamma":
                                z_option_feature[x_position, y_position] = self.gamma(
                                    self.underlying_price_df.iloc[-1, 0],
                                    self.strike,
                                    x_feature_value,
                                    self.volatility,
                                    self.interest_rate,
                                    y_feature_value,
                                )
                            case "rho":
                                z_option_feature[x_position, y_position] = self.rho(
                                    self.underlying_price_df.iloc[-1, 0],
                                    self.strike,
                                    x_feature_value,
                                    self.volatility,
                                    self.interest_rate,
                                    self.option_type,
                                    y_feature_value,
                                )
                            case "theta":
                                z_option_feature[x_position, y_position] = self.theta(
                                    self.underlying_price_df.iloc[-1, 0],
                                    self.strike,
                                    x_feature_value,
                                    self.volatility,
                                    self.interest_rate,
                                    self.option_type,
                                    y_feature_value,
                                )
                            case "vega":
                                z_option_feature[x_position, y_position] = self.vega(
                                    self.underlying_price_df.iloc[-1, 0],
                                    self.strike,
                                    x_feature_value,
                                    self.volatility,
                                    self.interest_rate,
                                    y_feature_value,
                                )

            case "43" | "34":
                if matching_indices == "43":
                    switch_1 = self.x_min
                    self.x_min = self.y_min
                    self.y_min = switch_1
                    switch_1 = self.x_max
                    self.x_max = self.y_max
                    self.y_max = switch_1
                    switch_2 = x_features
                    x_features = y_feature
                    y_feature = switch_2
                    switch_3 = self.feature_in_x
                    self.feature_in_x = self.feature_in_y
                    self.feature_in_y = switch_3

                for x_position, x_feature_value in enumerate(x_features):

                    for y_position, y_feature_value in enumerate(y_feature):
                        match self.feature_in_z:
                            case "black scholes":
                                z_option_feature[x_position, y_position] = (
                                    self.black_scholes_plot(
                                        self.underlying_price_df.iloc[-1, 0],
                                        self.strike,
                                        self.number_of_days,
                                        x_feature_value,
                                        y_feature_value,
                                        self.option_type,
                                        self.dividend,
                                    )
                                )
                            case "delta":
                                z_option_feature[x_position, y_position] = self.delta(
                                    self.underlying_price_df.iloc[-1, 0],
                                    self.strike,
                                    self.number_of_days,
                                    x_feature_value,
                                    y_feature_value,
                                    self.option_type,
                                    self.dividend,
                                )
                            case "gamma":
                                z_option_feature[x_position, y_position] = self.gamma(
                                    self.underlying_price_df.iloc[-1, 0],
                                    self.strike,
                                    self.number_of_days,
                                    x_feature_value,
                                    y_feature_value,
                                    self.dividend,
                                )
                            case "rho":
                                z_option_feature[x_position, y_position] = self.rho(
                                    self.underlying_price_df.iloc[-1, 0],
                                    self.strike,
                                    self.number_of_days,
                                    x_feature_value,
                                    y_feature_value,
                                    self.option_type,
                                    self.dividend,
                                )
                            case "theta":
                                z_option_feature[x_position, y_position] = self.theta(
                                    self.underlying_price_df.iloc[-1, 0],
                                    self.strike,
                                    self.number_of_days,
                                    x_feature_value,
                                    y_feature_value,
                                    self.option_type,
                                    self.dividend,
                                )
                            case "vega":
                                z_option_feature[x_position, y_position] = self.vega(
                                    self.underlying_price_df.iloc[-1, 0],
                                    self.strike,
                                    self.number_of_days,
                                    x_feature_value,
                                    y_feature_value,
                                    self.dividend,
                                )

            case "35" | "53":
                if matching_indices == "53":
                    switch_1 = self.x_min
                    self.x_min = self.y_min
                    self.y_min = switch_1
                    switch_1 = self.x_max
                    self.x_max = self.y_max
                    self.y_max = switch_1
                    switch_2 = x_features
                    x_features = y_feature
                    y_feature = switch_2
                    switch_3 = self.feature_in_x
                    self.feature_in_x = self.feature_in_y
                    self.feature_in_y = switch_3

                for x_position, x_feature_value in enumerate(x_features):

                    for y_position, y_feature_value in enumerate(y_feature):
                        match self.feature_in_z:
                            case "black scholes":
                                z_option_feature[x_position, y_position] = (
                                    self.black_scholes_plot(
                                        self.underlying_price_df.iloc[-1, 0],
                                        self.strike,
                                        self.number_of_days,
                                        x_feature_value,
                                        self.interest_rate,
                                        self.option_type,
                                        y_feature_value,
                                    )
                                )
                            case "delta":
                                z_option_feature[x_position, y_position] = self.delta(
                                    self.underlying_price_df.iloc[-1, 0],
                                    self.strike,
                                    self.number_of_days,
                                    x_feature_value,
                                    self.interest_rate,
                                    self.option_type,
                                    y_feature_value,
                                )
                            case "gamma":
                                z_option_feature[x_position, y_position] = self.gamma(
                                    self.underlying_price_df.iloc[-1, 0],
                                    self.strike,
                                    self.number_of_days,
                                    x_feature_value,
                                    self.interest_rate,
                                    y_feature_value,
                                )
                            case "rho":
                                z_option_feature[x_position, y_position] = self.rho(
                                    self.underlying_price_df.iloc[-1, 0],
                                    self.strike,
                                    self.number_of_days,
                                    x_feature_value,
                                    self.interest_rate,
                                    self.option_type,
                                    y_feature_value,
                                )
                            case "theta":
                                z_option_feature[x_position, y_position] = self.theta(
                                    self.underlying_price_df.iloc[-1, 0],
                                    self.strike,
                                    self.number_of_days,
                                    x_feature_value,
                                    self.interest_rate,
                                    self.option_type,
                                    y_feature_value,
                                )
                            case "vega":
                                z_option_feature[x_position, y_position] = self.vega(
                                    self.underlying_price_df.iloc[-1, 0],
                                    self.strike,
                                    self.number_of_days,
                                    x_feature_value,
                                    self.interest_rate,
                                    y_feature_value,
                                )

            case "54" | "45":
                if matching_indices == "54":
                    switch_1 = self.x_min
                    self.x_min = self.y_min
                    self.y_min = switch_1
                    switch_1 = self.x_max
                    self.x_max = self.y_max
                    self.y_max = switch_1
                    switch_2 = x_features
                    x_features = y_feature
                    y_feature = switch_2
                    switch_3 = self.feature_in_x
                    self.feature_in_x = self.feature_in_y
                    self.feature_in_y = switch_3

                for x_position, x_feature_value in enumerate(x_features):

                    for y_position, y_feature_value in enumerate(y_feature):
                        match self.feature_in_z:
                            case "black scholes":
                                z_option_feature[x_position, y_position] = (
                                    self.black_scholes_plot(
                                        self.underlying_price_df.iloc[-1, 0],
                                        self.strike,
                                        self.number_of_days,
                                        self.volatility,
                                        x_feature_value,
                                        self.option_type,
                                        y_feature_value,
                                    )
                                )
                            case "delta":
                                z_option_feature[x_position, y_position] = self.delta(
                                    self.underlying_price_df.iloc[-1, 0],
                                    self.strike,
                                    self.number_of_days,
                                    self.volatility,
                                    x_feature_value,
                                    self.option_type,
                                    y_feature_value,
                                )
                            case "gamma":
                                z_option_feature[x_position, y_position] = self.gamma(
                                    self.underlying_price_df.iloc[-1, 0],
                                    self.strike,
                                    self.number_of_days,
                                    self.volatility,
                                    x_feature_value,
                                    y_feature_value,
                                )
                            case "rho":
                                z_option_feature[x_position, y_position] = self.rho(
                                    self.underlying_price_df.iloc[-1, 0],
                                    self.strike,
                                    self.number_of_days,
                                    self.volatility,
                                    x_feature_value,
                                    self.option_type,
                                    y_feature_value,
                                )
                            case "theta":
                                z_option_feature[x_position, y_position] = self.theta(
                                    self.underlying_price_df.iloc[-1, 0],
                                    self.strike,
                                    self.number_of_days,
                                    self.volatility,
                                    x_feature_value,
                                    self.option_type,
                                    y_feature_value,
                                )
                            case "vega":
                                z_option_feature[x_position, y_position] = self.vega(
                                    self.underlying_price_df.iloc[-1, 0],
                                    self.strike,
                                    self.number_of_days,
                                    self.volatility,
                                    x_feature_value,
                                    y_feature_value,
                                )

            case _:
                print("error")
        x_features_mech, y_feature_mech = np.meshgrid(x_features, y_feature)
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        ax.plot_surface(
            x_features_mech,
            y_feature_mech,
            z_option_feature.transpose(),
            cmap="viridis",
        )  # Plot the 3D curve
        ax.set_xlabel(f"{self.feature_in_x} feature")
        ax.set_ylabel(f"{self.feature_in_y} feature")
        ax.set_zlabel(f"{self.feature_in_z} option")
        ax.set_title(
            f"Plot of how {self.feature_in_z} option behave if {self.feature_in_x} and {self.feature_in_y} features variate"
        )
        plt.show()
        self.image = np.meshgrid(x_features, y_feature), z_option_feature.transpose()



# ["black scholes", "delta", "gamma", "rho", "theta", "vega"]
# [
#             "Stock Price",
#             "Strike",
#             "Maturity",
#             "Volatility",
#             "Interest rate",
#             "Dividend",
#         ]
# p = OptionFeaturesPlot(
#     190,
#     "AAPL",
#     100,
#     "Call",
#     "2024-10-15",
#     0.05,
#     0.005,
#     0.3,
#     0,
#     300,
#     "Strike",
#     0,
#     300,
#     "Stock Price",
#     "black scholes",
#     "3d",
# )
# p.compute_option_contract()
