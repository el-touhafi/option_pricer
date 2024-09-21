import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

sys.path.append("..")
from scipy.stats import norm
from OptionMother import OptionMother


class EuropeenOption(OptionMother):
    """
    This class is inherited from OptionMother class, this class is for computing underlying europeen option's price.
    I will introduce the ambigious variables in this part:
    underlying_ticker : it's the stock ticker of any firm that exists in yahoo finance website or api (eg 'apple stock' : 'AAPL')
    steps_number : number of stepsin the tree (we can find convergence to the Black and Scholes vanilla option pricing model in 50 steps)
    option_type : 'Call' or 'Put'
    expiry_date : this date will serve to customize you time to maturity
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
        interest_rate: float,
        dividend: float,
        volatility: float,
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
        super().compute_option_contract()

    def cox_ross_rubinstein(self):
        """
        Conditionnal expectation formula :
        
        C_t = e^(-risk_free_rate * (time_to_expiration - current_time)) * E[option_payoff(stock_price_at_expiration - strike_price) | information_up_to_current_time]

        Explicit formula : 
        
        C_t = e^(-risk_free_rate * (time_to_expiration - current_time)) * sum(k=0 to number_of_time_steps) [ Combination(number_of_time_steps, k) * probability_up^k * probability_down^(number_of_time_steps-k) * option_payoff(stock_price * up_factor^k * down_factor^(number_of_time_steps-k) - strike_price) ]
        """
        time_delta = self.number_of_days / self.steps_number
        up_percentage = np.exp(self.volatility * np.sqrt(time_delta))
        down_percentage = 1 / up_percentage
        upper_probability = (
            np.exp((self.interest_rate - self.dividend) * time_delta) - down_percentage
        ) / (up_percentage - down_percentage)
        discount = np.exp(-self.interest_rate * time_delta)

        underlying_price_tree = (
            float(self.underlying_price_df.iloc[-1, 0])
            * down_percentage ** pd.Series(list(range(self.steps_number, -1, -1)))
            * up_percentage ** pd.Series(list(range(0, self.steps_number + 1, 1)))
        )
        if self.option_type == "Call":
            option_price_tree = pd.Series(
                np.maximum(
                    (underlying_price_tree - self.strike).tolist(),
                    np.zeros(self.steps_number + 1),
                )
            ).tolist()
        else:
            option_price_tree = pd.Series(
                np.maximum(
                    (self.strike - underlying_price_tree).tolist(),
                    np.zeros(self.steps_number + 1),
                )
            ).tolist()

        for step in range(self.steps_number, 0, -1):
            option_price_tree = (
                discount
                * (
                    upper_probability * pd.Series(option_price_tree[1 : step + 1])
                    + (1 - upper_probability) * pd.Series(option_price_tree[0:step])
                )
            ).tolist()
        return option_price_tree[0]

    def trinomial_model(self):
        """
        Conditionnal expectation formula :

        OptionPrice(S, t) = exp(-riskFreeRate * timeStep) * [upProbability * OptionPrice(upPrice, t + timeStep) + downProbability * OptionPrice(downPrice, t + timeStep) + middleProbability * OptionPrice(middlePrice, t + timeStep)]
        """
        time_delta = self.number_of_days / self.steps_number
        up_percentage = np.exp(self.volatility * np.sqrt(2 * time_delta))
        down_percentage = 1 / up_percentage
        temporary_var1 = np.exp(self.interest_rate * time_delta / 2)
        temporary_var2 = np.exp(self.volatility * np.sqrt(time_delta / 2))
        upper_probability = (
            (temporary_var1 - 1 / temporary_var2)
            / (temporary_var2 - 1 / temporary_var2)
        ) ** 2
        down_probability = (
            (temporary_var2 - temporary_var1) / (temporary_var2 - 1 / temporary_var2)
        ) ** 2
        no_change_probability = 1 - upper_probability - down_probability
        discount = np.exp(-self.interest_rate * time_delta)

        underlying_price_tree = (
            float(self.underlying_price_df.iloc[-1, 0])
            * down_percentage
            ** pd.Series(list(np.arange(self.steps_number, -1 / 2, -1 / 2)))
            * up_percentage
            ** pd.Series(list(np.arange(0, self.steps_number + 1 / 2, 1 / 2)))
        )
        if self.option_type == "Call":
            option_price_tree = pd.Series(
                np.maximum(
                    (underlying_price_tree - self.strike).tolist(),
                    np.zeros(2 * self.steps_number + 1),
                )
            ).tolist()
        else:
            option_price_tree = pd.Series(
                np.maximum(
                    (self.strike - underlying_price_tree).tolist(),
                    np.zeros(2 * self.steps_number + 1),
                )
            ).tolist()

        for step in range(2 * self.steps_number, 0, -2):
            option_price_tree = (
                discount
                * (
                    upper_probability * pd.Series(option_price_tree[2 : step + 1])
                    + (down_probability * pd.Series(option_price_tree[0 : step - 1]))
                    + (no_change_probability * pd.Series(option_price_tree[1:step]))
                )
            ).tolist()
        return option_price_tree[0]

    def black_scholes_model(self):
        """
        Black-Scholes Formula for European Call Option :

        CallOptionPrice = S0 * CDF(d1) - X * exp(-riskFreeRate * (timeToExpiration - currentTime)) * CDF(d2)

        Black-Scholes Formula for European Put Option:

        PutOptionPrice = X * exp(-riskFreeRate * (timeToExpiration - currentTime)) * CDF(-d2) - S0 * CDF(-d1)

        Variable d1 in the Black-Scholes Model:

        d1 = (ln(S0 / X) + (riskFreeRate + (volatility ** 2) / 2) * (timeToExpiration - currentTime)) / (volatility * sqrt(timeToExpiration - currentTime))

        Variable d2 in the Black-Scholes Model:

        d2 = d1 - volatility * sqrt(timeToExpiration - currentTime)
        """
        d1 = (1 / (self.volatility * np.sqrt(self.number_of_days))) * (
            np.log(float(self.underlying_price_df.iloc[-1, 0]) / self.strike)
            + (self.interest_rate - self.dividend + 0.5 * (self.volatility) ** 2)
            * self.number_of_days
        )
        d2 = d1 - self.volatility * np.sqrt(self.number_of_days)

        if self.option_type == "Call":
            return float(self.underlying_price_df.iloc[-1, 0]) * norm.cdf(
                d1
            ) - self.strike * np.exp(
                -self.interest_rate * self.number_of_days
            ) * norm.cdf(
                d2
            )
        elif self.option_type == "Put":
            return -float(self.underlying_price_df.iloc[-1, 0]) * norm.cdf(
                -d1
            ) + self.strike * np.exp(
                -self.interest_rate * self.number_of_days
            ) * norm.cdf(
                -d2
            )

    def delta(self):
        """
        This method will serve to compute the sensitivity of the option price to the underlying price.

        It's like the percentage of how the price of the option will perform if the underlying fluctuate.

        Delta = CDF(d1)

        NB : We can refer to it like the speed of the option price trying to catch the underlying price.
        """
        d1 = (1 / (self.volatility * np.sqrt(self.number_of_days))) * (
            np.log(self.underlying_price_df.iloc[-1, 0] / self.strike)
            + ((self.interest_rate - self.dividend) + 0.5 * (self.volatility) ** 2)
            * self.number_of_days
        )
        return (
            np.exp(-self.dividend * self.number_of_days) * norm.cdf(d1)
            if self.option_type == "Call"
            else -np.exp(-self.dividend * self.number_of_days) * norm.cdf(-d1)
        )

    def gamma(self):
        """
        This method will serve to compute the sensitivity of the option delta price to the underlying price.

        It's like the accelerationn of how the price of the option will move based on it's delta(speed) if the underlying fluctuate.

        Gamma = PDF(d1) / (S0 * volatility * sqrt(timeToExpiration - currentTime))
        """
        d1 = (1 / (self.volatility * np.sqrt(self.number_of_days))) * (
            np.log(self.underlying_price_df.iloc[-1, 0] / self.strike)
            + ((self.interest_rate - self.dividend) + 0.5 * (self.volatility) ** 2)
            * self.number_of_days
        )
        return (
            np.exp(-self.dividend * self.number_of_days)
            * norm.pdf(d1)
            / (
                self.underlying_price_df.iloc[-1, 0]
                * self.volatility
                * np.sqrt(self.number_of_days)
            )
        )

    def theta(self):
        """
        This method will serve to compute the sensibility of the option price and the days to expiry.

        Theta_Call = -(S0 * PDF(d1) * volatility) / (2 * sqrt(timeToExpiration - currentTime)) - riskFreeRate * X * exp(-riskFreeRate * (timeToExpiration - currentTime)) * CDF(d2)

        The theta indicate how the time value variate during the hold of the option contract.
        """
        d1 = (1 / (self.volatility * np.sqrt(self.number_of_days))) * (
            np.log(self.underlying_price_df.iloc[-1, 0] / self.strike)
            + ((self.interest_rate - self.dividend) + 0.5 * (self.volatility) ** 2)
            * self.number_of_days
        )
        d2 = d1 - self.volatility * np.sqrt(self.number_of_days)
        call = (
            -np.exp(-self.dividend * self.number_of_days)
            * self.underlying_price_df.iloc[-1, 0]
            * norm.pdf(d1, 0, 1)
            * self.volatility
            / (2 * np.sqrt(self.number_of_days))
            - self.interest_rate
            * self.strike
            * np.exp(-self.interest_rate * self.number_of_days)
            * norm.cdf(d2, 0, 1)
            + self.dividend
            * self.underlying_price_df.iloc[-1, 0]
            * np.exp(-self.dividend * self.number_of_days)
            * norm.cdf(d1, 0, 1)
        )
        put = (
            (
                -np.exp(-self.dividend * self.number_of_days)
                * self.underlying_price_df.iloc[-1, 0]
                * self.volatility
                * norm.pdf(d1)
            )
            / (2 * np.sqrt(self.number_of_days))
            + self.interest_rate
            * self.strike
            * np.exp(-self.interest_rate * self.number_of_days)
            * norm.cdf(-d2)
            - self.dividend
            * self.underlying_price_df.iloc[-1, 0]
            * np.exp(-self.dividend * self.number_of_days)
            * norm.cdf(-d1, 0, 1)
        )
        return call if self.option_type == "Call" else put

    def vega(self):
        """
        This method will serve to compute the sensibility of the option price to the implied volatility of the underlying price.

        Vega = S0 * sqrt(timeToExpiration - currentTime) * PDF(d1)
        """
        d1 = (1 / (self.volatility * np.sqrt(self.number_of_days))) * (
            np.log(self.underlying_price_df.iloc[-1, 0] / self.strike)
            + ((self.interest_rate - self.dividend) + 0.5 * (self.volatility) ** 2)
            * self.number_of_days
        )
        return (
            np.exp(-self.dividend * self.number_of_days)
            * self.underlying_price_df.iloc[-1, 0]
            * norm.pdf(d1)
            * np.sqrt(self.number_of_days)
        )

    def rho(self):
        """
        This method will serve to compute the sensibility of the option price to the interest rate.

        Rho = X * (timeToExpiration - currentTime) * exp(-riskFreeRate * (timeToExpiration - currentTime)) * CDF(d2)
        """
        d1 = (1 / (self.volatility * np.sqrt(self.number_of_days))) * (
            np.log(self.underlying_price_df.iloc[-1, 0] / self.strike)
            + ((self.interest_rate - self.dividend) + 0.5 * (self.volatility) ** 2)
            * self.number_of_days
        )
        d2 = d1 - self.volatility * np.sqrt(self.number_of_days)
        call = (
            self.strike
            * self.number_of_days
            * np.exp(-self.interest_rate * self.number_of_days)
            * norm.cdf(d2, 0, 1)
        )
        put = (
            -self.strike
            * self.number_of_days
            * np.exp(-self.interest_rate * self.number_of_days)
            * norm.cdf(-d2, 0, 1)
        )
        return call if self.option_type == "Call" else put

    def plot_binomial_tree(self, f):
        """
        This method will serve to plot the binomial(Cox-Ross-Rubinstein) model tree of option price
        """
        if self.steps_number > 10:
            self.steps_number = 10
        time_delta = self.number_of_days / self.steps_number
        up_percentage = np.exp(self.volatility * np.sqrt(time_delta))
        down_percentage = 1 / up_percentage
        upper_probability = (
            np.exp((self.interest_rate - self.dividend) * time_delta) - down_percentage
        ) / (up_percentage - down_percentage)
        discount = np.exp(-self.interest_rate * time_delta)

        underlying_price_tree = (
            float(self.underlying_price_df.iloc[-1, 0])
            * down_percentage ** pd.Series(list(range(self.steps_number, -1, -1)))
            * up_percentage ** pd.Series(list(range(0, self.steps_number + 1, 1)))
        )
        euro_option_price_tree = []
        if self.option_type == "Call":
            option_price_tree = pd.Series(
                np.maximum(
                    (underlying_price_tree - self.strike).tolist(),
                    np.zeros(self.steps_number + 1),
                )
            ).tolist()
            euro_option_price_tree.append(option_price_tree)
        else:
            option_price_tree = pd.Series(
                np.maximum(
                    (self.strike - underlying_price_tree).tolist(),
                    np.zeros(self.steps_number + 1),
                )
            ).tolist()
            euro_option_price_tree.append(option_price_tree)

        for step in range(self.steps_number, 0, -1):
            option_price_tree = (
                discount
                * (
                    upper_probability * pd.Series(option_price_tree[1 : step + 1])
                    + (1 - upper_probability) * pd.Series(option_price_tree[0:step])
                )
            ).tolist()
            euro_option_price_tree.append(option_price_tree)

        binomial_tree_graph = nx.DiGraph()
        for step in range(self.steps_number, -self.steps_number - 1, -1):
            for mid_step in range(step, -step - 1, -2):
                k = np.arange(0, step + 1, 1 / 2)
                binomial_tree_graph.add_node(
                    (step, mid_step),
                    label=f"{round(euro_option_price_tree[self.steps_number - step][step - int(k[step - mid_step])], 2)}",
                )
                if step < self.steps_number:
                    binomial_tree_graph.add_edge(
                        (step, mid_step),
                        (step + 1, mid_step + 1),
                        label=round(upper_probability, 3),
                    )
                    binomial_tree_graph.add_edge(
                        (step, mid_step),
                        (step + 1, mid_step - 1),
                        label=round(1 - upper_probability, 3),
                    )
        position = {node: node for node in binomial_tree_graph.nodes()}
        labels = {
            node: binomial_tree_graph.nodes[node]["label"]
            for node in binomial_tree_graph.nodes()
        }
        plt.figure(figsize=(10, 10))
        nx.draw(
            binomial_tree_graph,
            position,
            with_labels=True,
            labels=labels,
            node_size=400,
            node_color="lightblue",
            font_size=8,
            font_color="black",
        )
        edge_labels = nx.get_edge_attributes(binomial_tree_graph, "label")
        nx.draw_networkx_edge_labels(
            binomial_tree_graph, position, edge_labels=edge_labels, font_size=8
        )
        plt.title("Cox-Ross-Rubinstein tree for option pricing -Eur Case-")
        plt.axis("off")
        plt.show()
        if f:
            plt.savefig(f, format="png")

    def plot_trinomial_tree(self, f):
        """
        This method will serve to plot the trinomial model tree of option price
        """
        if self.steps_number > 10:
            self.steps_number = 10
        time_delta = self.number_of_days / self.steps_number
        up_percentage = np.exp(self.volatility * np.sqrt(2 * time_delta))
        down_percentage = 1 / up_percentage
        temporary_var1 = np.exp(self.interest_rate * time_delta / 2)
        temporary_var2 = np.exp(self.volatility * np.sqrt(time_delta / 2))
        upper_probability = (
            (temporary_var1 - 1 / temporary_var2)
            / (temporary_var2 - 1 / temporary_var2)
        ) ** 2
        down_probability = (
            (temporary_var2 - temporary_var1) / (temporary_var2 - 1 / temporary_var2)
        ) ** 2
        no_change_probability = 1 - upper_probability - down_probability
        discount = np.exp(-self.interest_rate * time_delta)
        underlying_price_tree = (
            float(self.underlying_price_df.iloc[-1, 0])
            * down_percentage
            ** pd.Series(list(np.arange(self.steps_number, -1 / 2, -1 / 2)))
            * up_percentage
            ** pd.Series(list(np.arange(0, self.steps_number + 1 / 2, 1 / 2)))
        )
        euro_option_price_tree = []
        if self.option_type == "Call":
            option_price_tree = pd.Series(
                np.maximum(
                    (underlying_price_tree - self.strike).tolist(),
                    np.zeros(2 * self.steps_number + 1),
                )
            ).tolist()
        else:
            option_price_tree = pd.Series(
                np.maximum(
                    (self.strike - underlying_price_tree).tolist(),
                    np.zeros(2 * self.steps_number + 1),
                )
            ).tolist()
        euro_option_price_tree.append(option_price_tree)
        for step in range(2 * self.steps_number, 0, -2):
            option_price_tree = (
                discount
                * (
                    upper_probability * pd.Series(option_price_tree[2 : step + 1])
                    + (down_probability * pd.Series(option_price_tree[0 : step - 1]))
                    + (no_change_probability * pd.Series(option_price_tree[1:step]))
                )
            ).tolist()
            euro_option_price_tree.append(option_price_tree)
        trinom_tree_graph = nx.DiGraph()
        for step in range(self.steps_number, -self.steps_number - 1, -1):
            for mid_step in range(step, -step - 1, -1):
                k = np.arange(0, 2 * step + 1, 1)
                trinom_tree_graph.add_node(
                    (step, mid_step),
                    label=f"{round(euro_option_price_tree[self.steps_number-step][int(k[step + mid_step])], 2)}",
                )
                if step < self.steps_number:
                    trinom_tree_graph.add_edge(
                        (step, mid_step),
                        (step + 1, mid_step + 1),
                        label=round(upper_probability, 3),
                    )
                    trinom_tree_graph.add_edge(
                        (step, mid_step),
                        (step + 1, mid_step - 1),
                        label=round(down_probability, 3),
                    )
                    trinom_tree_graph.add_edge(
                        (step, mid_step),
                        (step + 1, mid_step),
                        label=round(no_change_probability, 3),
                    )
        position = {node: node for node in trinom_tree_graph.nodes()}
        labels = {
            node: trinom_tree_graph.nodes[node]["label"]
            for node in trinom_tree_graph.nodes()
        }
        plt.figure(figsize=(10, 10))
        nx.draw(
            trinom_tree_graph,
            position,
            with_labels=True,
            labels=labels,
            node_size=400,
            node_color="lightblue",
            font_size=8,
            font_color="black",
        )
        edge_labels = nx.get_edge_attributes(trinom_tree_graph, "label")
        nx.draw_networkx_edge_labels(
            trinom_tree_graph, position, edge_labels=edge_labels, font_size=8
        )
        plt.title("Boyle tree for option pricing -Eur Case-")
        plt.axis("off")
        plt.show()
        if f:
            plt.savefig(f, format="png")

    def compute_option_contract(self):
        super().compute_option_contract()
        self.crr = self.cox_ross_rubinstein()
        self.trinom = self.trinomial_model()
        self.black_scholes = self.black_scholes_model()

p = EuropeenOption(65000, "BTC-USD", 100, "Put", "2024-12-27", 0.05, 0, 0.8)
p.compute_option_contract()
p.plot_trinomial_tree(None)
p.plot_binomial_tree(None)
