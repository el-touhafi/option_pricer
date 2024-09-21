import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import sys

sys.path.append("..")
from OptionMother import OptionMother


class AmericanOption(OptionMother):
    """
    This class is inherited from OptionMother class, this class is for
    computing underlying american option's price.
    I will introduce the ambigious variables in this part:
    underlying_ticker : it's the stock ticker of any firm that exists in yahoo finance website or api
    (eg 'apple stock' : 'AAPL')
    steps_number : number of stepsin the tree (we can find convergence to the Black and Scholes vanilla option pricing
    model in 50 steps)
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

    def cox_ross_rubinstein(self):
        """
        Conditionnal expectation formula :

        C_t = e^(-risk_free_rate * (time_to_expiration - current_time))
        * E[option_payoff(stock_price_at_expiration - strike_price) | information_up_to_current_time]

        Explicit formula :

        C_t = e^(-risk_free_rate * (time_to_expiration - current_time))
        * sum(k=0 to number_of_time_steps)
        [ Combination(number_of_time_steps, k) * probability_up^k * probability_down^(number_of_time_steps-k)
        * option_payoff(stock_price * up_factor^k * down_factor^(number_of_time_steps-k) - strike_price) ]

        For the american option and specially the american put option we should capture in each step of
        the tree the maximum of the europeen option price and the payoff.
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
            call_price_tree = pd.Series(
                np.maximum(
                    (underlying_price_tree - self.strike).tolist(),
                    np.zeros(self.steps_number + 1),
                )
            ).tolist()
            for step in range(self.steps_number - 1, -1, -1):
                underlying_price_tree = (
                    float(self.underlying_price_df.iloc[-1, 0])
                    * down_percentage ** pd.Series(list(range(step, -1, -1)))
                    * up_percentage ** pd.Series(list(range(0, step + 1, 1)))
                )
                call_price_tree[: step + 1] = (
                    discount
                    * (
                        upper_probability * pd.Series(call_price_tree[1 : step + 2])
                        + (1 - upper_probability)
                        * pd.Series(call_price_tree[0 : step + 1])
                    )
                ).tolist()
                call_price_tree = call_price_tree[:-1]
                call_price_tree = pd.Series(
                    np.maximum(
                        (underlying_price_tree - self.strike).tolist(), call_price_tree
                    )
                ).tolist()
            try:
                return call_price_tree[0]
            except call_price_tree[0] as nonvalue:
                if nonvalue is None:
                    print(
                        "check your variables if they are compatible with the model "
                        "or maybe it's a data importing problem"
                    )
        else:
            put_price_tree = pd.Series(
                np.maximum(
                    (self.strike - underlying_price_tree).tolist(),
                    np.zeros(self.steps_number + 1),
                )
            ).tolist()
            for step in range(self.steps_number - 1, -1, -1):
                underlying_price_tree = (
                    float(self.underlying_price_df.iloc[-1, 0])
                    * down_percentage ** pd.Series(list(range(step, -1, -1)))
                    * up_percentage ** pd.Series(list(range(0, step + 1, 1)))
                )
                put_price_tree[: step + 1] = (
                    discount
                    * (
                        upper_probability * pd.Series(put_price_tree[1 : step + 2])
                        + (1 - upper_probability)
                        * pd.Series(put_price_tree[0 : step + 1])
                    )
                ).tolist()
                put_price_tree = put_price_tree[:-1]
                put_price_tree = pd.Series(
                    np.maximum(
                        (self.strike - underlying_price_tree).tolist(), put_price_tree
                    )
                ).tolist()
            try:
                return put_price_tree[0]
            except put_price_tree[0] as nonvalue:
                if nonvalue is None:
                    print(
                        "check your variables if they are compatible with the model "
                        "or maybe it's a data importing problem"
                    )

    def trinomial_model(self):
        """
        Conditionnal expectation formula :

        OptionPrice(S, t) = exp(-riskFreeRate * timeStep) *
        [upProbability * OptionPrice(upPrice, t + timeStep) + downProbability * OptionPrice(downPrice, t + timeStep)
        + middleProbability * OptionPrice(middlePrice, t + timeStep)]

        For the american option and specially the american put option we should capture in each step of
        the tree the maximum of the europeen option price and the payoff.
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
            call_price_tree = pd.Series(
                np.maximum(
                    (underlying_price_tree - self.strike).tolist(),
                    np.zeros(2 * self.steps_number + 1),
                )
            ).tolist()
            for step in range(self.steps_number - 1, -1, -1):
                underlying_price_tree = (
                    float(self.underlying_price_df.iloc[-1, 0])
                    * down_percentage
                    ** pd.Series(list(np.arange(step, -1 / 2, -1 / 2)))
                    * up_percentage
                    ** pd.Series(list(np.arange(0, step + 1 / 2, 1 / 2)))
                )
                call_price_tree[: 2 * step + 1] = (
                    discount
                    * (
                        upper_probability * pd.Series(call_price_tree[0 : 2 * step + 1])
                        + (
                            down_probability
                            * pd.Series(call_price_tree[2 : 2 * step + 3])
                        )
                        + (
                            no_change_probability
                            * pd.Series(call_price_tree[1 : 2 * step + 2])
                        )
                    )
                ).tolist()
                call_price_tree = call_price_tree[: 2 * step + 1]
                call_price_tree = pd.Series(
                    np.maximum(
                        (underlying_price_tree - self.strike).tolist(), call_price_tree
                    )
                ).tolist()
            return call_price_tree[0]
        else:
            put_price_tree = pd.Series(
                np.maximum(
                    (self.strike - underlying_price_tree).tolist(),
                    np.zeros(2 * self.steps_number + 1),
                )
            ).tolist()
            for step in range(self.steps_number - 1, -1, -1):
                underlying_price_tree = (
                    float(self.underlying_price_df.iloc[-1, 0])
                    * down_percentage
                    ** pd.Series(list(np.arange(step, -1 / 2, -1 / 2)))
                    * up_percentage
                    ** pd.Series(list(np.arange(0, step + 1 / 2, 1 / 2)))
                )
                put_price_tree[: 2 * step + 1] = (
                    discount
                    * (
                        upper_probability * pd.Series(put_price_tree[0 : 2 * step + 1])
                        + (
                            down_probability
                            * pd.Series(put_price_tree[2 : 2 * step + 3])
                        )
                        + (
                            no_change_probability
                            * pd.Series(put_price_tree[1 : 2 * step + 2])
                        )
                    )
                ).tolist()
                put_price_tree = put_price_tree[: 2 * step + 1]
                put_price_tree = pd.Series(
                    np.maximum(
                        (self.strike - underlying_price_tree).tolist(), put_price_tree
                    )
                ).tolist()
            return put_price_tree[0]

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
        payoff_option_tree = []
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
        euro_option_price_tree.append(option_price_tree)
        payoff_option_tree.append(option_price_tree)

        for step in range(self.steps_number - 1, -1, -1):
            underlying_price_tree = (
                float(self.underlying_price_df.iloc[-1, 0])
                * down_percentage ** pd.Series(list(range(step, -1, -1)))
                * up_percentage ** pd.Series(list(range(0, step + 1, 1)))
            )
            option_price_tree[: step + 1] = (
                discount
                * (
                    upper_probability * pd.Series(option_price_tree[1 : step + 2])
                    + (1 - upper_probability)
                    * pd.Series(option_price_tree[0 : step + 1])
                )
            ).tolist()
            option_price_tree = option_price_tree[:-1]
            if self.option_type == "Call":
                payoff_option_tree.append(
                    (underlying_price_tree - self.strike).tolist()
                )
                euro_option_price_tree.append(option_price_tree)
                option_price_tree = pd.Series(
                    np.maximum(
                        (underlying_price_tree - self.strike).tolist(),
                        option_price_tree,
                    )
                ).tolist()
            else:
                payoff_option_tree.append(
                    (self.strike - underlying_price_tree).tolist()
                )
                euro_option_price_tree.append(option_price_tree)
                option_price_tree = pd.Series(
                    np.maximum(
                        (self.strike - underlying_price_tree).tolist(),
                        option_price_tree,
                    )
                ).tolist()
        binom_tree_graph = nx.DiGraph()
        for step in range(self.steps_number, -self.steps_number - 1, -1):
            for mid_step in range(step, -step - 1, -2):
                k = np.arange(0, step + 1, 1 / 2)
                if (
                    euro_option_price_tree[self.steps_number - step][
                        int(k[step + mid_step])
                    ]
                    < payoff_option_tree[self.steps_number - step][
                        int(k[step + mid_step])
                    ]
                ):
                    state = "exercé"
                else:
                    state = ""
                binom_tree_graph.add_node(
                    (step, mid_step),
                    label=f"{round(euro_option_price_tree[self.steps_number - step][int(k[step + mid_step])], 3)}\n"
                    f"{round(payoff_option_tree[self.steps_number - step][int(k[step + mid_step])], 3)}\n\n{state}",
                )
                if step < self.steps_number:
                    binom_tree_graph.add_edge(
                        (step, mid_step),
                        (step + 1, mid_step + 1),
                        label=round(upper_probability, 2),
                    )
                    binom_tree_graph.add_edge(
                        (step, mid_step),
                        (step + 1, mid_step - 1),
                        label=round(1 - upper_probability, 2),
                    )
        position = {node: node for node in binom_tree_graph.nodes()}
        labels = {
            node: binom_tree_graph.nodes[node]["label"]
            for node in binom_tree_graph.nodes()
        }
        plt.figure(figsize=(10, 10))
        nx.draw(
            binom_tree_graph,
            position,
            with_labels=True,
            labels=labels,
            node_size=600,
            node_color="lightblue",
            font_size=8,
            font_color="black",
        )
        edge_labels = nx.get_edge_attributes(binom_tree_graph, "label")
        nx.draw_networkx_edge_labels(
            binom_tree_graph, position, edge_labels=edge_labels, font_size=8
        )
        plt.title("Cox-Ross-Rubinstein tree for option pricing -Amr Case-")
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
        payoff_option_tree = []
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
        payoff_option_tree.append(option_price_tree)
        for step in range(self.steps_number - 1, -1, -1):
            underlying_price_tree = (
                float(self.underlying_price_df.iloc[-1])
                * down_percentage ** pd.Series(list(np.arange(step, -1 / 2, -1 / 2)))
                * up_percentage ** pd.Series(list(np.arange(0, step + 1 / 2, 1 / 2)))
            )
            option_price_tree[: 2 * step + 1] = (
                discount
                * (
                    upper_probability * pd.Series(option_price_tree[0 : 2 * step + 1])
                    + (
                        down_probability
                        * pd.Series(option_price_tree[2 : 2 * step + 3])
                    )
                    + (
                        no_change_probability
                        * pd.Series(option_price_tree[1 : 2 * step + 2])
                    )
                )
            ).tolist()
            option_price_tree = option_price_tree[: 2 * step + 1]
            if self.option_type == "Call":
                euro_option_price_tree.append(option_price_tree)
                payoff_option_tree.append(
                    (underlying_price_tree - self.strike).tolist()
                )
                option_price_tree = pd.Series(
                    np.maximum(
                        (underlying_price_tree - self.strike).tolist(),
                        option_price_tree,
                    )
                ).tolist()
            else:
                euro_option_price_tree.append(option_price_tree)
                payoff_option_tree.append(
                    (self.strike - underlying_price_tree).tolist()
                )
                option_price_tree = pd.Series(
                    np.maximum(
                        (self.strike - underlying_price_tree).tolist(),
                        option_price_tree,
                    )
                ).tolist()
        trinom_tree_graph = nx.DiGraph()
        for step in range(self.steps_number, -self.steps_number - 1, -1):
            for mid_step in range(step, -step - 1, -1):
                k = np.arange(0, 2 * step + 1, 1)
                if (
                    euro_option_price_tree[self.steps_number - step][
                        int(k[step + mid_step])
                    ]
                    < payoff_option_tree[self.steps_number - step][
                        int(k[step + mid_step])
                    ]
                ):
                    state = "exercé"
                else:
                    state = ""
                trinom_tree_graph.add_node(
                    (step, mid_step),
                    label=f"{round(euro_option_price_tree[self.steps_number - step][int(k[step + mid_step])], 3)}\n"
                    f"{round(payoff_option_tree[self.steps_number - step][int(k[step + mid_step])], 3)}\n\n{state}",
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
            node_size=600,
            node_color="lightblue",
            font_size=8,
            font_color="black",
        )
        edge_labels = nx.get_edge_attributes(trinom_tree_graph, "label")
        nx.draw_networkx_edge_labels(
            trinom_tree_graph, position, edge_labels=edge_labels, font_size=8
        )
        plt.title("Cox-Ross-Rubinstein tree for option pricing -Amr Case-")
        plt.axis("off")
        plt.show()
        if f:
            plt.savefig(f, format="png")

    def compute_option_contract(self):
        super().compute_option_contract()
        self.crr = self.cox_ross_rubinstein()
        self.trinom = self.trinomial_model()



p = AmericanOption(180, "GOOG", 5, "Put", "2024-09-14", 0.07, 0, 0.3262, 2)
p.compute_option_contract()
print(p.crr)
print(p.trinom)
p.plot_trinomial_tree(None)
p.plot_binomial_tree(None)
