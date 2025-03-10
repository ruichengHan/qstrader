from qstrader.execution.order import Order


class OnePortfolioConstructionModel(object):
    def __init__(
            self,
            broker,
            broker_portfolio_id,
            universe,
            order_sizer,
            buy_model=None,
            sell_model=None,
            data_handler=None,
    ):
        self.broker = broker
        self.broker_portfolio_id = broker_portfolio_id
        self.universe = universe
        self.order_sizer = order_sizer
        self.buy_model = buy_model
        self.sell_model = sell_model
        self.data_handler = data_handler

    def __call__(self, dt, stats=None):
        """
        Execute the portfolio construction process at a particular
        provided date-time.

        Use the optional alpha model, risk model and cost model instances
        to create a list of desired weights that are then sent to the
        target weight generator instance to be optimised.

        Parameters
        ----------
        dt : `pd.Timestamp`
            The date-time used to for Asset list determination and
            weight generation.
        stats : `dict`, optional
            An optional statistics dictionary to append values to
            throughout the simulation lifetime.

        Returns
        -------
        `list[Order]`
            The list of rebalancing orders to be sent to Execution.
        """

        # Obtain current Broker account portfolio
        current_portfolio = self._obtain_current_portfolio()

        # buy mode
        buy_weights = self.buy_model(dt)
        buy_portfolio = self._generate_target_portfolio(dt, buy_weights)
        # for asset in current_portfolio:
        #     buy_portfolio.pop(asset)
        sell_weights = self.sell_model(dt)
        sell_portfolio = {}
        for asset in current_portfolio.keys():
            if sell_weights[asset] > 0.1:
                sell_portfolio[asset] = current_portfolio[asset]

        orders = self._generate_rebalance_orders(dt, buy_portfolio, sell_portfolio)

        return orders

    def _obtain_current_portfolio(self):
        """
        Query the broker for the current account asset quantities and
        return as a portfolio dictionary.

        Returns
        -------
        `dict{str: dict}`
            Current broker account asset quantities in integral units.
        """
        return self.broker.get_portfolio_as_dict(self.broker_portfolio_id)

    def _generate_rebalance_orders(self, dt, buy_portfolio, sell_portfolio):
        out = []
        for asset in buy_portfolio.keys():
            target_qty = buy_portfolio[asset]["quantity"]
            if target_qty > 0:
                out.append(Order(dt, asset, target_qty))
        for asset in sell_portfolio.keys():
            target_qty = sell_portfolio[asset]["quantity"]
            out.append(Order(dt, asset, -target_qty))
        return out

    def _generate_target_portfolio(self, dt, weights):
        """
        Generate the number of units (shares/lots) per Asset based on the
        target weight vector.

        Parameters
        ----------
        dt : `pd.Timestamp`
            The current timestamp.
        weights : `dict{str: float}`
            The union of the zero-weights and optimised weights, where the
            optimised weights take precedence.

        Returns
        -------
        `dict{str: dict}`
            Target asset quantities in integral units.
        """
        return self.order_sizer(dt, weights)
