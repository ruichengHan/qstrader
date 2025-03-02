from qstrader.execution.execution_algo.market_order import (
    MarketOrderExecutionAlgorithm
)
from qstrader.execution.execution_handler import (
    ExecutionHandler
)
from qstrader.portcon.one_pcm import (
    OnePortfolioConstructionModel
)
from qstrader.portcon.order_sizer.one_dollar_weighted import (
   OneDollarWeightedCashBufferedOrderSizer
)


class OneQuantTradingSystem(object):
    """
    Encapsulates all components associated with the quantitative
    trading system. This includes the alpha model(s), the risk
    model, the transaction cost model along with portfolio construction
    and execution mechanism.

    Parameters
    ----------
    universe : `Universe`
        The Asset Universe.
    broker : `Broker`
        The Broker to execute orders against.
    broker_portfolio_id : `str`
        The specific broker portfolio to send orders to.
    data_handler : `DataHandler`
        The data handler instance used for all market/fundamental data.
    alpha_model : `AlphaModel`
        The alpha model used within the portfolio construction.
    risk_model : `AlphaModel`, optional
        An optional risk model used within the portfolio construction.
    long_only : `Boolean`, optional
        Whether to invoke the long only order sizer or allow
        long/short leveraged portfolios. Defaults to long/short leveraged.
    submit_orders : `Boolean`, optional
        Whether to actually submit generated orders. Defaults to no submission.
    """

    def __init__(
            self,
            universe,
            broker,
            broker_portfolio_id,
            data_handler,
            buy_model,
            sell_model,
            submit_orders=False,
            **kwargs
    ):
        self.universe = universe
        self.broker = broker
        self.broker_portfolio_id = broker_portfolio_id
        self.data_handler = data_handler
        self.buy_model = buy_model
        self.sell_model = sell_model
        self.submit_orders = submit_orders
        self._initialise_models(**kwargs)

    def _create_order_sizer(self, **kwargs):
        order_sizer = OneDollarWeightedCashBufferedOrderSizer(
            self.broker,
            self.broker_portfolio_id,
            self.data_handler,
            cash_buffer_percentage=0
        )
        return order_sizer

    def _initialise_models(self, **kwargs):
        # Generate the portfolio construction
        self.portfolio_construction_model = OnePortfolioConstructionModel(
            self.broker,
            self.broker_portfolio_id,
            self.universe,
            order_sizer=self._create_order_sizer(**kwargs),
            buy_model=self.buy_model,
            sell_model=self.sell_model,
            data_handler=self.data_handler
        )

        # Execution
        execution_algo = MarketOrderExecutionAlgorithm()
        self.execution_handler = ExecutionHandler(
            self.broker,
            self.broker_portfolio_id,
            self.universe,
            submit_orders=self.submit_orders,
            execution_algo=execution_algo,
            data_handler=self.data_handler
        )

    def __call__(self, dt, stats=None):
        """
        Construct the portfolio and (optionally) execute the orders
        with the broker.

        Parameters
        ----------
        dt : `pd.Timestamp`
            The current time.
        stats : `dict`, optional
            An optional statistics dictionary to append values to
            throughout the simulation lifetime.

        Returns
        -------
        `None`
        """
        # Construct the target portfolio
        orders = self.portfolio_construction_model(dt, stats=stats)

        # Execute the orders
        self.execution_handler(dt, orders)
