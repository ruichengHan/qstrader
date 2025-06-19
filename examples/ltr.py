import operator
import os

import pandas as pd
import pytz

from qstrader.alpha_model.alpha_model import AlphaModel
from qstrader.alpha_model.fixed_signals import FixedSignalsAlphaModel
from qstrader.asset.equity import Equity
from qstrader.asset.universe.dynamic import DynamicUniverse
from qstrader.asset.universe.static import StaticUniverse
from qstrader.signals.momentum import MomentumSignal
from qstrader.signals.vol import VolatilitySignal
from qstrader.signals.signals_collection import SignalsCollection
from qstrader.data.backtest_data_handler import BacktestDataHandler
from qstrader.data.daily_bar_csv import CSVDailyBarDataSource
from qstrader.statistics.tearsheet import TearsheetStatistics
from qstrader.trading.backtest import BacktestTradingSession
from qstrader.broker.fee_model.percent_fee_model import PercentFeeModel
import akshare as ak


class TopNMomentumAlphaModel(AlphaModel):

    def __init__(self, signals, mom_lookback, mom_top_n, universe, data_handler):
        """
        Initialise the TopNMomentumAlphaModel

        Parameters
        ----------
        signals : `SignalsCollection`
            The entity for interfacing with various pre-calculated
            signals. In this instance we want to use 'momentum'.
        mom_lookback : `integer`
            The number of business days to calculate momentum
            lookback over.
        mom_top_n : `integer`
            The number of assets to include in the portfolio,
            ranking from highest momentum descending.
        universe : `Universe`
            The collection of assets utilised for signal generation.
        data_handler : `DataHandler`
            The interface to the CSV data.

        Returns
        -------
        None
        """
        self.signals = signals
        self.mom_lookback = mom_lookback
        self.mom_top_n = mom_top_n
        self.universe = universe
        self.data_handler = data_handler
        self.model_result = self.load_model_result()
    
    def load_model_result(self):
        output = {}
        for line in open("../train/top_predictions.csv"):
            date, code, prediction = line.strip().split(",")
            print(date, code, prediction)
            code = "EQ:%06d" % int(code)
            if date not in output:
                output[date] = []
            output[date].append(code)
        return output

    def _generate_signals(
            self, dt, weights
    ):

        put_factor = 0
        top_assets = self.model_result.get(dt.strftime("%Y-%m-%d"), [])
        top_assets = list(filter(lambda x: x in weights, top_assets))
        if len(top_assets) == 0:
            return weights
        for asset in top_assets:
            weights[asset] = (1.0 - put_factor) / len(top_assets)
        weights["EQ:FXP"] = put_factor
        return weights

    def __call__(
            self, dt
    ):
        """
        Calculates the signal weights for the top N
        momentum alpha model, assuming that there is
        sufficient data to begin calculating momentum
        on the desired assets.

        Parameters
        ----------
        dt : `pd.Timestamp`
            The datetime for which the signal weights
            should be calculated.

        Returns
        -------
        `dict{str: float}`
            The newly created signal weights dictionary.
        """
        assets = self.universe.get_assets(dt)
        weights = {asset: 0.0 for asset in assets}

        # Only generate weights if the current time exceeds the
        # momentum lookback period
        if self.signals.warmup >= self.mom_lookback:
            weights = self._generate_signals(dt, weights)
        return weights


def get_index_stock(index_code="000300", year='20218'):
    """
    获取一个指数的成分股
    Parameters
    ----------
    index_code
    year

    Returns
    -------

    """
    index_stock_cons_csindex_df = ak.index_stock_cons(symbol=index_code)
    filter_df = index_stock_cons_csindex_df[index_stock_cons_csindex_df["纳入日期"] < str(year)]
    out = []
    for code in filter_df["品种代码"].to_list():
        out.append(code)
    return out


def get_symbols():
    stock_list = get_index_stock()
    stock_list.append("FXP")
    out = []
    for root, dirs, files in os.walk(csv_dir):
        for file_name in files:
            out.append(file_name.split(".")[0])

    out = list(filter(lambda x: x in stock_list, out))
    out.append("sh000300")
    return out


csv_dir = '/Users/rui.chengcr/PycharmProjects/qstrader/qs_data/price/'

if __name__ == "__main__":
    # Duration of the backtest
    start_dt = pd.Timestamp('2020-01-04 14:30:00', tz=pytz.UTC)
    burn_in_dt = pd.Timestamp('2021-01-01 14:30:00', tz=pytz.UTC)
    end_dt = pd.Timestamp('2024-12-31 23:59:00', tz=pytz.UTC)

    # Model parameters
    mom_top_n = 20

    # Construct the symbols and assets necessary for the backtest
    # This utilises the SPDR US sector ETFs, all beginning with XL
    strategy_symbols = get_symbols()
    assets = ['EQ:%s' % symbol for symbol in strategy_symbols]

    asset_dates = {asset: start_dt for asset in assets}
    strategy_universe = DynamicUniverse(asset_dates)

    # To avoid loading all CSV files in the directory, set the
    # data source to load only those provided symbols

    strategy_data_source = CSVDailyBarDataSource(csv_dir, Equity, csv_symbols=strategy_symbols)
    strategy_data_handler = BacktestDataHandler(strategy_universe, data_sources=[strategy_data_source])


    signals = SignalsCollection({}, strategy_data_handler)

    # Generate the alpha model instance for the top-N momentum alpha model
    strategy_alpha_model = TopNMomentumAlphaModel(
        signals, -1, mom_top_n, strategy_universe, strategy_data_handler
    )

    # Construct the strategy backtest and run it
    strategy_backtest = BacktestTradingSession(
        start_dt,
        end_dt,
        strategy_universe,
        strategy_alpha_model,
        signals=signals,
        rebalance='weekly',
        rebalance_weekday="MON",
        long_only=True,
        cash_buffer_percentage=0.05,
        burn_in_dt=burn_in_dt,
        data_handler=strategy_data_handler,
        memo_path="strategy.csv",
        fee_model=PercentFeeModel(commission_pct=0.001, tax_pct=0)
    )
    strategy_backtest.run()

    # Construct benchmark assets (buy & hold SPY)
    benchmark_symbols = ['sh000300']
    benchmark_assets = ['EQ:sh000300']
    benchmark_universe = StaticUniverse(benchmark_assets)
    benchmark_data_source = CSVDailyBarDataSource(csv_dir, Equity, csv_symbols=benchmark_symbols)
    benchmark_data_handler = BacktestDataHandler(benchmark_universe, data_sources=[benchmark_data_source])

    # Construct a benchmark Alpha Model that provides
    # 100% static allocation to the SPY ETF, with no rebalance
    benchmark_alpha_model = FixedSignalsAlphaModel({'EQ:sh000300': 1.0})
    benchmark_backtest = BacktestTradingSession(
        burn_in_dt,
        end_dt,
        benchmark_universe,
        benchmark_alpha_model,
        rebalance='buy_and_hold',
        long_only=True,
        cash_buffer_percentage=0,
        data_handler=benchmark_data_handler,
        memo_path="benchmark.csv"
    )
    benchmark_backtest.run()

    # Performance Output
    tearsheet = TearsheetStatistics(
        strategy_equity=strategy_backtest.get_equity_curve(),
        benchmark_equity=benchmark_backtest.get_equity_curve(),
        title='LTR Test'
    )
    tearsheet.plot_results()
