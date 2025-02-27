import os

import pandas as pd
import pytz

from qstrader.alpha_model.alpha_model import AlphaModel
from qstrader.alpha_model.fixed_signals import FixedSignalsAlphaModel
from qstrader.asset.equity import Equity
from qstrader.asset.universe.dynamic import DynamicUniverse
from qstrader.asset.universe.static import StaticUniverse
from qstrader.data.backtest_data_handler import BacktestDataHandler
from qstrader.data.daily_bar_csv import CSVDailyBarDataSource
from qstrader.signals.rsi import RSISignal, CumulateRSISignal
from qstrader.signals.signals_collection import SignalsCollection
from qstrader.signals.sma import SMASignal
from qstrader.statistics.tearsheet import TearsheetStatistics
from qstrader.trading.backtest import BacktestTradingSession


class CustomModel(AlphaModel):
    def __init__(
            self, signals, mom_lookback, mom_top_n, universe, data_handler
    ):
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

    def calculate_weight(self, dt):
        signal_list = [self.cal_signals(asset) for asset in assets]
        filter_asset_list = list(filter(lambda x: x["sma"] is not None and x["sma"] > 0 and x["rsi"] < 2, signal_list))
        out = list(map(lambda x: x["asset"], sorted(filter_asset_list, key=lambda x: x["rsi"])))
        if len(out) > self.mom_top_n:
            return out[:self.mom_top_n]
        else:
            return out

    def cal_signals(self, asset):
        j = {
            "asset": asset,
            "rsi": self.signals["rsi"](asset, self.mom_lookback),
            "sma": self.signals["sma"].diff(asset, self.mom_lookback)
        }
        return j

    def _generate_signals(self, dt, weights):

        top_assets = self.calculate_weight(dt)
        for asset in top_assets:
            weights[asset] = 1.0 / self.mom_top_n
        return weights

    def __call__(self, dt):
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


csv_dir = '/Users/rui.chengcr/PycharmProjects/qstrader/qs_data/price/'


def get_symbols():
    out = []
    for root, dirs, files in os.walk(csv_dir):
        for file_name in files:
            out.append(file_name.split(".")[0])
    return out


if __name__ == "__main__":
    # Duration of the backtest
    start_dt = pd.Timestamp('2016-01-04 14:30:00', tz=pytz.UTC)
    burn_in_dt = pd.Timestamp('2016-06-01 14:30:00', tz=pytz.UTC)
    end_dt = pd.Timestamp('2024-12-31 23:59:00', tz=pytz.UTC)

    # Model parameters
    mom_lookback = 126  # Six months worth of business days
    mom_top_n = 20  # Number of assets to include at any one time

    # Construct the symbols and assets necessary for the backtest
    # This utilises the SPDR US sector ETFs, all beginning with XL
    strategy_symbols = get_symbols()
    assets = ['EQ:%s' % symbol for symbol in strategy_symbols]

    # As this is a dynamic universe of assets (XLC is added later)
    # we need to tell QSTrader when XLC can be included. This is
    # achieved using an asset dates dictionary
    asset_dates = {asset: start_dt for asset in assets}
    strategy_universe = DynamicUniverse(asset_dates)

    # To avoid loading all CSV files in the directory, set the
    # data source to load only those provided symbols
    strategy_data_source = CSVDailyBarDataSource(csv_dir, Equity, csv_symbols=strategy_symbols)
    strategy_data_handler = BacktestDataHandler(strategy_universe, data_sources=[strategy_data_source])

    # Generate the signals (in this case holding-period return based
    # momentum) used in the top-N momentum alpha model
    rsi = RSISignal(start_dt, strategy_universe, lookbacks=[mom_lookback])
    cumulate_rsi = CumulateRSISignal(start_dt, strategy_universe, lookbacks=[mom_lookback])
    sma = SMASignal(start_dt, strategy_universe, lookbacks=[mom_lookback])
    signals = SignalsCollection({"rsi": rsi, "sma": sma, "cu_rsi": cumulate_rsi}, strategy_data_handler)

    # Generate the alpha model instance for the top-N momentum alpha model
    strategy_alpha_model = CustomModel(signals, mom_lookback, mom_top_n, strategy_universe, strategy_data_handler)

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
        cash_buffer_percentage=0.01,
        burn_in_dt=burn_in_dt,
        data_handler=strategy_data_handler,
        memo_path="strategy.csv"
    )
    strategy_backtest.run(True)

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
        cash_buffer_percentage=0.01,
        data_handler=benchmark_data_handler,
        memo_path="benchmark.csv"
    )
    benchmark_backtest.run()

    # Performance Output
    tearsheet = TearsheetStatistics(
        strategy_equity=strategy_backtest.get_equity_curve(),
        benchmark_equity=benchmark_backtest.get_equity_curve(),
        title='US Sector Momentum - Top 3 Sectors'
    )
    tearsheet.plot_results()
