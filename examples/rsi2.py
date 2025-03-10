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
from qstrader.statistics.tearsheet import TearsheetStatistics
from qstrader.trading.one_backtest import OneBacktestTradingSession
from qstrader.trading.backtest import BacktestTradingSession


class IndexBuyModel(AlphaModel):
    def calculate_weight(self, assets):
        signal_list = [self.cal_signals(asset) for asset in assets]
        filter_asset_list = list(filter(lambda x: 0 < x["rsi"] < 5 or 0 < x["cu_rsi"] < 65, signal_list))
        out = list(map(lambda x: x["asset"], sorted(filter_asset_list, key=lambda x: x["rsi"])))
        return out

    def cal_signals(self, asset):
        j = {
            "asset": asset,
            "rsi": self.signals["rsi"](asset, self.mom_lookback),
            "cu_rsi": self.signals["cu_rsi"](asset, self.mom_lookback)
        }
        return j

    def __init__(self, signals, mom_lookback, universe, data_handler):
        self.signals = signals
        self.mom_lookback = mom_lookback
        self.universe = universe
        self.data_handler = data_handler

    def _generate_signals(self, assets, weights):
        top_assets = self.calculate_weight(assets)
        for asset in top_assets:
            weights[asset] = 1
        return weights

    def __call__(self, dt):
        assets = self.universe.get_assets(dt)
        weights = {asset: 0.0 for asset in assets}

        # Only generate weights if the current time exceeds the
        # momentum lookback period
        if self.signals.warmup >= self.mom_lookback:
            weights = self._generate_signals(assets, weights)
        return weights


class IndexSellModel(AlphaModel):
    def calculate_weight(self, assets):
        signal_list = [self.cal_signals(asset) for asset in assets]
        filter_asset_list = list(filter(lambda x: x["rsi"] >= 75, signal_list))
        out = list(map(lambda x: x["asset"], sorted(filter_asset_list, key=lambda x: x["rsi"])))
        return out

    def cal_signals(self, asset):
        j = {
            "asset": asset,
            "rsi": self.signals["rsi"](asset, self.mom_lookback)
        }
        return j

    def __init__(self, signals, mom_lookback, universe, data_handler):
        self.signals = signals
        self.mom_lookback = mom_lookback
        self.universe = universe
        self.data_handler = data_handler

    def _generate_signals(self, assets, weights):
        top_assets = self.calculate_weight(assets)
        for asset in top_assets:
            weights[asset] = 1
        return weights

    def __call__(self, dt):
        assets = self.universe.get_assets(dt)
        weights = {asset: 0.0 for asset in assets}

        # Only generate weights if the current time exceeds the
        # momentum lookback period
        if self.signals.warmup >= self.mom_lookback:
            weights = self._generate_signals(assets, weights)
        return weights


csv_dir = '/Users/rui.chengcr/PycharmProjects/qstrader/qs_data/price/'


def get_symbols():
    return ["sh000300"]


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
    total_assets = ['EQ:%s' % symbol for symbol in strategy_symbols]

    # As this is a dynamic universe of assets (XLC is added later)
    # we need to tell QSTrader when XLC can be included. This is
    # achieved using an asset dates dictionary
    asset_dates = {asset: start_dt for asset in total_assets}
    strategy_universe = DynamicUniverse(asset_dates)

    # To avoid loading all CSV files in the directory, set the
    # data source to load only those provided symbols
    strategy_data_source = CSVDailyBarDataSource(csv_dir, Equity, csv_symbols=strategy_symbols)
    strategy_data_handler = BacktestDataHandler(strategy_universe, data_sources=[strategy_data_source])

    # Generate the signals (in this case holding-period return based
    # momentum) used in the top-N momentum alpha model
    rsi = RSISignal(start_dt, strategy_universe, lookbacks=[mom_lookback])
    cumulate_rsi = CumulateRSISignal(start_dt, strategy_universe, lookbacks=[mom_lookback])
    signals = SignalsCollection({"rsi": rsi, "cu_rsi": cumulate_rsi}, strategy_data_handler)

    # Generate the alpha model instance for the top-N momentum alpha model
    buy_model = IndexBuyModel(signals, mom_lookback, strategy_universe, strategy_data_handler)
    sell_model = IndexSellModel(signals, mom_lookback, strategy_universe, strategy_data_handler)

    # Construct the strategy backtest and run it
    strategy_backtest = OneBacktestTradingSession(
        start_dt,
        end_dt,
        strategy_universe,
        buy_model=buy_model,
        sell_model=sell_model,
        signals=signals,
        rebalance='daily',
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
        memo_path="benchmark.csv")
    benchmark_backtest.run()

    # Performance Output
    tearsheet = TearsheetStatistics(
        strategy_equity=strategy_backtest.get_equity_curve(),
        benchmark_equity=benchmark_backtest.get_equity_curve(),
        title='US Sector Momentum - Top 3 Sectors'
    )
    tearsheet.plot_results()
