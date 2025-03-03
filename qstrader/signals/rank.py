import numpy as np

from qstrader.signals.signal import Signal


class RankSignal(Signal):
    """
    Indicator class to calculate simple moving average
    of last N periods for a set of prices.

    Parameters
    ----------
    start_dt : `pd.Timestamp`
        The starting datetime (UTC) of the signal.
    universe : `Universe`
        The universe of assets to calculate the signals for.
    lookbacks : `list[int]`
        The number of lookback periods to store prices for.
    """

    def __init__(self, start_dt, universe, lookbacks):
        super().__init__(start_dt, universe, lookbacks)

    def __call__(self, asset, lookback):
        prices = self.buffers.prices['%s_%s' % (asset, lookback)]
        today = prices[-1]
        lower_cnt = 0
        for index in range(0, len(prices) - 1):
            if prices[index] < today:
                lower_cnt += 1
        return lower_cnt / (len(prices) - 1)

