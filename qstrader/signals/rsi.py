import pandas as pd
import talib

from qstrader.signals.signal import Signal


class RSISignal(Signal):
    """
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
        bumped_lookbacks = [lookback + 1 for lookback in lookbacks]
        super().__init__(start_dt, universe, bumped_lookbacks)

    @staticmethod
    def _asset_lookback_key(asset, lookback):
        """
        Create the buffer dictionary lookup key based
        on asset name and lookback period.

        Parameters
        ----------
        asset : `str`
            The asset symbol name.
        lookback : `int`
            The lookback period.

        Returns
        -------
        `str`
            The lookup key.
        """
        return '%s_%s' % (asset, lookback + 1)

    def __call__(self, asset, lookback):
        series = pd.Series(
            self.buffers.prices[RSISignal._asset_lookback_key(asset, lookback)]
        )
        series = series.dropna().to_numpy()
        if series.size > 0:
            rsi = talib.RSI(series, timeperiod=2)

            if len(rsi) < 1:
                return -1.0
            else:
                return rsi[-1]
        else:
            return -1.0


class CumulateRSISignal(Signal):
    """
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
        bumped_lookbacks = [lookback + 1 for lookback in lookbacks]
        super().__init__(start_dt, universe, bumped_lookbacks)

    @staticmethod
    def _asset_lookback_key(asset, lookback):
        """
        Create the buffer dictionary lookup key based
        on asset name and lookback period.

        Parameters
        ----------
        asset : `str`
            The asset symbol name.
        lookback : `int`
            The lookback period.

        Returns
        -------
        `str`
            The lookup key.
        """
        return '%s_%s' % (asset, lookback + 1)

    def __call__(self, asset, lookback):
        series = pd.Series(
            self.buffers.prices[RSISignal._asset_lookback_key(asset, lookback)]
        )
        series = series.dropna().to_numpy()
        if series.size > 0:
            rsi = talib.RSI(series, timeperiod=2)

            if len(rsi) < 3:
                return -1.0
            else:
                out = rsi[-1] + rsi[-2] + rsi[-3]
                return out
        else:
            return -1.0
