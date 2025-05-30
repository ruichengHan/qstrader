import pandas as pd

from qstrader.signals.signal import Signal


class OopBreakSignal(Signal):
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
        infos = self.info_buffers.infos[OopBreakSignal._asset_lookback_key(asset, lookback)]
        if infos and len(infos) >= 2:
            today = infos[-1]
            yesterday = infos[-2]
            if today["open"] < yesterday["low"]:
                return 1
        return 0
