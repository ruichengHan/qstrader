import pandas as pd
import talib

from qstrader.signals.signal import Signal


class DirectionVolumeSignal(Signal):
    """
    带方向的成交量
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
        infos = self.info_buffers.infos[DirectionVolumeSignal._asset_lookback_key(asset, lookback)]
        if infos and len(infos) >= 2:
            today = infos[-1]
            yesterday = infos[-2]

            signed = 1 if today["close"] > yesterday["close"] else -1 if today["close"] < yesterday["close"] else 0
            return signed * today["volume"]
        else:
            return None
