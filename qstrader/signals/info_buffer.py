from collections import deque


class AssetInfoBuffers(object):
    """
    Utility class to store double-ended queue ("deque")
    based price buffers for usage in lookback-based
    indicator calculations.

    Parameters
    ----------
    assets : `list[str]`
        The list of assets to create price buffers for.
    lookbacks : `list[int]`, optional
        The number of lookback periods to store prices for.
    """

    def __init__(self, assets, lookbacks=[12]):
        self.assets = assets
        self.lookbacks = lookbacks
        self.infos = self._create_all_assets_infos_buffer_dict()

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
        return '%s_%s' % (asset, lookback)

    def _create_single_asset_infos_buffer_dict(self, asset):
        """
        Creates a dictionary of asset-lookback pair
        price buffers for a single asset.

        Returns
        -------
        `dict{str: deque[float]}`
            The price buffer dictionary.
        """
        return {
            AssetInfoBuffers._asset_lookback_key(
                asset, lookback
            ): deque(maxlen=lookback)
            for lookback in self.lookbacks
        }

    def _create_all_assets_infos_buffer_dict(self):
        """
        Creates a dictionary of asset-lookback pair
        price buffers for all assets.

        Returns
        -------
        `dict{str: deque[float]}`
            The price buffer dictionary.
        """
        prices = {}
        for asset in self.assets:
            prices.update(self._create_single_asset_infos_buffer_dict(asset))
        return prices

    def add_asset(self, asset):
        """
        Add an asset to the list of current assets. This is necessary if
        the asset is part of a DynamicUniverse and isn't present at
        the beginning of a backtest.

        Parameters
        ----------
        asset : `str`
            The asset symbol name.
        """
        if asset in self.assets:
            raise ValueError(
                'Unable to add asset "%s" since it already '
                'exists in this price buffer.' % asset
            )
        else:
            self.infos.update(self._create_single_asset_infos_buffer_dict(asset))

    def append(self, asset, info):
        """
        Append a new price onto the price deque for
        the specific asset provided.

        Parameters
        ----------
        asset : `str`
            The asset symbol name.
        info : `dict`
            The dict infor of the asset.
        """

        # The asset may have been added to the universe subsequent
        # to the beginning of the backtest and as such needs a
        # newly created pricing buffer
        asset_lookback_key = AssetInfoBuffers._asset_lookback_key(asset, self.lookbacks[0])
        if asset_lookback_key not in self.infos:
            self.infos.update(self._create_single_asset_infos_buffer_dict(asset))

        for lookback in self.lookbacks:
            self.infos[
                AssetInfoBuffers._asset_lookback_key(
                    asset, lookback
                )
            ].append(info)
