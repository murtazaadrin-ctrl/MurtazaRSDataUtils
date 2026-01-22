"""
MurtazaRSDataUtils - A Python utility library for processing and visualizing satellite imagery data.

This package provides tools for working with:
- Sentinel-2 L2A multispectral imagery
- MODIS Land Surface Temperature (LST) data
- Video creation from satellite image time series
"""

# Import from submodules for convenient access
from . import ImageUtils
from . import MODIS
from . import SentinelHub

# Import commonly used functions at package level
from .ImageUtils import (
    get_data_config,
    video_maker,
)

from .MODIS import (
    get_modis_day_lst_daily,
    get_modis_night_lst_daily,
    zip_to_video,
    plot_lst_timeseries,
)

from .SentinelHub import (
    create_config,
    get_sentinel2_catalog_results,
    fetch_sentinel2_images_from_results,
)

__version__ = '1.0.0'

__all__ = [
    # Submodules
    'ImageUtils',
    'MODIS',
    'SentinelHub',
    # ImageUtils functions
    'get_data_config',
    'video_maker',
    # MODIS functions
    'get_modis_day_lst_daily',
    'get_modis_night_lst_daily',
    'zip_to_video',
    'plot_lst_timeseries',
    # SentinelHub functions
    'create_config',
    'get_sentinel2_catalog_results',
    'fetch_sentinel2_images_from_results',
]

