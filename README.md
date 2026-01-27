# MurtazaRSDataUtils

A Python utility library for processing and visualizing satellite imagery data, with support for Sentinel-2 L2A multispectral imagery and MODIS Land Surface Temperature (LST) data. This repository provides tools for fetching, processing, and creating time-lapse videos from satellite imagery.

## Overview

MurtazaRSDataUtils is designed to facilitate working with satellite imagery data from multiple sources:
- **Sentinel-2 L2A**: High-resolution multispectral imagery from the Copernicus Sentinel-2 mission
- **MODIS LST**: Land Surface Temperature data from NASA's MODIS sensors (Terra and Aqua)

The library provides end-to-end functionality from data acquisition to video generation, making it easy to create time-lapse visualizations of satellite imagery.

## Project Structure

```
MurtazaRSDataUtils/
├── ImageUtils/
│   ├── data_config.py      # Configuration utilities for bounding boxes and data requests
│   └── video_maker.py       # Video creation from image stacks
├── MODIS/
│   ├── get_collection.py   # MODIS collection retrieval from Google Earth Engine
│   └── get_data.py         # MODIS data processing, caching, and visualization
└── SentinelHub/
    ├── create_config.py    # Sentinel Hub API configuration
    └── L2A_data_fetch.py   # Sentinel-2 L2A data fetching and processing
```

## Features

### ImageUtils
- **Bounding Box Generation**: Create bounding boxes from coordinates and distances
- **Coordinate Transformations**: Convert between geographic coordinates and pixel coordinates
- **Video Creation**: Generate time-lapse videos from satellite image stacks
- **Band Combinations**: Support for NCC (Natural Color Composite), FCC (False Color Composite), and SWIR (Shortwave Infrared) visualizations

### MODIS
- **Collection Management**: Retrieve MODIS LST collections from Google Earth Engine
- **Data Download**: Download and cache MODIS data as ZIP files containing GeoTIFFs
- **Time Series Analysis**: Plot LST time series for specific locations or global statistics
- **Video Generation**: Create videos from MODIS LST data with customizable colormaps

### SentinelHub
- **Catalog Search**: Search Sentinel-2 L2A catalog with cloud cover filtering
- **Image Fetching**: Retrieve Sentinel-2 images with 10 spectral bands
- **Cloud Masking**: Process images with cloud mask support
- **Time Interval Management**: Generate time intervals for data acquisition

## Installation

### Prerequisites

- Python 3.7+
- Google Earth Engine account (for MODIS data)
- Sentinel Hub account with API credentials (for Sentinel-2 data)

### Required Dependencies

```bash
pip install sentinelhub
pip install earthengine-api
pip install numpy
pip install matplotlib
pip install opencv-python
pip install rasterio
pip install shapely
pip install pandas
pip install requests
```

### Authentication Setup

#### Google Earth Engine
```python
import ee
ee.Authenticate()
ee.Initialize()
```

#### Sentinel Hub
Create a configuration using your client ID and secret:
```python
from MurtazaRSDataUtils import create_config
config = create_config(client_id="your_client_id", client_secret="your_client_secret")
```

## Import Options

The package provides multiple ways to import functions for convenience:

### Option 1: Import from root package (Recommended)
```python
from MurtazaRSDataUtils import get_data_config, video_maker, create_config
from MurtazaRSDataUtils import get_sentinel2_catalog_results, fetch_sentinel2_images_from_results
```

### Option 2: Import from submodules
```python
from MurtazaRSDataUtils.ImageUtils import get_data_config, video_maker
from MurtazaRSDataUtils.MODIS import get_modis_day_lst_daily, zip_to_video
from MurtazaRSDataUtils.SentinelHub import create_config, get_sentinel2_catalog_results
```

### Option 3: Import entire submodules
```python
from MurtazaRSDataUtils import ImageUtils, MODIS, SentinelHub

config = ImageUtils.get_data_config(...)
collection = MODIS.get_modis_day_lst_daily(...)
sh_config = SentinelHub.create_config(...)
```

## Usage Examples

### 1. Creating a Data Configuration

```python
from MurtazaRSDataUtils import get_data_config

# Define coordinates (latitude, longitude) and date range
coords = (28.6139, 77.2090)  # Example: New Delhi
date_range = ("2023-01-01", "2023-12-31")
distances = (40, 40)  # 40km x 40km area
resolution = 42  # meters per pixel
max_cloud_cover = 20  # Maximum cloud cover percentage (for Sentinel-2 only)

data_config = get_data_config(coords, date_range, distances, resolution, max_cloud_cover=max_cloud_cover)
```

### 2. Fetching Sentinel-2 Images

```python
from MurtazaRSDataUtils import (
    create_config,
    get_sentinel2_catalog_results,
    fetch_sentinel2_images_from_results
)

# Create Sentinel Hub configuration
config = create_config(client_id="your_id", client_secret="your_secret")

# Search catalog (max_cloud_cover is now part of data_config)
results = get_sentinel2_catalog_results(
    sh_config=config,
    data_config=data_config,
    min_time_diff_seconds=3600
)

# Fetch images
images, dates, cloud_masks = fetch_sentinel2_images_from_results(
    sh_config=config,
    data_config=data_config,
    results=results,
    cloud_masking=True
)
```

### 3. Creating a Video from Sentinel-2 Images

```python
from MurtazaRSDataUtils import video_maker

video_maker(
    sentinel2_images=images,
    dates=dates,
    output_path="output_video.mp4",
    band_combo="NCC",  # or "FCC", "SWIR", or custom band indices
    fps=0.5,
    stretch_percentiles=(1, 99),
    blur_factor=1
)
```

### 4. Working with MODIS LST Data

```python
from MurtazaRSDataUtils.MODIS import get_modis_day_lst_daily, save_collection_as_zip, zip_to_video

# Get MODIS collection
collection = get_modis_day_lst_daily(data_config)

# Download as ZIP
roi = ee.Geometry.Rectangle([...])  # Define ROI
zip_path = save_collection_as_zip(collection, roi, scale=1000, zip_name="modis_lst.zip")

# Create video
zip_to_video(
    zip_path="modis_lst.zip",
    out_mp4="lst_video.mp4",
    fps=2,
    cmap='RdBu_r',
    pmin=1,
    pmax=99
)
```

### 5. Plotting MODIS Time Series

```python
from MurtazaRSDataUtils.MODIS import plot_lst_timeseries, plot_global_stats

# Plot global statistics
plot_global_stats("modis_lst.zip", stats=('mean', 'min', 'max', 'median'))

# Plot time series for specific locations
plot_lst_timeseries([
    ("modis_lst.zip", (28.6139, 77.2090), "Location 1"),
    ("modis_lst.zip", "mean", "Global Mean")
])
```

## Module Documentation

### ImageUtils.data_config
- `get_data_config()`: Generate configuration dictionary for satellite data requests
- `get_polygon_coords()`: Calculate polygon coordinates from center point and distances
- `bbox_to_dimensions_alternative()`: Calculate image dimensions from bounding box and resolution
- `haversine()`: Calculate great-circle distance between two geographic points

### ImageUtils.video_maker
- `video_maker()`: Create video from Sentinel-2 image stack with band combinations
- `create_video_from_frames()`: Convert numpy array frames to video file
- `numpy_array_to_image()`: Convert numpy array to image file with text overlay

### MODIS.get_collection
- `get_modis_collection()`: Generic function to retrieve MODIS collections
- `get_modis_day_lst_daily()`: Get Terra MODIS daily daytime LST
- `get_modis_night_lst_daily()`: Get Terra MODIS daily nighttime LST
- `get_modis_day_lst_8day()`: Get Terra MODIS 8-day composite daytime LST
- `get_aqua_modis_day_lst_daily()`: Get Aqua MODIS daily daytime LST
- Similar functions for Aqua MODIS nighttime and 8-day composites

### MODIS.get_data
- `save_collection_as_zip()`: Download MODIS collection and save as ZIP of GeoTIFFs
- `load_zip_stack()`: Load GeoTIFF stack from ZIP with caching
- `zip_to_video()`: Create video from MODIS LST ZIP file
- `plot_lst_timeseries()`: Plot LST time series for locations or statistics
- `plot_global_stats()`: Plot global statistics over time
- `latlon_to_pixel()`: Convert geographic coordinates to pixel coordinates

### SentinelHub.create_config
- `create_config()`: Create Sentinel Hub configuration from credentials

### SentinelHub.L2A_data_fetch
- `get_sentinel_image_10_bands()`: Fetch Sentinel-2 image with 10 spectral bands
- `get_sentinel2_catalog_results()`: Search Sentinel-2 catalog with filtering
- `fetch_sentinel2_images_from_results()`: Fetch multiple images from catalog results
- `generate_time_intervals()`: Generate time intervals for data acquisition

## Band Combinations

### Sentinel-2 Band Combinations
- **NCC (Natural Color Composite)**: Bands [2, 1, 0] - Blue, Green, Red
- **FCC (False Color Composite)**: Bands [6, 2, 1] - Red Edge, Green, Blue
- **SWIR (Shortwave Infrared)**: Bands [9, 8, 7] - SWIR2, SWIR1, NIR2

## Notes

- MODIS data requires Google Earth Engine authentication
- Sentinel-2 data requires Sentinel Hub API credentials
- Large datasets may require significant processing time and storage
- Video generation creates temporary files that are automatically cleaned up
- MODIS ZIP files are cached in memory for efficient repeated access

## License

[Specify your license here]

## Contributing

[Add contribution guidelines if applicable]

## Contact

[Add contact information if applicable]

