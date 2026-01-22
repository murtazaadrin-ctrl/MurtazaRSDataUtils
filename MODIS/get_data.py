"""
MODIS data processing utilities for downloading, caching, and visualizing LST data.
"""

import os
import zipfile
from io import BytesIO

import cv2
import ee
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
import requests
from rasterio.warp import transform as crs_transform

# Global in-memory cache for loaded ZIP stacks
ZIP_CACHE = {}


# ============================================================================
# Earth Engine Collection Utilities
# ============================================================================

def save_collection_as_zip(collection, roi, scale, zip_name):
    """
    Download MODIS ImageCollection and save as a ZIP file of GeoTIFFs.

    Parameters
    ----------
    collection : ee.ImageCollection
        Earth Engine ImageCollection to download
    roi : ee.Geometry
        Region of interest for clipping
    scale : float
        Pixel scale in meters
    zip_name : str
        Output ZIP file path

    Returns
    -------
    str
        Path to the created ZIP file
    """
    temp_dir = "temp_tiffs"
    os.makedirs(temp_dir, exist_ok=True)

    size = collection.size().getInfo()
    tiff_paths = []

    for i in range(size):
        img = ee.Image(collection.toList(size).get(i))
        date = ee.Date(img.get('system:time_start')).format('YYYY-MM-dd').getInfo()

        url = img.getDownloadURL({
            'region': roi,
            'scale': scale,
            'crs': 'EPSG:4326',
            'format': 'GEO_TIFF'
        })

        path = os.path.join(temp_dir, f'lst_{date}.tif')
        with open(path, 'wb') as f:
            f.write(requests.get(url).content)

        tiff_paths.append(path)

    with zipfile.ZipFile(zip_name, 'w', zipfile.ZIP_DEFLATED) as z:
        for p in tiff_paths:
            z.write(p, os.path.basename(p))

    # Cleanup temporary files
    for p in tiff_paths:
        os.remove(p)
    os.rmdir(temp_dir)

    return zip_name


def get_closest_image(collection, target_date):
    """
    Get the image from collection closest to the target date.

    Parameters
    ----------
    collection : ee.ImageCollection
        Earth Engine ImageCollection
    target_date : str
        Target date string (e.g., '2023-01-15')

    Returns
    -------
    ee.Image
        Closest image to target date
    """
    target = ee.Date(target_date)

    def add_diff(img):
        diff = ee.Number(img.get('system:time_start')).subtract(target.millis()).abs()
        return img.set('diff', diff)

    return ee.Image(
        collection.map(add_diff).sort('diff').first()
    )


# ============================================================================
# ZIP Stack Loading and Caching
# ============================================================================

def load_zip_stack(zip_path):
    """
    Load GeoTIFF stack from ZIP file with caching.

    Parameters
    ----------
    zip_path : str
        Path to ZIP file containing GeoTIFFs

    Returns
    -------
    dict
        Dictionary containing:
        - 'stack': numpy array of shape (time, height, width)
        - 'dates': list of date strings
        - 'transform': rasterio Affine transform
        - 'crs': coordinate reference system
        - 'shape': tuple of (height, width)
    """
    if zip_path in ZIP_CACHE:
        return ZIP_CACHE[zip_path]

    arrays = []
    dates = []
    transform = None
    crs = None
    height = width = None

    with zipfile.ZipFile(zip_path) as z:
        for name in sorted(z.namelist()):
            date = name.replace('lst_', '').replace('.tif', '')

            with z.open(name) as f:
                with rasterio.open(BytesIO(f.read())) as src:
                    arr = src.read(1)
                    arr[arr == src.nodata] = np.nan

                    if transform is None:
                        transform = src.transform
                        crs = src.crs
                        height, width = src.height, src.width

            arrays.append(arr)
            dates.append(date)

    stack = np.stack(arrays)

    result = {
        "stack": stack,
        "dates": dates,
        "transform": transform,
        "crs": crs,
        "shape": (height, width)
    }

    ZIP_CACHE[zip_path] = result
    return result


# ============================================================================
# Coordinate Transformation
# ============================================================================

def latlon_to_pixel(lat, lon, transform, raster_crs):
    """
    Convert (lat, lon) coordinates in EPSG:4326 to pixel (row, col) coordinates.

    Parameters
    ----------
    lat : float
        Latitude in degrees
    lon : float
        Longitude in degrees
    transform : rasterio.Affine
        Affine transform from raster
    raster_crs : str or rasterio.crs.CRS
        Coordinate reference system of the raster

    Returns
    -------
    tuple
        (row, col) pixel coordinates
    """
    # Reproject point to raster CRS
    xs, ys = crs_transform(
        'EPSG:4326',
        raster_crs,
        [lon],
        [lat]
    )

    # World → pixel
    col, row = ~transform * (xs[0], ys[0])

    return int(row), int(col)


# ============================================================================
# Visualization Functions
# ============================================================================

def plot_global_stats(zip_path, stats=('mean', 'min', 'max', 'median')):
    """
    Plot global statistics (mean, min, max, median) over time from ZIP stack.

    Parameters
    ----------
    zip_path : str
        Path to ZIP file containing GeoTIFFs
    stats : tuple, default ('mean', 'min', 'max', 'median')
        Statistics to compute and plot
    """
    data = load_zip_stack(zip_path)
    stack = data["stack"]
    dates = data["dates"]

    stats_dict = {}
    if 'mean' in stats:
        stats_dict['mean'] = np.nanmean(stack, axis=(1, 2))
    if 'min' in stats:
        stats_dict['min'] = np.nanmin(stack, axis=(1, 2))
    if 'max' in stats:
        stats_dict['max'] = np.nanmax(stack, axis=(1, 2))
    if 'median' in stats:
        stats_dict['median'] = np.nanmedian(stack, axis=(1, 2))

    df = pd.DataFrame(stats_dict, index=pd.to_datetime(dates))
    df.plot(figsize=(10, 4), grid=True)
    plt.ylabel("LST (°C)")
    plt.title("Global LST Statistics")
    plt.show()


def plot_lst_timeseries(plots):
    """
    Plot LST time series for multiple locations or statistics.

    Parameters
    ----------
    plots : list of tuples
        Each tuple contains (zip_path, loc, label) where:
        - zip_path: Path to ZIP file
        - loc: Either a string ('mean', 'min', 'max', 'median') or (lat, lon) tuple
        - label: Label for the plot legend
    """
    plt.figure(figsize=(10, 4))

    for zip_path, loc, label in plots:
        data = load_zip_stack(zip_path)

        stack = data["stack"]
        dates = data["dates"]
        transform = data["transform"]
        crs = data["crs"]
        h, w = data["shape"]

        if isinstance(loc, str):
            if loc == 'mean':
                ts = np.nanmean(stack, axis=(1, 2))
            elif loc == 'min':
                ts = np.nanmin(stack, axis=(1, 2))
            elif loc == 'max':
                ts = np.nanmax(stack, axis=(1, 2))
            elif loc == 'median':
                ts = np.nanmedian(stack, axis=(1, 2))
            else:
                raise ValueError(f"Invalid statistic: {loc}. Use 'mean', 'min', 'max', or 'median'")
        else:
            lat, lon = loc
            row, col = latlon_to_pixel(lat, lon, transform, crs)

            if not (0 <= row < h and 0 <= col < w):
                raise ValueError(f"Point ({lat}, {lon}) outside raster bounds")

            ts = stack[:, row, col]

        plt.plot(pd.to_datetime(dates), ts, label=label)

    plt.legend()
    plt.grid(True)
    plt.ylabel("LST (°C)")
    plt.xlabel("Date")
    plt.title("LST Time Series")
    plt.show()


def zip_to_video(zip_path, out_mp4, fps=2, cmap='RdBu_r', pmin=1, pmax=99):
    """
    Create a video from a ZIP of GeoTIFFs using percentile scaling.

    Parameters
    ----------
    zip_path : str
        Path to ZIP file containing GeoTIFFs
    out_mp4 : str
        Output video file path
    fps : int, default 2
        Frames per second
    cmap : str, default 'RdBu_r'
        Matplotlib colormap name
    pmin : float, default 1
        Minimum percentile for scaling
    pmax : float, default 99
        Maximum percentile for scaling
    """
    # Load from cache or ZIP
    data = load_zip_stack(zip_path)
    stack = data["stack"]
    dates = data["dates"]

    # Percentile-based scaling
    vmin, vmax = np.nanpercentile(stack, [pmin, pmax])

    # Temporary frame directory
    temp_dir = "_frames"
    os.makedirs(temp_dir, exist_ok=True)

    frame_paths = []
    cmap_obj = plt.get_cmap(cmap)
    cmap_obj.set_bad('lightgrey')

    # Create frames
    for i, img in enumerate(stack):
        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(img, cmap=cmap_obj, vmin=vmin, vmax=vmax)

        ax.set_title(f"LST - {dates[i]}")
        ax.axis('off')

        cbar = fig.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label("LST (°C)")

        frame_path = os.path.join(temp_dir, f"frame_{i:04d}.png")
        plt.savefig(frame_path, dpi=150, bbox_inches='tight')
        plt.close(fig)

        frame_paths.append(frame_path)

    # Create video
    first = cv2.imread(frame_paths[0])
    h, w, _ = first.shape

    writer = cv2.VideoWriter(
        out_mp4,
        cv2.VideoWriter_fourcc(*'mp4v'),
        fps,
        (w, h)
    )

    for f in frame_paths:
        writer.write(cv2.imread(f))

    writer.release()

    # Cleanup frames
    for f in frame_paths:
        os.remove(f)
    os.rmdir(temp_dir)

    print(f"Video saved to: {out_mp4}")
    print(f"Scaling used: {pmin}–{pmax} percentile → [{vmin:.2f}, {vmax:.2f}] °C")
