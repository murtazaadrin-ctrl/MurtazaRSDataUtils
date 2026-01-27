from sentinelhub import SHConfig,SentinelHubRequest, SentinelHubCatalog, DataCollection, bbox_to_dimensions, MimeType, BBox, CRS, CustomUrlParam
import numpy as np
import matplotlib.pyplot as plt
import datetime
from datetime import datetime, timedelta
import pytz

def get_sentinel_image_10_bands(config , time_interval, bbox, size, plot=False):
    request = SentinelHubRequest(
        evalscript=f"""
            //VERSION=3
            function setup() {{
                return {{
                    input: [{{
                        bands: ["B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12", "CLM"]
                    }}],
                    output: {{
                        bands: 11,
                        sampleType: "FLOAT32",  // Explicitly request 16-bit output
                        interpolation: "LANCZOS"
                    }}
                }};
            }}

            function evaluatePixel(sample) {{
                return [sample.B02, sample.B03, sample.B04, sample.B05, sample.B06, sample.B07, sample.B08, sample.B8A, sample.B11, sample.B12, sample.CLM];
            }}
        """,
        input_data=[
            SentinelHubRequest.input_data(
                data_collection=DataCollection.SENTINEL2_L2A,
                time_interval=time_interval,
                mosaicking_order='leastCC'
            )
        ],
        responses=[
            SentinelHubRequest.output_response('default', MimeType.TIFF)
        ],
        bbox=bbox,
        size=size,
        config=config
    )

    image_data = request.get_data(save_data=False)
    a = image_data[0]

    #print(a.dtype)

    # Extract the bands and the cloud mask
    bands = a[:, :, :10]
    clm = a[:, :, 10]

    clm1= np.zeros_like(clm)

    # Process the image
    final_img = np.zeros((np.shape(a)[0], np.shape(a)[1], 10))
    for i in range(10):
      final_img[:, :, i] = np.where(clm1 == 0, bands[:,:,i], np.nan)

    # Normalize and clip images
    a1 = np.nan_to_num(np.clip(final_img[:, :, [2, 1, 0]] / np.nanpercentile(final_img[:, :, [2, 1, 0]], 99), 0, 1), 0)
    a2 = np.nan_to_num(np.clip(final_img[:, :, [6, 3, 2]] / np.nanpercentile(final_img[:, :, [6, 3, 2]], 99), 0, 1), 0)

    if plot:
        # Create a figure
        fig = plt.figure(figsize=(10, 5))

        # Add the first subplot and plot the image
        ax1 = fig.add_subplot(1, 2, 1)
        ax1.imshow(a1)
        ax1.set_title('NCC')
        ax1.axis('off')

        # Add the second subplot and plot the image
        ax2 = fig.add_subplot(1, 2, 2)
        ax2.imshow(a2)
        ax2.set_title('FCC')
        ax2.axis('off')

        # Show the plot
        plt.show()

    # Return both the image and cloud mask
    # clm is the cloud mask (0 = no cloud, >0 = cloud)
    return final_img, clm

def get_sentinel2_catalog_results(
    sh_config,
    data_config,
    min_time_diff_seconds=3600
):
    """
    Returns Sentinel-2 L2A catalog results filtered by:
    - cloud cover
    - minimum time difference between acquisitions

    Parameters
    ----------
    sh_config : SHConfig
        Sentinel Hub configuration
    data_config : dict
        Output of get_data_config (must include max_cloud_cover for S2)
    min_time_diff_seconds : int
        Minimum time difference between acquisitions (default = 1 hour)

    Returns
    -------
    results : list
        Filtered catalog search results
    """

    bbox = data_config["bbox"]
    time_interval = (
        data_config["start_date"],
        data_config["end_date"]
    )
    
    # Get max_cloud_cover from data_config, default to 20 if not provided
    max_cloud_cover = data_config.get("max_cloud_cover", 20)

    catalog = SentinelHubCatalog(config=sh_config)

    search_iterator = catalog.search(
        DataCollection.SENTINEL2_L2A,
        bbox=bbox,
        time=time_interval,
        filter=f"eo:cloud_cover < {max_cloud_cover}"
    )

    # Reverse to chronological order (oldest → newest)
    raw_results = list(search_iterator)[::-1]

    filtered_results = []
    selected_datetimes = []

    for result in raw_results:
        dt_str = result["properties"]["datetime"]
        current_dt = datetime.strptime(dt_str, "%Y-%m-%dT%H:%M:%SZ")

        is_unique = True
        for prev_dt in selected_datetimes:
            if abs((current_dt - prev_dt).total_seconds()) < min_time_diff_seconds:
                is_unique = False
                break

        if is_unique:
            filtered_results.append(result)
            selected_datetimes.append(current_dt)

    return filtered_results

def get_time_interval(timestamp):
    """
    Given a timestamp in ISO 8601 format, return a tuple with the time interval
    1 hour before and 1 hour after the given time.

    Parameters:
        timestamp (str): The input timestamp in ISO 8601 format (e.g., '2024-03-15T05:10:52Z').

    Returns:
        tuple: A tuple containing the start and end time as ISO 8601 formatted strings.
    """
    # Convert the timestamp to a datetime object
    time = datetime.strptime(timestamp, "%Y-%m-%dT%H:%M:%SZ")

    # Calculate 1 hour before and 1 hour after
    time_before = time - timedelta(hours=1)
    time_after = time + timedelta(hours=1)

    # Convert back to ISO 8601 format (with 'Z' for UTC)
    time_before_str = time_before.strftime("%Y-%m-%dT%H:%M:%SZ")
    time_after_str = time_after.strftime("%Y-%m-%dT%H:%M:%SZ")

    return (time_before_str, time_after_str)

def generate_time_intervals(start_date: str, end_date: str, interval_days: int):
    start = datetime.strptime(start_date, '%Y-%m-%d')
    end = datetime.strptime(end_date, '%Y-%m-%d')
    intervals = []

    while start < end:
        interval_end = start + timedelta(days=interval_days - 1)  # End date inclusive
        if interval_end > end:
            interval_end = end
        intervals.append((start.strftime('%Y-%m-%d'), interval_end.strftime('%Y-%m-%d')))
        start = interval_end + timedelta(days=1)  # Move to the next interval

    return intervals

def fetch_sentinel2_images_from_results(
    sh_config,
    data_config,
    results,
    plot=False,
    cloud_masking=False,
    verbose=True
):
    """
    Returns
    -------
    sentinel2_images : list of np.ndarray
        Each element is (H, W, 10)
    dates : list of tuple
        (start_datetime, end_datetime)
    cloud_masks : list of np.ndarray
        Each element is (H, W), bool
    """

    sentinel2_images = []
    cloud_masks = []
    dates = []

    for idx, result in enumerate(results, 1):
        timestamp = result["properties"]["datetime"]
        cloud_coverage = result["properties"].get("eo:cloud_cover", "N/A")

        if verbose:
            print(f"{idx}: Date: {timestamp}, Cloud cover: {cloud_coverage}%")

        # Convert timestamp → time interval
        time_interval = get_time_interval(timestamp)
        dates.append(time_interval)

        # Extract required parameters from data_config
        bbox = data_config["bbox"]
        size = (data_config["size"]["width"], data_config["size"]["height"])

        # Call function with correct parameters
        final_img, cloud_mask_raw = get_sentinel_image_10_bands(
            config=sh_config,
            time_interval=time_interval,
            bbox=bbox,
            size=size,
            plot=plot
        )

        # Convert cloud mask to boolean (0 = no cloud, >0 = cloud)
        cloud_mask = (cloud_mask_raw > 0).astype(bool)
        
        # Apply cloud masking if requested
        if cloud_masking:
            # Apply cloud mask to the image (set cloud pixels to NaN)
            masked_img = final_img.copy()
            for i in range(10):
                masked_img[:, :, i] = np.where(cloud_mask, np.nan, final_img[:, :, i])
            sentinel2_images.append(masked_img)
        else:
            sentinel2_images.append(final_img)
        
        cloud_masks.append(cloud_mask)

    # Log run time in IST
    ist = pytz.timezone("Asia/Kolkata")
    current_time_ist = datetime.now(ist)
    print("Last Run :", current_time_ist.strftime("%H:%M:%S"))

    return sentinel2_images, dates, cloud_masks
