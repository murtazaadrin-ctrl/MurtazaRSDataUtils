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

    return final_img

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

        # Create a shallow copy so original data_config is untouched
        data_config_tmp = data_config.copy()
        data_config_tmp["start_date"] = time_interval[0]
        data_config_tmp["end_date"] = time_interval[1]

        img, cloud_mask = get_sentinel_image_10_bands(
            sh_config=sh_config,
            data_config=data_config_tmp,
            plot=plot,
            cloud_masking=cloud_masking
        )

        sentinel2_images.append(img)
        cloud_masks.append(cloud_mask)

    # Log run time in IST
    ist = pytz.timezone("Asia/Kolkata")
    current_time_ist = datetime.now(ist)
    print("Last Run :", current_time_ist.strftime("%H:%M:%S"))

    return sentinel2_images, dates, cloud_masks
