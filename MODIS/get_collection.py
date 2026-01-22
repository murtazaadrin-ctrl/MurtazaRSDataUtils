import ee


def get_modis_collection(
    data_config,
    product=None,
    bands=(None, None)
):
    """
    Returns a MODIS LST ImageCollection clipped to data_config ROI and dates.

    Parameters
    ----------
    data_config : dict
        Output of get_data_config
    product : str
        MODIS product ID (default: MOD11A1 – daily)
    bands : tuple
        Bands to select

    Returns
    -------
    ee.ImageCollection
    """

    # Extract bbox bounds from data_config
    bbox = data_config["bbox"]
    min_lon, min_lat, max_lon, max_lat = bbox.bbox

    roi = ee.Geometry.Rectangle([
        min_lon, min_lat,
        max_lon, max_lat
    ])

    start_date = data_config["start_date"]
    end_date = data_config["end_date"]

    collection = (
        ee.ImageCollection(product)
        .select(list(bands))
        .filterDate(start_date, end_date)
        .filterBounds(roi)
    )

    return collection

# ----------------------------------------------------------------------------
# MODIS LST Daily Collections
# ----------------------------------------------------------------------------
def get_modis_day_lst_daily(data_config):
    return get_modis_collection(
        data_config=data_config,
        product="MODIS/061/MOD11A1",
        bands=("LST_Day_1km", "QC_Day")
    )

def get_modis_night_lst_daily(data_config):
    return get_modis_collection(
        data_config=data_config,
        product="MODIS/061/MOD11A1",
        bands=("LST_Night_1km", "QC_Night")
    )

def get_modis_day_lst_8day(data_config):
    return get_modis_collection(
        data_config=data_config,
        product="MODIS/061/MOD11A2",
        bands=("LST_Day_1km", "QC_Day")
    )

def get_modis_night_lst_8day(data_config):
    return get_modis_collection(
        data_config=data_config,
        product="MODIS/061/MOD11A2",
        bands=("LST_Night_1km", "QC_Night")
    )

# ----------------------------------------------------------------------------
# Aqua MODIS LST Daily Collections
# ----------------------------------------------------------------------------


def get_aqua_modis_day_lst_daily(data_config):
    return get_modis_collection(
        data_config=data_config,
        product="MODIS/061/MYD11A1",
        bands=("LST_Day_1km", "QC_Day")
    )

def get_aqua_modis_night_lst_daily(data_config):
    return get_modis_collection(
        data_config=data_config,
        product="MODIS/061/MYD11A1",
        bands=("LST_Night_1km", "QC_Night")
    )

def get_aqua_modis_day_lst_8day(data_config):
    return get_modis_collection(
        data_config=data_config,
        product="MODIS/061/MYD11A2",
        bands=("LST_Day_1km", "QC_Day")
    )

def get_aqua_modis_night_lst_8day(data_config):
    return get_modis_collection(
        data_config=data_config,
        product="MODIS/061/MYD11A2",
        bands=("LST_Night_1km", "QC_Night")
    )






