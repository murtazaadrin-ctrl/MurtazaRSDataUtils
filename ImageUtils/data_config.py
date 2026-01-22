import math
import datetime
from sentinelhub import SHConfig,SentinelHubRequest, SentinelHubCatalog, DataCollection, bbox_to_dimensions, MimeType, BBox, CRS, CustomUrlParam
from shapely.geometry import Polygon

def haversine(lat1, lon1, lat2, lon2):
    """Calculate the great-circle distance between two points on the Earth."""
    R = 6378137  # Radius of the Earth in meters
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)

    a = math.sin(delta_phi / 2)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    return R * c  # Distance in meters

def bbox_to_dimensions_alternative(bbox, resolution):
    xmin, ymin, xmax, ymax = bbox

    # Calculate midpoint latitude and longitude
    mid_lat = (ymin + ymax) / 2
    mid_lon = (xmin + xmax) / 2

    # Width in meters from (mid_lat, xmin) to (mid_lat, xmax)
    width = haversine(mid_lat, xmin, mid_lat, xmax)  # Distance in meters

    # Height in meters from (ymin, mid_lon) to (ymax, mid_lon)
    height = haversine(ymin, mid_lon, ymax, mid_lon)  # Distance in meters

    return width/resolution, height/resolution


def convert_date1(date_string):
    # Parse the input date string
    date_obj = datetime.strptime(date_string, "%Y-%m-%dT%H:%M:%SZ")
    # Format the date to DD MM YYYY
    return date_obj.strftime("%d %m %Y")


def get_polygon_coords(lon, lat, distancex, distancey):
    # Earth's radius in meters
    R = 6378137
    distancex = distancex*1000
    distancey = distancey*1000
    # Convert distances from meters to degrees
    d_lat = distancey / R
    d_lon = distancex / (R * math.cos(math.pi * lat / 180))

    # Convert degrees to radians
    d_lat = d_lat * 180 / math.pi
    d_lon = d_lon * 180 / math.pi

    # Calculate the coordinates of the rectangle corners
    top_left = (lon - d_lon / 2, lat + d_lat / 2)
    top_right = (lon + d_lon / 2, lat + d_lat / 2)
    bottom_left = (lon - d_lon / 2, lat - d_lat / 2)
    bottom_right = (lon + d_lon / 2, lat - d_lat / 2)

    return [top_left, top_right, bottom_right, bottom_left, top_left]


def get_data_config(coords, date_range, distances= (40,40), resolution = 42):
    # Example usage
    lat =   coords[0]
    lon =   coords[1]

    distancex = distances[0] # km
    distancey =  distances[1] # km

    polygon_coords = get_polygon_coords(lon, lat, distancex, distancey)

    # Create a Shapely Polygon object
    polygon = Polygon(polygon_coords)

    # Create a BBox object with WGS84 CRS
    bbox = BBox(polygon.bounds, crs=CRS.WGS84)

    # Determine the size of the polygon in pixels based on resolution # meters
    size = bbox_to_dimensions_alternative(bbox, resolution=resolution)
    print("Size:", (size[0], size[1]))

    start_date = date_range[0]
    end_date = date_range[1]
    return {
        "bbox": bbox,
        "size": {
            "width": size[0],
            "height": size[1]
        },
        "resolution": resolution,
        "distances_km": {
            "x": distancex,
            "y": distancey
        },
        "coords": {
            "lat": lat,
            "lon": lon
        },
        "start_date": start_date,
        "end_date": end_date
    }