#!/usr/bin/env python
"""Fetch latest Sentinel-2 L1C scene in a date range for a centered AOI.

Default behavior: return an in-memory dictionary with metadata and arrays:
- mx_vnir: 42 m, 6 bands
- hys: 191 m, 10 bands
- cloud_probability (CLP): 42 m, 1 band in [0, 1]
- clm: thresholded cloud mask from CLP at 42 m

Optional behavior: pass save_outputs=True to also write TIFF/YAML files.
"""

from __future__ import annotations

import argparse
import math
import os
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import rasterio
from rasterio.transform import from_bounds
from sentinelhub import (
    BBox,
    CRS,
    DataCollection,
    MimeType,
    SHConfig,
    SentinelHubCatalog,
    SentinelHubRequest,
    bbox_to_dimensions,
)

MX_BANDS: Sequence[str] = ("B02", "B03", "B04", "B06", "B07", "B8A")
HYS_BANDS: Sequence[str] = ("B02", "B03", "B04", "B05", "B06", "B07", "B8A", "B09", "B11", "B12")
CLP_BANDS: Sequence[str] = ("CLP",)


@dataclass
class Acquisition:
    dt: datetime
    tile: str


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fetch latest mx_vnir + hys around a center coordinate")
    p.add_argument("--start", required=True, help="Start date/time, e.g. 2025-11-01 or 2025-11-01T00:00:00Z")
    p.add_argument("--end", required=True, help="End date/time, e.g. 2026-02-01 or 2026-02-01T23:59:59Z")
    p.add_argument("--lat", type=float, required=True, help="Center latitude in WGS84")
    p.add_argument("--lon", type=float, required=True, help="Center longitude in WGS84")
    p.add_argument(
        "--half-size-m",
        type=float,
        default=1500.0,
        help="Half side length (meters) for centered square AOI. Default: 1500 m",
    )
    p.add_argument("--min-coverage-pct", type=float, default=50.0, help="Minimum AOI data coverage percent required.")
    p.add_argument(
        "--cloud-prob-threshold",
        type=float,
        default=0.4,
        help="Threshold on CLP probability (0..1) to derive clm.",
    )
    p.add_argument("--out-dir", default=".", help="Output root directory")
    p.add_argument("--sh-client-id", default=None, help="Sentinel Hub OAuth client id")
    p.add_argument("--sh-client-secret", default=None, help="Sentinel Hub OAuth client secret")
    p.add_argument(
        "--save-outputs",
        action="store_true",
        help="If set, save TIFF/YAML outputs to disk; otherwise return only in-memory dictionary.",
    )
    return p.parse_args()


def build_config(client_id: Optional[str] = None, client_secret: Optional[str] = None) -> SHConfig:
    cfg = SHConfig()
    cfg.sh_client_id = client_id or os.environ.get("SH_CLIENT_ID", "")
    cfg.sh_client_secret = client_secret or os.environ.get("SH_CLIENT_SECRET", "")

    if not cfg.sh_client_id or not cfg.sh_client_secret:
        raise RuntimeError("Sentinel Hub credentials missing. Set SH_CLIENT_ID and SH_CLIENT_SECRET environment variables.")
    return cfg


def parse_iso_utc(value: str) -> datetime:
    text = value.strip()
    if len(text) == 10:
        text = f"{text}T00:00:00Z"
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    dt = datetime.fromisoformat(text)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def centered_bbox(lon: float, lat: float, half_size_m: float) -> BBox:
    earth_radius = 6_378_137.0
    d_lat = (half_size_m / earth_radius) * (180.0 / math.pi)
    d_lon = (half_size_m / (earth_radius * math.cos(math.radians(lat)))) * (180.0 / math.pi)
    return BBox((lon - d_lon, lat - d_lat, lon + d_lon, lat + d_lat), crs=CRS.WGS84)


def get_acquisitions_desc(bbox: BBox, start: str, end: str, config: SHConfig) -> List[Acquisition]:
    catalog = SentinelHubCatalog(config=config)
    search = catalog.search(
        collection=DataCollection.SENTINEL2_L1C,
        bbox=bbox,
        time=(start, end),
        fields={"include": ["properties.datetime", "properties.s2:mgrs_tile"], "exclude": []},
    )

    acqs: List[Acquisition] = []
    for item in search:
        dt = parse_iso_utc(item["properties"]["datetime"])
        tile = item["properties"].get("s2:mgrs_tile", "XXXXX")
        tile = tile if tile.startswith("T") else f"T{tile}"
        acqs.append(Acquisition(dt=dt, tile=tile))

    acqs.sort(key=lambda x: x.dt, reverse=True)
    return acqs


def get_datamask_coverage_pct(bbox: BBox, acq_dt: datetime, config: SHConfig, resolution_m: float = 42) -> float:
    evalscript = """
//VERSION=3
function setup() {
  return {
    input: [{ bands: [\"dataMask\"] }],
    output: { bands: 1, sampleType: \"UINT8\" }
  };
}
function evaluatePixel(s) { return [s.dataMask]; }
"""
    time_window = (
        (acq_dt - timedelta(minutes=30)).strftime("%Y-%m-%dT%H:%M:%SZ"),
        (acq_dt + timedelta(minutes=30)).strftime("%Y-%m-%dT%H:%M:%SZ"),
    )

    req = SentinelHubRequest(
        evalscript=evalscript,
        input_data=[
            SentinelHubRequest.input_data(
                data_collection=DataCollection.SENTINEL2_L1C,
                time_interval=time_window,
                mosaicking_order="mostRecent",
            )
        ],
        responses=[SentinelHubRequest.output_response("default", MimeType.TIFF)],
        bbox=bbox,
        size=bbox_to_dimensions(bbox, resolution=resolution_m),
        config=config,
    )

    arr = req.get_data(save_data=False)[0]
    if arr.ndim == 3:
        arr = arr[:, :, 0]
    return float(np.mean(arr == 1) * 100.0)


def get_latest_acquisition_with_min_coverage(
    bbox: BBox,
    start: str,
    end: str,
    config: SHConfig,
    min_coverage_pct: float = 50.0,
) -> Tuple[Acquisition, float]:
    acqs = get_acquisitions_desc(bbox, start, end, config)

    for acq in acqs:
        cov = get_datamask_coverage_pct(bbox, acq.dt, config, resolution_m=42)
        if cov >= min_coverage_pct:
            return acq, cov

    raise RuntimeError(f"No acquisition found with >= {min_coverage_pct:.1f}% AOI coverage.")


def make_evalscript(bands: Sequence[str]) -> str:
    band_list = ", ".join(f'"{b}"' for b in bands)
    band_return = ", ".join(f"sample.{b}" for b in bands)
    return f"""
//VERSION=3
function setup() {{
  return {{
    input: [{{ bands: [{band_list}] }}],
    output: {{ bands: {len(bands)}, sampleType: \"FLOAT32\" }}
  }};
}}
function evaluatePixel(sample) {{
  return [{band_return}];
}}
"""


def fetch_stack(
    bbox: BBox,
    resolution_m: float,
    bands: Sequence[str],
    acquisition_dt: datetime,
    config: SHConfig,
) -> np.ndarray:
    time_window = (
        (acquisition_dt - timedelta(minutes=30)).strftime("%Y-%m-%dT%H:%M:%SZ"),
        (acquisition_dt + timedelta(minutes=30)).strftime("%Y-%m-%dT%H:%M:%SZ"),
    )

    req = SentinelHubRequest(
        evalscript=make_evalscript(bands),
        input_data=[
            SentinelHubRequest.input_data(
                data_collection=DataCollection.SENTINEL2_L1C,
                time_interval=time_window,
                mosaicking_order="mostRecent",
            )
        ],
        responses=[SentinelHubRequest.output_response("default", MimeType.TIFF)],
        bbox=bbox,
        size=bbox_to_dimensions(bbox, resolution=resolution_m),
        config=config,
    )

    data = req.get_data(save_data=False)
    if not data:
        raise RuntimeError("No raster returned from Sentinel Hub request.")
    return data[0]


def write_tif(path: Path, bbox: BBox, arr_hwc: np.ndarray) -> None:
    h, w, bands = arr_hwc.shape
    transform = from_bounds(*bbox, w, h)

    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=w,
        height=h,
        count=bands,
        dtype=arr_hwc.dtype,
        crs="EPSG:4326",
        transform=transform,
        tiled=True,
        blockxsize=512,
        blockysize=512,
        compress="deflate",
    ) as dst:
        for idx in range(bands):
            dst.write(arr_hwc[:, :, idx], idx + 1)


def write_yaml(path: Path, product: str, date_tag: str, tile: str, file_name: str, dt: datetime, mode: str) -> None:
    text = f"""$schema: https://schemas.opendatacube.org/dataset
id: {product}_{date_tag}_{tile}
label: {product}_{date_tag}_{tile}
product:
  name: {product}
crs: auto
geometry: auto
grids:
  default:
    shape: auto
    transform: auto
properties:
  datetime: {dt.strftime('%Y-%m-%dT%H:%M:%SZ')}
  eo:platform: sentinel-2
  eo:instrument: msi
  g1a:mode: {mode}
measurements:
  default:
    path: {file_name}
"""
    path.write_text(text, encoding="utf-8")


def fetch_latest_g1a_centered(
    start: str,
    end: str,
    lat: float,
    lon: float,
    half_size_m: float = 1500.0,
    min_coverage_pct: float = 50.0,
    out_dir: str = ".",
    client_id: Optional[str] = None,
    client_secret: Optional[str] = None,
    config: Optional[SHConfig] = None,
    save_outputs: bool = False,
    cloud_prob_threshold: float = 0.4,
) -> Dict[str, Any]:
    cfg = config or build_config(client_id=client_id, client_secret=client_secret)

    start_dt = parse_iso_utc(start)
    end_dt = parse_iso_utc(end)
    if end_dt < start_dt:
        raise ValueError("end must be greater than or equal to start")

    if not 0.0 <= cloud_prob_threshold <= 1.0:
        raise ValueError("cloud_prob_threshold must be in [0, 1]")

    bbox = centered_bbox(lon, lat, half_size_m)
    acq, coverage_pct = get_latest_acquisition_with_min_coverage(
        bbox=bbox,
        start=start_dt.strftime("%Y-%m-%dT%H:%M:%SZ"),
        end=end_dt.strftime("%Y-%m-%dT%H:%M:%SZ"),
        config=cfg,
        min_coverage_pct=min_coverage_pct,
    )

    date_tag = acq.dt.strftime("%Y%m%dT%H%M%S")
    output_dir = Path(out_dir) / f"G1A_MOCK_{date_tag}_{acq.tile}"

    mx = fetch_stack(bbox=bbox, resolution_m=42, bands=MX_BANDS, acquisition_dt=acq.dt, config=cfg)
    hys = fetch_stack(bbox=bbox, resolution_m=191, bands=HYS_BANDS, acquisition_dt=acq.dt, config=cfg)

    clp_raw = fetch_stack(bbox=bbox, resolution_m=42, bands=CLP_BANDS, acquisition_dt=acq.dt, config=cfg)
    clp = clp_raw if clp_raw.ndim == 2 else clp_raw[:, :, 0]
    clp = clp.astype(np.float32)
    if np.nanmax(clp) > 1.0:
        clp = clp / 255.0
    clm = (clp >= cloud_prob_threshold).astype(np.uint8)

    mx_name = f"MX_VNIR_{date_tag}_{acq.tile}.tif"
    hys_name = f"HYS__{date_tag}_{acq.tile}.tif"
    mx_path = output_dir / mx_name
    hys_path = output_dir / hys_name
    mx_yaml_path = output_dir / f"dataset_g1a_mx_vnir_{date_tag}_{acq.tile}.yaml"
    hys_yaml_path = output_dir / f"dataset_g1a_hys_{date_tag}_{acq.tile}.yaml"

    if save_outputs:
        write_tif(mx_path, bbox, mx)
        write_tif(hys_path, bbox, hys)
        write_yaml(mx_yaml_path, "g1a_mx_vnir", date_tag, acq.tile, mx_name, acq.dt, "MX")
        write_yaml(hys_yaml_path, "g1a_hys", date_tag, acq.tile, hys_name, acq.dt, "HYS")

    return {
        "acquisition_datetime": acq.dt.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "tile": acq.tile,
        "coverage_pct": coverage_pct,
        "bbox_wgs84": tuple(bbox),
        "mx_vnir": {
            "bands": list(MX_BANDS),
            "resolution_m": 42,
            "array": mx,
        },
        "hys": {
            "bands": list(HYS_BANDS),
            "resolution_m": 191,
            "array": hys,
        },
        "cloud_probability": {
            "bands": ["CLP"],
            "resolution_m": 42,
            "array": clp,
        },
        "clm": {
            "bands": ["CLM_from_CLP"],
            "resolution_m": 42,
            "threshold": cloud_prob_threshold,
            "array": clm,
        },
        "saved": save_outputs,
        "paths": {
            "output_dir": str(output_dir),
            "mx_path": str(mx_path),
            "hys_path": str(hys_path),
            "mx_yaml_path": str(mx_yaml_path),
            "hys_yaml_path": str(hys_yaml_path),
        },
    }


def main() -> None:
    args = parse_args()
    result = fetch_latest_g1a_centered(
        start=args.start,
        end=args.end,
        lat=args.lat,
        lon=args.lon,
        half_size_m=args.half_size_m,
        min_coverage_pct=args.min_coverage_pct,
        out_dir=args.out_dir,
        client_id=args.sh_client_id,
        client_secret=args.sh_client_secret,
        save_outputs=args.save_outputs,
        cloud_prob_threshold=args.cloud_prob_threshold,
    )
    print(f"Latest acquisition: {result['acquisition_datetime']} ({result['tile']})")
    print(f"Coverage %: {result['coverage_pct']:.2f}")
    print(f"CLP threshold: {result['clm']['threshold']}")
    print(f"Saved outputs: {result['saved']}")
    if result["saved"]:
        print(f"Output directory: {result['paths']['output_dir']}")
        print(f"MX_VNIR: {result['paths']['mx_path']}")
        print(f"HYS: {result['paths']['hys_path']}")


if __name__ == "__main__":
    main()
