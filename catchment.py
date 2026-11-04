#!/usr/bin/env python3
"""Catchment delineation as a reusable library.

Everything here is independent of the web server, so the flow engine, the
CLI and tests can all compute a catchment the same way sheed.py does.

    result = await delineate(lat, lon)
    result.polygon        # shapely Polygon (lon/lat)
    result.area_km2       # drainage area
    result.to_geojson()   # FeatureCollection dict
"""
from __future__ import annotations

import argparse
import asyncio
import dataclasses
import hashlib
import json
import logging
import math
import os
from typing import Awaitable, Callable, Optional

import aiohttp
import numpy as np
from pysheds.grid import Grid
from shapely.geometry import Polygon, mapping, shape as shapely_shape

DATASET_ARCSEC = {"USGS30m": 1.0, "USGS10m": 1.0 / 3.0}
DEFAULT_DATASET = "USGS10m"
DEFAULT_EXPAND = 0.05
RETRY_EXPAND = 0.1
SNAP_ACC_THRESHOLD = 1000
DIRMAP = (64, 128, 1, 2, 4, 8, 16, 32)
USGS_3DEP_URL = (
    "https://elevation.nationalmap.gov/arcgis/rest/services/"
    "3DEPElevation/ImageServer/exportImage"
)
METERS_PER_DEGREE = 111_320.0
CACHE_VERSION = 1

LogFn = Callable[[str], Awaitable[None]]


async def _default_log(msg: str) -> None:
    logging.info(msg)


@dataclasses.dataclass(frozen=True)
class BBox:
    min_x: float
    min_y: float
    max_x: float
    max_y: float

    @classmethod
    def around(cls, lat: float, lon: float, expand: float) -> "BBox":
        return cls(
            round(lon - expand, 5),
            round(lat - expand, 5),
            round(lon + expand, 5),
            round(lat + expand, 5),
        )


@dataclasses.dataclass
class CatchmentResult:
    lat: float
    lon: float
    dataset: str
    expand_factor: float
    polygons: list[Polygon]
    snapped: Optional[tuple[float, float]]  # (lon, lat) of the pour point used
    area_km2: float
    elev_min_m: float
    elev_mean_m: float
    elev_max_m: float
    clipped: bool

    @property
    def polygon(self) -> Polygon:
        """The largest polygon (pysheds can occasionally return slivers)."""
        return max(self.polygons, key=lambda p: p.area)

    def stats(self) -> dict:
        return {
            "lat": self.lat,
            "lon": self.lon,
            "dataset": self.dataset,
            "expand_factor": self.expand_factor,
            "snapped": list(self.snapped) if self.snapped else None,
            "area_km2": round(self.area_km2, 4),
            "elev_min_m": round(self.elev_min_m, 1),
            "elev_mean_m": round(self.elev_mean_m, 1),
            "elev_max_m": round(self.elev_max_m, 1),
            "clipped": self.clipped,
        }

    def to_geojson(self) -> dict:
        return {
            "type": "FeatureCollection",
            "features": [
                {"type": "Feature", "geometry": mapping(p), "properties": self.stats()}
                for p in self.polygons
            ],
        }

    def to_json(self) -> str:
        return json.dumps(self.to_geojson())

    @classmethod
    def from_json(cls, text: str) -> "CatchmentResult":
        fc = json.loads(text)
        props = fc["features"][0]["properties"]
        return cls(
            lat=props["lat"],
            lon=props["lon"],
            dataset=props["dataset"],
            expand_factor=props["expand_factor"],
            polygons=[shapely_shape(f["geometry"]) for f in fc["features"]],
            snapped=tuple(props["snapped"]) if props["snapped"] else None,
            area_km2=props["area_km2"],
            elev_min_m=props["elev_min_m"],
            elev_mean_m=props["elev_mean_m"],
            elev_max_m=props["elev_max_m"],
            clipped=props["clipped"],
        )


def dem_request_params(bbox: BBox, dataset: str) -> dict:
    # Match the source's native arc-second grid so output cells are square
    # in degrees — pysheds' D8 flow direction assumes square pixels, and a
    # cos(lat) correction here gave non-square cells that snapped to the
    # wrong drainage.
    deg_per_cell = DATASET_ARCSEC[dataset] / 3600.0
    width_px = max(1, round((bbox.max_x - bbox.min_x) / deg_per_cell))
    height_px = max(1, round((bbox.max_y - bbox.min_y) / deg_per_cell))
    return {
        "bbox": f"{bbox.min_x},{bbox.min_y},{bbox.max_x},{bbox.max_y}",
        "bboxSR": "4326",
        "imageSR": "4326",
        "size": f"{width_px},{height_px}",
        "format": "tiff",
        "pixelType": "F32",
        "noData": "-9999",
        "interpolation": "RSP_NearestNeighbor",
        "f": "image",
    }


async def fetch_dem(
    bbox: BBox, dataset: str, path: str, log: LogFn = _default_log
) -> str:
    """Download a DEM GeoTIFF for bbox to path (skipped if already present)."""
    if os.path.isfile(path):
        await log(f"Using cached DEM {path}")
        return path

    params = dem_request_params(bbox, dataset)
    await log(f"Fetching {dataset} DEM ({params['size'].replace(',', 'x')} px) from USGS 3DEP...")
    async with aiohttp.ClientSession() as session:
        async with session.get(USGS_3DEP_URL, params=params) as response:
            if response.status != 200:
                body = await response.text()
                raise RuntimeError(
                    f"USGS 3DEP returned {response.status} for {params}: {body[:500]}"
                )
            content = await response.read()
    tmp = f"{path}.part"
    with open(tmp, "wb") as f:
        f.write(content)
    os.replace(tmp, path)
    await log(f"{path} downloaded successfully.")
    return path


def compute_catchment(
    dem_path: str,
    lon: float,
    lat: float,
    snap: bool = True,
    snap_threshold: int = SNAP_ACC_THRESHOLD,
) -> tuple[list[Polygon], Optional[tuple[float, float]], np.ndarray, float]:
    """Delineate the catchment draining to (lon, lat) on the DEM.

    Returns (polygons, snapped_point, catchment_elevations, cell_size_deg).
    CPU-bound; call via asyncio.to_thread from async code.
    """
    grid = Grid.from_raster(dem_path)
    dem = grid.read_raster(dem_path)

    inflated = grid.resolve_flats(grid.fill_depressions(grid.fill_pits(dem)))
    fdir = grid.flowdir(inflated, dirmap=DIRMAP)

    snapped = None
    x, y = lon, lat
    if snap:
        acc = grid.accumulation(fdir, dirmap=DIRMAP)
        x, y = grid.snap_to_mask(acc > snap_threshold, (lon, lat))
        snapped = (float(x), float(y))

    catch = grid.catchment(x=x, y=y, fdir=fdir, dirmap=DIRMAP, xytype="coordinate")

    elevations = np.asarray(dem)[np.asarray(catch, dtype=bool)]
    cell_size = abs(float(grid.affine.a))

    grid.clip_to(catch)
    catch_view = grid.view(catch, dtype=np.uint8)
    polygons = [
        shapely_shape(geom) for geom, value in grid.polygonize(catch_view) if value
    ]
    return polygons, snapped, elevations, cell_size


def catchment_area_km2(n_cells: int, cell_size_deg: float, lat: float) -> float:
    """Area of n square-in-degree cells near latitude lat."""
    cell_w = cell_size_deg * METERS_PER_DEGREE * math.cos(math.radians(lat))
    cell_h = cell_size_deg * METERS_PER_DEGREE
    return n_cells * cell_w * cell_h / 1e6


def is_clipped(polygon: Polygon, bbox: BBox, dataset: str) -> bool:
    """True if the catchment touches the DEM edge, i.e. is probably truncated."""
    # A polygonized vertex sitting on the bbox edge can land up to one cell
    # inward, so 2 cells is a robust margin at any resolution.
    threshold = 2 * DATASET_ARCSEC[dataset] / 3600.0
    for x, y in polygon.exterior.coords:
        d = min(
            abs(bbox.max_x - x),
            abs(bbox.min_x - x),
            abs(bbox.max_y - y),
            abs(bbox.min_y - y),
        )
        if d < threshold:
            return True
    return False


def cache_key(lat: float, lon: float, dataset: str, snap: bool, expand: float) -> str:
    raw = f"v{CACHE_VERSION}|{lat:.5f}|{lon:.5f}|{dataset}|{int(bool(snap))}|{expand:.5f}"
    return hashlib.sha1(raw.encode()).hexdigest()[:16]


async def delineate(
    lat: float,
    lon: float,
    dataset: str = DEFAULT_DATASET,
    snap: bool = True,
    expand_factor: float = DEFAULT_EXPAND,
    workdir: str = "output",
    name: Optional[str] = None,
    use_cache: bool = True,
    log: LogFn = _default_log,
) -> CatchmentResult:
    """Fetch a DEM and delineate the catchment for a point, with one automatic
    retry on a larger DEM if the first result touches the edge.

    Results are cached in <workdir>/cache keyed by the request, so repeat
    lookups of the same canyon skip the DEM download and computation.
    """
    if dataset not in DATASET_ARCSEC:
        raise ValueError(f"unknown dataset {dataset!r}")

    cache_dir = os.path.join(workdir, "cache")
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(
        cache_dir, f"catchment_{cache_key(lat, lon, dataset, snap, expand_factor)}.geojson"
    )
    if use_cache and os.path.isfile(cache_path):
        await log(f"Using cached catchment {cache_path}")
        with open(cache_path) as f:
            return CatchmentResult.from_json(f.read())

    result = await _delineate_once(lat, lon, dataset, snap, expand_factor, workdir, name, log)
    if result.clipped and expand_factor < RETRY_EXPAND:
        await log(f"Clipping detected, retrying with expand_factor={RETRY_EXPAND}")
        result = await _delineate_once(lat, lon, dataset, snap, RETRY_EXPAND, workdir, name, log)

    tmp = f"{cache_path}.part"
    with open(tmp, "w") as f:
        f.write(result.to_json())
    os.replace(tmp, cache_path)
    return result


async def _delineate_once(
    lat, lon, dataset, snap, expand_factor, workdir, name, log
) -> CatchmentResult:
    bbox = BBox.around(lat, lon, expand_factor)
    label = name or f"{lat}_{lon}_{expand_factor}"
    dem_path = os.path.join(workdir, f"dem_{dataset}_{label}.tif")
    await fetch_dem(bbox, dataset, dem_path, log)

    await log("Calculating catchment")
    polygons, snapped, elevations, cell_size = await asyncio.to_thread(
        compute_catchment, dem_path, lon, lat, snap
    )
    if not polygons:
        raise RuntimeError("no catchment found for this point")

    valid = elevations[elevations > -9999]
    result = CatchmentResult(
        lat=lat,
        lon=lon,
        dataset=dataset,
        expand_factor=expand_factor,
        polygons=polygons,
        snapped=snapped,
        area_km2=catchment_area_km2(int(elevations.size), cell_size, lat),
        elev_min_m=float(valid.min()) if valid.size else float("nan"),
        elev_mean_m=float(valid.mean()) if valid.size else float("nan"),
        elev_max_m=float(valid.max()) if valid.size else float("nan"),
        clipped=False,
    )
    result.clipped = is_clipped(result.polygon, bbox, dataset)
    if result.clipped:
        await log("Catchment touches the DEM edge and may be truncated")
    await log(f"Catchment area {result.area_km2:.2f} km²")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Delineate the catchment for a point.")
    parser.add_argument("lat", type=float)
    parser.add_argument("lon", type=float)
    parser.add_argument("--dem", choices=sorted(DATASET_ARCSEC), default=DEFAULT_DATASET)
    parser.add_argument("--no-snap", action="store_true", help="Don't snap to the nearest stream")
    parser.add_argument("--expand", type=float, default=DEFAULT_EXPAND)
    parser.add_argument("--output", help="Write GeoJSON here (default: print stats only)")
    parser.add_argument("--no-cache", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    result = asyncio.run(
        delineate(
            args.lat, args.lon, dataset=args.dem, snap=not args.no_snap,
            expand_factor=args.expand, use_cache=not args.no_cache,
        )
    )
    if args.output:
        with open(args.output, "w") as f:
            f.write(result.to_json())
    print(json.dumps(result.stats(), indent=2))


if __name__ == "__main__":
    main()
