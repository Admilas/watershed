import os
import sys

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import catchment  # noqa: E402

# Synthetic DEM on the 30 m (1 arc-second) grid: two valleys draining south,
# separated by a north-south ridge down the middle. Each valley floor slopes
# toward the south edge, and the valley walls slope toward the floor.
ARCSEC = 1 / 3600.0
SIZE = 201  # cells per side
LAT0, LON0 = 38.5, -110.0  # centre of the DEM
WEST = LON0 - SIZE // 2 * ARCSEC - ARCSEC / 2
NORTH = LAT0 + SIZE // 2 * ARCSEC + ARCSEC / 2
LEFT_CHANNEL_COL = SIZE // 4
RIGHT_CHANNEL_COL = 3 * SIZE // 4
RIDGE_COL = SIZE // 2


def col_to_lon(col: int) -> float:
    return WEST + (col + 0.5) * ARCSEC


def row_to_lat(row: int) -> float:
    return NORTH - (row + 0.5) * ARCSEC


def make_dem() -> np.ndarray:
    rows, cols = np.mgrid[0:SIZE, 0:SIZE]
    channel = np.where(cols < RIDGE_COL, LEFT_CHANNEL_COL, RIGHT_CHANNEL_COL)
    across = np.abs(cols - channel) * 2.0  # walls rise away from the channel
    ridge = np.where(cols == RIDGE_COL, 500.0, 0.0)
    down = (SIZE - rows) * 1.0  # whole surface falls toward the south edge
    return (1000.0 + across + down + ridge).astype("float32")


@pytest.fixture
def dem_path(tmp_path):
    path = tmp_path / "dem.tif"
    with rasterio.open(
        path, "w", driver="GTiff", height=SIZE, width=SIZE, count=1,
        dtype="float32", crs="EPSG:4326", nodata=-9999,
        transform=from_origin(WEST, NORTH, ARCSEC, ARCSEC),
    ) as dst:
        dst.write(make_dem(), 1)
    return str(path)


@pytest.fixture
def offline_dem(monkeypatch, dem_path):
    """Make catchment.fetch_dem copy the synthetic DEM instead of calling USGS."""
    calls = []

    async def fake_fetch(bbox, dataset, path, log=None):
        calls.append((bbox, dataset, path))
        if not os.path.exists(path):
            with open(dem_path, "rb") as src, open(path, "wb") as dst:
                dst.write(src.read())
        return path

    monkeypatch.setattr(catchment, "fetch_dem", fake_fetch)
    return calls
