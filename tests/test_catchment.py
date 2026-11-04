import asyncio

import pytest
from shapely.geometry import Point

import catchment
from conftest import (
    ARCSEC, LEFT_CHANNEL_COL, RIDGE_COL, RIGHT_CHANNEL_COL, SIZE,
    col_to_lon, row_to_lat,
)


def outlet(col):
    # A few rows up from the south edge, on the channel
    return row_to_lat(SIZE - 5), col_to_lon(col)


def test_left_valley_stays_west_of_ridge(dem_path):
    lat, lon = outlet(LEFT_CHANNEL_COL)
    polygons, snapped, elevations, cell = catchment.compute_catchment(dem_path, lon, lat, snap=False)
    poly = max(polygons, key=lambda p: p.area)
    ridge_lon = col_to_lon(RIDGE_COL)
    assert poly.bounds[2] <= ridge_lon + ARCSEC
    assert poly.buffer(ARCSEC).contains(Point(col_to_lon(LEFT_CHANNEL_COL), row_to_lat(SIZE // 2)))
    assert not poly.contains(Point(col_to_lon(RIGHT_CHANNEL_COL), row_to_lat(SIZE // 2)))
    assert snapped is None
    assert cell == pytest.approx(ARCSEC)
    # roughly half the DEM minus the ridge and rows below the outlet
    assert 0.4 * SIZE * SIZE < elevations.size < 0.5 * SIZE * SIZE


def test_snap_moves_point_onto_channel(dem_path):
    # Well away from the DEM edge, whose bottom row collects flow too
    lat, lon = row_to_lat(SIZE - 40), col_to_lon(LEFT_CHANNEL_COL)
    off_channel_lon = lon + 6 * ARCSEC
    _, snapped, _, _ = catchment.compute_catchment(dem_path, off_channel_lon, lat, snap=True, snap_threshold=500)
    assert snapped is not None
    # pysheds reports the snapped cell by its corner, so allow one cell
    assert snapped[0] == pytest.approx(lon, abs=ARCSEC)


def test_area_matches_cell_count():
    # one 1-arcsec cell at the equator is ~30.9 m square
    assert catchment.catchment_area_km2(1, ARCSEC, 0.0) == pytest.approx(0.000957, rel=1e-2)
    # shrinks with cos(lat)
    assert catchment.catchment_area_km2(1000, ARCSEC, 60.0) == pytest.approx(
        catchment.catchment_area_km2(1000, ARCSEC, 0.0) / 2, rel=1e-6
    )


def test_is_clipped():
    from shapely.geometry import box
    bbox = catchment.BBox(0.0, 0.0, 1.0, 1.0)
    assert catchment.is_clipped(box(0.0, 0.2, 0.5, 0.5), bbox, "USGS30m")
    assert not catchment.is_clipped(box(0.2, 0.2, 0.5, 0.5), bbox, "USGS30m")


def test_dem_request_is_square_cells():
    bbox = catchment.BBox.around(38.5, -110.0, 0.05)
    params = catchment.dem_request_params(bbox, "USGS10m")
    assert params["size"] == "1080,1080"
    assert params["bbox"] == "-110.05,38.45,-109.95,38.55"


def test_delineate_stats_and_cache(offline_dem, tmp_path):
    lat, lon = outlet(RIGHT_CHANNEL_COL)
    # expand_factor large enough that the synthetic DEM is the whole bbox
    result = asyncio.run(catchment.delineate(
        lat, lon, dataset="USGS30m", snap=False, expand_factor=0.1, workdir=str(tmp_path),
    ))
    assert len(offline_dem) == 1
    assert result.area_km2 > 0
    assert result.elev_min_m < result.elev_mean_m < result.elev_max_m
    assert result.polygon.bounds[0] >= col_to_lon(RIDGE_COL) - ARCSEC

    again = asyncio.run(catchment.delineate(
        lat, lon, dataset="USGS30m", snap=False, expand_factor=0.1, workdir=str(tmp_path),
    ))
    assert len(offline_dem) == 1, "second call should come from the cache"
    assert again.stats() == result.stats()
    assert again.polygon.equals_exact(result.polygon, 1e-9)


def test_delineate_rejects_unknown_dataset(tmp_path):
    with pytest.raises(ValueError):
        asyncio.run(catchment.delineate(38.5, -110.0, dataset="SRTM", workdir=str(tmp_path)))
