#!/usr/bin/env python3
import argparse
import asyncio
import hashlib
import io
import json
import math
import os
import sys
import time

import aiohttp
import numpy as np
from dotenv import load_dotenv
from PIL import Image, ImageDraw, ImageFont


load_dotenv()

TILE_URL_TEMPLATE = "https://caltopo.com/tile/sentinel_{layer}-{timestamp}/{z}/{x}/{y}.png"
SECONDS_PER_DAY = 86400
LAYERS = ("tc", "fc", "ag", "burn")
DEFAULT_LAYER = "tc"
TILE_SIZE = 256
DEFAULT_ZOOM = 14
RADIUS_KM = 2.0
USER_AGENT = "watershed-snowpack/1.0"
CONCURRENCY = 8
CALTOPO_COOKIE = os.environ.get("CALTOPO_COOKIE")


def lonlat_to_tile(lon: float, lat: float, zoom: int) -> tuple[float, float]:
    n = 2 ** zoom
    x = (lon + 180.0) / 360.0 * n
    lat_rad = math.radians(lat)
    y = (1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n
    return x, y


def lonlat_to_pixel(
    lon: float, lat: float, zoom: int, west: float, north: float
) -> tuple[float, float]:
    x_tile, y_tile = lonlat_to_tile(lon, lat, zoom)
    x_nw, y_nw = lonlat_to_tile(west, north, zoom)
    return (x_tile - x_nw) * TILE_SIZE, (y_tile - y_nw) * TILE_SIZE


def bbox_around_point(
    lat: float, lon: float, radius_km: float
) -> tuple[float, float, float, float]:
    dlat = radius_km / 111.32
    dlon = radius_km / (111.32 * math.cos(math.radians(lat)))
    return lon - dlon, lat - dlat, lon + dlon, lat + dlat


def load_geojson_polygon(path: str) -> list[tuple[float, float]]:
    with open(path) as f:
        data = json.load(f)
    if data["type"] == "FeatureCollection":
        geom = data["features"][0]["geometry"]
    elif data["type"] == "Feature":
        geom = data["geometry"]
    else:
        geom = data
    if geom["type"] == "Polygon":
        return geom["coordinates"][0]
    if geom["type"] == "MultiPolygon":
        return geom["coordinates"][0][0]
    raise ValueError(f"unsupported geometry type: {geom['type']}")


def polygon_bbox(
    coords: list[tuple[float, float]],
) -> tuple[float, float, float, float]:
    lons = [c[0] for c in coords]
    lats = [c[1] for c in coords]
    return min(lons), min(lats), max(lons), max(lats)


async def fetch_tile(
    session: aiohttp.ClientSession,
    sem: asyncio.Semaphore,
    layer: str,
    timestamp: int,
    z: int,
    x: int,
    y: int,
) -> tuple[int, int, bytes]:
    url = TILE_URL_TEMPLATE.format(layer=layer, timestamp=timestamp, z=z, x=x, y=y)
    async with sem, session.get(url) as resp:
        resp.raise_for_status()
        return x, y, await resp.read()


def stitch(
    tiles: list[tuple[int, int, bytes]],
    x_min: int,
    y_min: int,
    cols: int,
    rows: int,
) -> Image.Image:
    mosaic = Image.new("RGB", (cols * TILE_SIZE, rows * TILE_SIZE))
    for x, y, data in tiles:
        tile = Image.open(io.BytesIO(data))
        mosaic.paste(tile, ((x - x_min) * TILE_SIZE, (y - y_min) * TILE_SIZE))
    return mosaic


async def fetch_bbox(
    west: float,
    south: float,
    east: float,
    north: float,
    layer: str,
    timestamp: int,
    zoom: int,
) -> Image.Image:
    x_nw, y_nw = lonlat_to_tile(west, north, zoom)
    x_se, y_se = lonlat_to_tile(east, south, zoom)
    x_min, x_max = math.floor(x_nw), math.floor(x_se)
    y_min, y_max = math.floor(y_nw), math.floor(y_se)
    cols, rows = x_max - x_min + 1, y_max - y_min + 1

    headers = {"User-Agent": USER_AGENT}
    if CALTOPO_COOKIE:
        headers["Cookie"] = CALTOPO_COOKIE
    sem = asyncio.Semaphore(CONCURRENCY)
    async with aiohttp.ClientSession(headers=headers) as session:
        tiles = await asyncio.gather(
            *(
                fetch_tile(session, sem, layer, timestamp, zoom, x, y)
                for x in range(x_min, x_max + 1)
                for y in range(y_min, y_max + 1)
            )
        )

    mosaic = stitch(tiles, x_min, y_min, cols, rows)
    return mosaic.crop(
        (
            (x_nw - x_min) * TILE_SIZE,
            (y_nw - y_min) * TILE_SIZE,
            (x_se - x_min) * TILE_SIZE,
            (y_se - y_min) * TILE_SIZE,
        )
    )


def mask_to_polygon(
    image: Image.Image,
    polygon: list[tuple[float, float]],
    zoom: int,
    west: float,
    north: float,
) -> Image.Image:
    pixels = [lonlat_to_pixel(lon, lat, zoom, west, north) for lon, lat in polygon]
    mask = Image.new("L", image.size, 0)
    ImageDraw.Draw(mask).polygon(pixels, fill=255)
    rgba = image.convert("RGBA")
    rgba.putalpha(mask)
    return rgba


def snow_fraction(image: Image.Image) -> float:
    """Fraction of polygon-masked pixels that look like snow in the 'ag' composite.
    Snow renders as bright teal/cyan because CalTopo mixes NDSI into the band stack.
    Restricted to alpha>0 when present so out-of-watershed pixels don't dilute the count."""
    hsv = np.array(image.convert("HSV"))
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    snow = (h >= 110) & (h <= 145) & (s >= 40) & (v >= 100)
    if image.mode == "RGBA":
        valid = np.array(image)[..., 3] > 0
        total = int(valid.sum())
        return float((snow & valid).sum()) / total if total else 0.0
    return float(snow.mean())


def add_date_bar(image: Image.Image, date: str) -> Image.Image:
    """Return image with a black bar along the bottom showing date in white text."""
    bar_height = max(24, round(image.height * 0.04))
    font_size = round(bar_height * 0.6)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", font_size)
    except OSError:
        font = ImageFont.load_default()

    has_alpha = image.mode == "RGBA"
    mode = "RGBA" if has_alpha else "RGB"
    black = (0, 0, 0, 255) if has_alpha else (0, 0, 0)
    white = (255, 255, 255, 255) if has_alpha else (255, 255, 255)

    canvas = Image.new(mode, (image.width, image.height + bar_height), black)
    canvas.paste(image, (0, 0))

    draw = ImageDraw.Draw(canvas)
    left, top, right, bottom = draw.textbbox((0, 0), date, font=font)
    x = (image.width - (right - left)) // 2 - left
    y = image.height + (bar_height - (bottom - top)) // 2 - top
    draw.text((x, y), date, fill=white, font=font)
    return canvas


async def fetch_point(
    lat: float,
    lon: float,
    layer: str,
    timestamp: int,
    zoom: int,
    radius_km: float = RADIUS_KM,
) -> Image.Image:
    bbox = bbox_around_point(lat, lon, radius_km)
    return await fetch_bbox(*bbox, layer=layer, timestamp=timestamp, zoom=zoom)


async def fetch_geojson(
    path: str, layer: str, timestamp: int, zoom: int
) -> Image.Image:
    polygon = load_geojson_polygon(path)
    west, south, east, north = polygon_bbox(polygon)
    image = await fetch_bbox(
        west, south, east, north, layer=layer, timestamp=timestamp, zoom=zoom
    )
    return mask_to_polygon(image, polygon, zoom, west, north)


def days_ago_to_timestamp(days_ago: int) -> int:
    return int(time.time()) - days_ago * SECONDS_PER_DAY


async def probe_tile_hashes(
    layer: str, timestamps: list[int], zoom: int, x: int, y: int
) -> dict[int, str | None]:
    """Fetch one tile per timestamp and return {timestamp: md5_hex} (None on failure).
    Useful for deduping which timestamps resolve to the same Sentinel pass."""
    headers = {"User-Agent": USER_AGENT}
    if CALTOPO_COOKIE:
        headers["Cookie"] = CALTOPO_COOKIE
    sem = asyncio.Semaphore(CONCURRENCY)
    async with aiohttp.ClientSession(headers=headers) as session:
        async def probe(ts: int) -> tuple[int, str | None]:
            try:
                _, _, data = await fetch_tile(session, sem, layer, ts, zoom, x, y)
                return ts, hashlib.md5(data).hexdigest()
            except Exception:
                return ts, None

        return dict(await asyncio.gather(*(probe(ts) for ts in timestamps)))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fetch Sentinel imagery tiles from CalTopo and stitch into a PNG. "
        f"Point mode covers a ~{RADIUS_KM*2:g}km square; geojson mode covers the polygon."
    )
    parser.add_argument("lat", type=float, nargs="?", help="Center latitude (WGS84)")
    parser.add_argument("lon", type=float, nargs="?", help="Center longitude (WGS84)")
    parser.add_argument("--geojson", help="Polygon geojson file to cover (alternative to lat/lon)")
    parser.add_argument("--zoom", type=int, default=DEFAULT_ZOOM, help=f"XYZ zoom (default {DEFAULT_ZOOM})")
    parser.add_argument(
        "--layer",
        choices=LAYERS,
        default=DEFAULT_LAYER,
        help=f"Sentinel band combination (default {DEFAULT_LAYER}): "
        "tc=true color, fc=false color, ag=agriculture, burn=burn scar",
    )
    parser.add_argument(
        "--days-ago",
        type=int,
        default=0,
        dest="days_ago",
        help="How many days back from now (default 0 = today). "
        "Resolved to a Unix timestamp; CalTopo returns the closest available capture.",
    )
    parser.add_argument("--output", default=None, help="Output PNG path")
    args = parser.parse_args()

    point_given = args.lat is not None and args.lon is not None
    if args.geojson and point_given:
        sys.exit("error: pass either lat/lon or --geojson, not both")
    if not args.geojson and not point_given:
        sys.exit("error: pass either lat/lon or --geojson")

    timestamp = days_ago_to_timestamp(args.days_ago)
    tag = f"{args.layer}-{timestamp}"
    if args.geojson:
        default_name = os.path.splitext(os.path.basename(args.geojson))[0]
        output = args.output or f"output/sentinel_{tag}_{default_name}.png"
        coro = fetch_geojson(
            args.geojson, layer=args.layer, timestamp=timestamp, zoom=args.zoom
        )
    else:
        output = args.output or f"output/sentinel_{tag}_{args.lat}_{args.lon}.png"
        coro = fetch_point(
            args.lat, args.lon, layer=args.layer, timestamp=timestamp, zoom=args.zoom
        )

    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
    image = asyncio.run(coro)
    image.save(output)
    print(f"wrote {output} ({image.size[0]}x{image.size[1]})")


if __name__ == "__main__":
    main()
