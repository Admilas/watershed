#!/usr/bin/env python3
from pysheds.grid import Grid
import numpy as np
import simplekml
import aiohttp
import asyncio
import logging
import geojson
import os
import uuid
from aiohttp import web
from typing import List, Union, cast
import urllib.parse
import json
import datetime
from shapely.geometry import shape as shapely_shape, Polygon
from dotenv import load_dotenv

import snowpack

load_dotenv()
THUNDERFOREST_API_KEY = os.environ.get("THUNDERFOREST_API_KEY", "")

SENTINEL_DAYS_RANGE = range(1, 31)
SENTINEL_ZOOM = 15

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S%z",
)

access_log = logging.getLogger("aiohttp.access")
access_log.setLevel(logging.INFO)
handler = logging.FileHandler("access.log")
formatter = logging.Formatter("%(asctime)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
handler.setFormatter(formatter)
access_log.addHandler(handler)
# access_log.propagate = False


def make_id() -> str:
    return str(uuid.uuid4())[:4]


DATASET_ARCSEC = {"USGS30m": 1.0, "USGS10m": 1.0 / 3.0}


class Watershed:
    def __init__(
        self,
        lat: float,
        lon: float,
        name: str,
        expand_factor: float,
        client_id,
        dem,
        snap,
        snowpack: bool = False,
        snowpack_layer: str = "tc",
        snowpack_compare_days: int = 0,
    ):
        self.id = client_id  # if client_id else make_id()
        self.outdir = "output"
        os.makedirs(self.outdir, exist_ok=True)

        self.lat = lat
        self.lon = lon
        self.expand_factor = expand_factor
        self.dem = dem
        self.snap = snap
        self.snowpack = snowpack
        self.snowpack_layer = snowpack_layer
        self.snowpack_compare_days = snowpack_compare_days
        self.snapped_x = None
        self.snapped_y = None

        if not name:
            self._generate_default_name()
        else:
            self.name = name

        self.generate_box()

    def _generate_default_name(self):
        self.name = f"{self.lat}_{self.lon}_{self.expand_factor}"

    def generate_box(self):
        self.min_x, self.min_y, self.max_x, self.max_y = [
            round(self.lon - self.expand_factor, 5),
            round(self.lat - self.expand_factor, 5),
            round(self.lon + self.expand_factor, 5),
            round(self.lat + self.expand_factor, 5),
        ]

    async def _log(self, msg):
        log_message = f"[{self.id}] {msg}"
        logging.info(log_message)
        await broadcast_message(f"log:{log_message}", self.id)

    async def work(self):
        self.dem_filename = await self.get_dem()
        self.catchment_shapes = await self.calculate_catchment()
        self.clipped = await self.clipping_check()

        if self.clipped:
            await self._log("Clipping detected, retrying with expand_factor=0.1")
            self.expand_factor = 0.1
            self.generate_box()
            self._generate_default_name()

            self.dem_filename = await self.get_dem()
            self.catchment_shapes = await self.calculate_catchment()
            self.clipped = await self.clipping_check()

        self.geojson = await self.export_geojson()
        self.kml = await self.export_kml()
        self.sentinels = await self.export_sentinels() if self.snowpack else []
        await self._log("Done!")

    async def get_dem(self) -> str:
        dataset = self.dem

        dem_filename = f"{self.outdir}/dem_{dataset}_{self.name}.tif"

        # Use cached DEM if possible
        if os.path.isfile(dem_filename):
            await self._log(f"Using cached DEM {dem_filename}")
            return dem_filename

        # Match the source's native arc-second grid so output cells are square
        # in degrees — pysheds' D8 flow direction assumes square pixels, and a
        # cos(lat) correction here gave non-square cells that snapped to the
        # wrong drainage.
        deg_per_cell = DATASET_ARCSEC[dataset] / 3600.0
        width_px = max(1, round((self.max_x - self.min_x) / deg_per_cell))
        height_px = max(1, round((self.max_y - self.min_y) / deg_per_cell))

        params = {
            "bbox": f"{self.min_x},{self.min_y},{self.max_x},{self.max_y}",
            "bboxSR": "4326",
            "imageSR": "4326",
            "size": f"{width_px},{height_px}",
            "format": "tiff",
            "pixelType": "F32",
            "noData": "-9999",
            "interpolation": "RSP_NearestNeighbor",
            "f": "image",
        }

        base_url = "https://elevation.nationalmap.gov/arcgis/rest/services/3DEPElevation/ImageServer/exportImage"

        await self._log(
            f"Fetching {dataset} DEM ({width_px}x{height_px} px) centered on {self.lat},{self.lon} from USGS 3DEP..."
        )

        async with aiohttp.ClientSession() as session:
            async with session.get(base_url, params=params) as response:
                if response.status == 200:
                    content = await response.read()
                    with open(dem_filename, "wb") as file:
                        file.write(content)
                    await self._log(f"{dem_filename} downloaded successfully.")
                else:
                    await self._log(f"Error: {response.status}. Request was {params}")
                    await self._log(await response.text())
                    raise
        return dem_filename

    async def calculate_catchment(self) -> List:
        if not self.dem_filename:
            raise

        await self._log("Preparing DEM")
        grid = Grid.from_raster(self.dem_filename)
        dem = grid.read_raster(self.dem_filename)

        pit_filled_dem = grid.fill_pits(dem)
        flooded_dem = grid.fill_depressions(pit_filled_dem)
        inflated_dem = grid.resolve_flats(flooded_dem)

        dirmap = (64, 128, 1, 2, 4, 8, 16, 32)

        await self._log("Calculating catchment")
        fdir = grid.flowdir(inflated_dem, dirmap=dirmap)
        acc = grid.accumulation(fdir, dirmap=dirmap)

        if self.snap:
            # Snap pour point to high accumulation cell
            self.snapped_x, self.snapped_y = grid.snap_to_mask(
                acc > 1000, (self.lon, self.lat)
            )
            catch = grid.catchment(
                x=self.snapped_x,
                y=self.snapped_y,
                fdir=fdir,
                dirmap=dirmap,
                xytype="coordinate",
            )
        else:
            catch = grid.catchment(
                x=self.lon, y=self.lat, fdir=fdir, dirmap=dirmap, xytype="coordinate"
            )

        grid.clip_to(catch)
        catch_view = grid.view(catch, dtype=np.uint8)

        shapes = [shape for shape in grid.polygonize(catch_view)]
        # assert len(shapes) == 1  # Can't have multiple watersheds

        return shapes

    async def clipping_check(self) -> bool:
        """
        Detect if the catchment polygon comes close to the edge of the DEM. If so,
        it's likely the catchment has been clipped and the DEM should be enlarged.
        """

        # Threshold scales with cell size: a polygonized vertex sitting on the
        # bbox edge can land up to one cell inward, so 2 cells is a robust
        # margin at any resolution.
        threshold = 2 * DATASET_ARCSEC[self.dem] / 3600.0

        shape = self.catchment_shapes[0][0]

        catchment_polygon = cast(Polygon, shapely_shape(shape))

        for x, y in catchment_polygon.exterior.coords:
            d = min(
                [
                    abs(self.max_x - x),
                    abs(self.min_x - x),
                    abs(self.max_y - y),
                    abs(self.min_y - y),
                ]
            )
            if d < threshold:
                await self._log(f"Clipping detected: {y},{x} {d}")
                return True
        return False

    async def export_geojson(self) -> str:
        if not self.catchment_shapes:
            raise

        features = []

        for shape in self.catchment_shapes:
            geo_polygon = geojson.Polygon([(shape[0]["coordinates"][0])])
            geo_polygon_feature = geojson.Feature(
                geometry=geo_polygon, properties={"name": "Example Polygon"}
            )
            features.append(geo_polygon_feature)

        filename = f"{self.outdir}/watershed - {self.name}.geojson"
        with open(f"{filename}", "w") as f:
            geojson.dump(geojson.FeatureCollection(features), f)
        await self._log(f'GeoJSON generated: "{filename}"')
        return filename

    async def _fetch_one_sentinel(self, days_ago: int) -> dict | None:
        timestamp = snowpack.days_ago_to_timestamp(days_ago)
        date = datetime.datetime.fromtimestamp(
            timestamp, tz=datetime.timezone.utc
        ).strftime("%Y-%m-%d")
        filename = f"{self.outdir}/sentinel_{self.snowpack_layer}-{timestamp}_{self.name}.png"
        try:
            await self._log(f"Fetching Sentinel imagery for {date} (-{days_ago}d)...")
            image = await snowpack.fetch_geojson(
                self.geojson, layer=self.snowpack_layer, timestamp=timestamp, zoom=SENTINEL_ZOOM
            )
            label = date
            if self.snowpack_layer == "ag":
                label = f"{date}  ·  snow {snowpack.snow_fraction(image) * 100:.0f}%"
            image = snowpack.add_date_bar(image, label)
            image.save(filename)
            await self._log(f'Sentinel image generated: "{filename}"')
            return {
                "days_ago": days_ago,
                "timestamp": timestamp,
                "date": date,
                "path": filename,
            }
        except Exception as e:
            await self._log(f"Sentinel fetch failed for {date}: {e}")
            return None

    async def export_sentinels(self) -> list[dict]:
        polygon = snowpack.load_geojson_polygon(self.geojson)
        west, south, east, north = snowpack.polygon_bbox(polygon)
        cx_lon, cy_lat = (west + east) / 2, (north + south) / 2
        cx_f, cy_f = snowpack.lonlat_to_tile(cx_lon, cy_lat, SENTINEL_ZOOM)
        cx, cy = int(cx_f), int(cy_f)

        days = list(SENTINEL_DAYS_RANGE)
        timestamps = [snowpack.days_ago_to_timestamp(d) for d in days]
        await self._log(
            f"Probing {len(days)} days for unique Sentinel snapshots..."
        )
        hashes = await snowpack.probe_tile_hashes(
            self.snowpack_layer, timestamps, SENTINEL_ZOOM, cx, cy
        )

        # Keep the smallest days_ago per unique hash (most recent label for the pass).
        earliest_for_hash: dict[str, int] = {}
        for d, ts in zip(days, timestamps):
            h = hashes.get(ts)
            if h is None:
                continue
            if h not in earliest_for_hash or d < earliest_for_hash[h]:
                earliest_for_hash[h] = d
        unique_days = sorted(earliest_for_hash.values())
        await self._log(
            f"Found {len(unique_days)} unique snapshots in last {len(days)} days"
        )

        offset = self.snowpack_compare_days
        fetch_days = list(unique_days)
        if offset > 0:
            fetch_days += [d + offset for d in unique_days]

        results = await asyncio.gather(
            *(self._fetch_one_sentinel(d) for d in fetch_days)
        )
        by_day = dict(zip(fetch_days, results))

        paired = []
        for d in unique_days:
            cur = by_day.get(d)
            if cur is None:
                continue
            if offset > 0:
                cur["comparison"] = by_day.get(d + offset)
            paired.append(cur)
        return paired

    async def export_kml(self) -> str:
        if not self.catchment_shapes:
            raise

        filename = f"{self.outdir}/watershed_-_{self.name}.kml"

        kml = simplekml.Kml()
        kml.newpoint(
            name=f"Requested Point - {self.name}",
            coords=[(self.lon, self.lat)],
        )

        if self.snapped_x and self.snapped_y:
            kml.newpoint(
                name=f"Snapped Point - {self.name}",
                coords=[(self.snapped_x, self.snapped_y)],
            )

        for shape in self.catchment_shapes:
            poly = kml.newpolygon(name=f"Watershed - {self.name}")
            poly.outerboundaryis = shape[0]["coordinates"][0]
            poly.style.linestyle.color = simplekml.Color.blue
            poly.style.linestyle.width = 1
            poly.style.polystyle.color = simplekml.Color.changealphaint(
                20, simplekml.Color.blue
            )

        kml.save(filename)
        await self._log(f'KML generated: "{filename}"')
        return filename


async def handle_index(request):
    with open("static/index.html") as f:
        index = f.read()
    index = index.replace("__THUNDERFOREST_API_KEY__", THUNDERFOREST_API_KEY)
    return web.Response(text=index, content_type="text/html")


async def handle_submit(request):
    if request.content_type == "application/json":
        data = await request.json()
    elif request.content_type == "application/x-www-form-urlencoded":
        data = await request.post()
    else:
        return web.Response(text="Unsupported content type", status=400)
    print(data)
    try:
        coordinates = data["coordinates"]
        lat, lon = map(float, coordinates.split(","))
        name = data["name"]
        expand_factor = float(data["expand_factor"])
        client_id = data["client_id"]
        dem = data["dem"]
        snap = data["snap"]
        snowpack_on = bool(int(data.get("snowpack", 0)))
        snowpack_layer = data.get("snowpack_layer", "tc")
        snowpack_compare_days = max(0, int(data.get("snowpack_compare_days", 0) or 0))

    except (KeyError, ValueError):
        return web.Response(text="Invalid input", status=400)

    watershed = Watershed(
        lat, lon, name, expand_factor, client_id, dem, snap,
        snowpack=snowpack_on, snowpack_layer=snowpack_layer,
        snowpack_compare_days=snowpack_compare_days,
    )
    await watershed.work()

    response_content = {}
    response_content["clipped"] = watershed.clipped
    response_content["dem"] = watershed.dem
    response_content["expand_factor"] = watershed.expand_factor
    response_content["geojson"] = watershed.geojson
    response_content["kml"] = watershed.kml
    response_content["sentinels"] = watershed.sentinels
    response_content["lat"] = watershed.lat
    response_content["lon"] = watershed.lon
    response_content["name"] = watershed.name

    return web.Response(text=json.dumps(response_content), content_type="text/json")


async def broadcast_message(message, client_id=None) -> None:
    if client_id:
        ws = clients[client_id]
        await ws.send_str(message)
    else:
        for ws in clients.values():
            await ws.send_str(message)


async def websocket_handler(request):
    ws = web.WebSocketResponse()
    await ws.prepare(request)

    client_id = make_id()
    clients[client_id] = ws

    await ws.send_str(f"client_id:{client_id}")

    try:
        async for msg in ws:
            if msg.type == aiohttp.WSMsgType.TEXT:
                if msg.data == "close":
                    await ws.close()
            elif msg.type == aiohttp.WSMsgType.ERROR:
                print(f"WebSocket connection closed with exception {ws.exception()}")

    finally:
        clients.pop(client_id)

    return ws


ROPEWIKI_API = "https://ropewiki.com/api.php"


async def handle_ropewiki_search(request):
    q = request.query.get("q", "").strip()
    if len(q) < 2:
        return web.json_response({"results": []})

    params = {
        "action": "opensearch",
        "search": q,
        "limit": "10",
        "namespace": "0",
        "format": "json",
    }
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(ROPEWIKI_API, params=params, timeout=aiohttp.ClientTimeout(total=5)) as resp:
                if resp.status != 200:
                    return web.json_response({"results": []})
                data = await resp.json()
    except Exception as e:
        logging.warning(f"Ropewiki search failed: {e}")
        return web.json_response({"results": []})

    titles = data[1] if len(data) > 1 else []
    urls = data[3] if len(data) > 3 else []
    results = [{"title": t, "url": u} for t, u in zip(titles, urls)]
    return web.json_response({"results": results})


async def handle_ropewiki_coords(request):
    title = request.query.get("title", "").strip()
    if not title:
        return web.json_response({"error": "title required"}, status=400)

    params = {
        "action": "ask",
        "query": f"[[{title}]]|?Has_coordinates",
        "format": "json",
    }
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(ROPEWIKI_API, params=params, timeout=aiohttp.ClientTimeout(total=5)) as resp:
                if resp.status != 200:
                    return web.json_response({"error": "upstream error"}, status=502)
                data = await resp.json()
    except Exception as e:
        logging.warning(f"Ropewiki coords failed: {e}")
        return web.json_response({"error": "upstream error"}, status=502)

    results = data.get("query", {}).get("results", {})
    page = next(iter(results.values()), None) if results else None
    printouts = page.get("printouts", {}) if page else {}
    coords = printouts.get("Has coordinates") or printouts.get("Has_coordinates") or []
    if not coords:
        return web.json_response({"error": "no coordinates"}, status=404)

    point = coords[0]
    lat = point.get("lat")
    lon = point.get("lon")
    if lat is None or lon is None:
        return web.json_response({"error": "no coordinates"}, status=404)
    return web.json_response({"lat": lat, "lon": lon})


os.makedirs("output/", exist_ok=True)
os.makedirs("static/", exist_ok=True)

print("Directories made")

async def close_websockets(app):
    for ws in list(clients.values()):
        await ws.close(code=1001, message=b"Server shutting down")
    clients.clear()


app = web.Application()
app.router.add_get("/", handle_index)
app.router.add_post("/", handle_submit)
app.router.add_static(prefix="/output", path="output/", show_index=True)
app.router.add_static(prefix="/static", path="static/", show_index=False)
app.router.add_get("/ws", websocket_handler)
app.router.add_get("/api/ropewiki/search", handle_ropewiki_search)
app.router.add_get("/api/ropewiki/coords", handle_ropewiki_coords)
app.on_shutdown.append(close_websockets)

clients: dict[str, web.WebSocketResponse] = {}

if __name__ == "__main__":
    print("Starting web...")
    web.run_app(app, host=["0.0.0.0", "::"], port=8080, access_log=access_log)
