import asyncio
import json
import os

import catchment
from conftest import RIGHT_CHANNEL_COL, SIZE, col_to_lon, row_to_lat


class FakeWS:
    def __init__(self):
        self.sent = []

    async def send_str(self, msg):
        self.sent.append(msg)


class FakeRequest:
    content_type = "application/json"

    def __init__(self, data):
        self._data = data

    async def json(self):
        return self._data


def test_submit_end_to_end_offline(offline_dem, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    import sheed

    ws = FakeWS()
    monkeypatch.setitem(sheed.clients, "abcd", ws)
    lat, lon = row_to_lat(SIZE - 5), col_to_lon(RIGHT_CHANNEL_COL)
    req = FakeRequest({
        "coordinates": f"{lat},{lon}", "name": "", "expand_factor": "0.1",
        "client_id": "abcd", "dem": "USGS30m", "snap": 0, "snowpack": 0,
    })
    resp = asyncio.run(sheed.handle_submit(req))
    assert resp.status == 200
    body = json.loads(resp.text)
    assert body["catchment"]["area_km2"] > 0
    assert os.path.isfile(body["geojson"])
    assert os.path.isfile(body["kml"])
    with open(body["geojson"]) as f:
        fc = json.load(f)
    assert fc["features"][0]["properties"]["area_km2"] == body["catchment"]["area_km2"]
    assert any("Done!" in m for m in ws.sent)


def test_submit_rejects_bad_dem(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    import sheed
    req = FakeRequest({
        "coordinates": "38.5,-110", "name": "", "expand_factor": "0.05",
        "client_id": "abcd", "dem": "nope", "snap": 1,
    })
    resp = asyncio.run(sheed.handle_submit(req))
    assert resp.status == 400
