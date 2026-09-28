"""
road_routing.py
----------------
Rute mengikuti jalan sungguhan memakai OSRM (Open Source Routing Machine),
berbasis data OpenStreetMap -- SAMA seperti sumber data geocoding.py, jadi
tetap tidak perlu Google Maps API key.

Dipakai untuk DUA hal:
  1. Matriks jarak antar semua titik (menggantikan haversine x faktor jarak
     jalan) lewat OSRM Table Service -> SATU permintaan untuk seluruh matriks
     sekaligus, dipanggil sekali di awal lalu dipakai ulang oleh semua kandidat
     algoritma (tidak boros permintaan meski ada 5 kandidat).
  2. Bentuk polyline rute sesungguhnya di peta (mengikuti jalan, bukan garis
     lurus) lewat OSRM Route Service -> SATU permintaan per rute kendaraan
     terpilih (bukan per kandidat, supaya hemat).

Server publik OSRM (router.project-osrm.org) gratis, tanpa API key, tapi:
  - dibatasi untuk pemakaian wajar (bukan produksi skala besar/komersial)
  - hanya melayani mode berkendara ("driving")
  - Table Service dibatasi maksimal 100 titik sekaligus di server publik ini

Bila server tidak bisa dihubungi (tidak ada internet, server sibuk, atau titik
> 100), otomatis jatuh ke perhitungan garis lurus x faktor jarak jalan
(perilaku lama route_optimization.py), supaya aplikasi tetap bisa jalan.

Cache lokal (data/road_cache.json) menyimpan hasil permintaan supaya rerun
Streamlit tidak berulang kali memanggil OSRM untuk data yang persis sama.
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen

CACHE_PATH = Path(__file__).resolve().parents[1] / "data" / "road_cache.json"
OSRM_BASE = "https://router.project-osrm.org"
EARTH_R = 6371.0
MAKS_TITIK_TABLE = 100  # batas wajar server demo publik OSRM untuk satu permintaan Table


def _load_cache() -> dict:
    try:
        if CACHE_PATH.exists():
            return json.loads(CACHE_PATH.read_text(encoding="utf-8"))
    except OSError:
        pass
    return {}


def _save_cache(cache: dict) -> None:
    try:
        CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        CACHE_PATH.write_text(json.dumps(cache), encoding="utf-8")
    except OSError:
        pass  # lingkungan read-only -> lanjut tanpa cache disk


def _haversine_km(lat1, lon1, lat2, lon2) -> float:
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * EARTH_R * math.asin(math.sqrt(a))


def _get(url: str, timeout: int = 20):
    try:
        with urlopen(url, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except (URLError, TimeoutError, ValueError, OSError):
        return None


def _fallback_matriks(lat: list[float], lon: list[float], detour_factor: float) -> list[list[float]]:
    n = len(lat)
    mat = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            km = round(_haversine_km(lat[i], lon[i], lat[j], lon[j]) * detour_factor, 3)
            mat[i][j] = mat[j][i] = km
    return mat


def matriks_jalan(lat: list[float], lon: list[float], detour_factor: float = 1.32,
                  use_online: bool = True) -> tuple[list[list[float]], str]:
    """Mengembalikan (matriks_jarak_km, sumber) untuk seluruh titik sekaligus.

    sumber bernilai "OSRM", "OSRM (cache)", atau "Fallback (garis lurus x faktor
    jarak jalan)" -- tampilkan ini ke pengguna supaya jelas dari mana angkanya.
    """
    n = len(lat)
    if use_online and 1 < n <= MAKS_TITIK_TABLE:
        cache = _load_cache()
        kunci = "table|" + "|".join(f"{la:.5f},{lo:.5f}" for la, lo in zip(lat, lon))
        if kunci in cache:
            return cache[kunci], "OSRM (cache)"

        koord = ";".join(f"{lo:.6f},{la:.6f}" for la, lo in zip(lat, lon))
        data = _get(f"{OSRM_BASE}/table/v1/driving/{koord}?annotations=distance")
        if data and data.get("code") == "Ok":
            jarak_m = data.get("distances")
            if jarak_m and all(v is not None for row in jarak_m for v in row):
                jarak_km = [[round(v / 1000, 3) for v in row] for row in jarak_m]
                cache[kunci] = jarak_km
                _save_cache(cache)
                return jarak_km, "OSRM"

    return _fallback_matriks(lat, lon, detour_factor), "Fallback (garis lurus x faktor jarak jalan)"


def polyline_rute(urutan_latlon: list[tuple[float, float]],
                  use_online: bool = True) -> tuple[list[list[float]], str]:
    """Mengembalikan (daftar koordinat [lon, lat] mengikuti jalan, sumber) untuk SATU
    rute lengkap (depot -> ...pelanggan... -> depot). Jatuh ke garis lurus antar
    titik (perilaku lama) bila OSRM gagal/offline."""
    if use_online and len(urutan_latlon) >= 2:
        cache = _load_cache()
        kunci = "route|" + "|".join(f"{la:.5f},{lo:.5f}" for la, lo in urutan_latlon)
        if kunci in cache:
            return cache[kunci], "OSRM (cache)"

        koord = ";".join(f"{lo:.6f},{la:.6f}" for la, lo in urutan_latlon)
        data = _get(f"{OSRM_BASE}/route/v1/driving/{koord}?overview=full&geometries=geojson")
        if data and data.get("code") == "Ok" and data.get("routes"):
            geom = data["routes"][0]["geometry"]["coordinates"]
            cache[kunci] = geom
            _save_cache(cache)
            time.sleep(0.2)  # jeda sopan antar-permintaan ke server publik
            return geom, "OSRM"

    return [[lo, la] for la, lo in urutan_latlon], "Fallback (garis lurus)"