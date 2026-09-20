"""
geocoding.py
------------
Mengisi kolom latitude/longitude dari kolom `address` TANPA Google Maps API key.

Strategi berlapis:
1. Cache lokal (data/geocode_cache.json) -> instan, tidak perlu internet.
2. Nominatim OpenStreetMap -> gratis, tanpa API key, butuh internet.
3. Centroid kecamatan bawaan -> fallback bila alamat tidak ditemukan.

Pakai dari terminal:
    python src/geocoding.py data/Data_Operasional_Rute.xlsx
"""
from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pandas as pd

CACHE_PATH = Path(__file__).resolve().parents[1] / "data" / "geocode_cache.json"
NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"
USER_AGENT = "ai-route-optimization/1.0 (kontak: ganti-dengan-email-anda)"


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
        CACHE_PATH.write_text(json.dumps(cache, indent=2, ensure_ascii=False), encoding="utf-8")
    except OSError:
        pass  # lingkungan read-only (mis. beberapa layanan cloud) -> lanjut tanpa cache disk

# Titik tengah ruas jalan di Medan. Dipakai bila geocoding online gagal atau
# dijalankan dengan --offline. Kunci "jalan|kecamatan" dipakai lebih dulu untuk
# jalan panjang yang melintasi beberapa kecamatan.
JALAN_CENTROID = {
    "sisingamangaraja|medan amplas": (3.5410, 98.7080),
    "sisingamangaraja": (3.5830, 98.6900),
    "yos sudarso|medan belawan": (3.7680, 98.6860),
    "yos sudarso|medan labuhan": (3.7255, 98.6841),
    "yos sudarso": (3.6180, 98.6760),
    "gatot subroto": (3.5930, 98.6600),
    "kapten muslim": (3.6030, 98.6480),
    "setia budi": (3.5760, 98.6320),
    "krakatau": (3.6120, 98.6890),
    "denai": (3.5640, 98.7100),
    "ar hakim": (3.5780, 98.6950),
    "brigjen katamso": (3.5760, 98.6830),
    "hm joni": (3.5720, 98.7000),
    "halat": (3.5740, 98.6930),
    "pancing": (3.6100, 98.7200),
    "willem iskandar": (3.6180, 98.7130),
    "marelan raya": (3.7120, 98.6470),
    "veteran|medan belawan": (3.7680, 98.6860),
    "veteran": (3.5940, 98.6721),
    "jamin ginting": (3.5500, 98.6480),
    "dr mansyur": (3.5679, 98.6412),
    "iskandar muda": (3.5820, 98.6560),
    "adam malik": (3.6000, 98.6650),
    "putri hijau": (3.5980, 98.6780),
    "sutomo": (3.5931, 98.6912),
    "letda sujono": (3.6018, 98.7202),
    "sunggal": (3.5851, 98.6142),
    "karya wisata": (3.5308, 98.6674),
    "thamrin": (3.5820, 98.6903),
    "asia": (3.5804, 98.6996),
    "pelabuhan raya": (3.7745, 98.6852),
}

# Titik tengah kecamatan di Medan, lapisan terakhir bila nama jalan tidak dikenali.
KECAMATAN_CENTROID = {
    "medan petisah": (3.5893, 98.6650), "medan helvetia": (3.6068, 98.6345),
    "medan selayang": (3.5622, 98.6296), "medan timur": (3.6135, 98.6903),
    "medan denai": (3.5663, 98.7118), "medan amplas": (3.5471, 98.7135),
    "medan area": (3.5780, 98.6950), "medan maimun": (3.5772, 98.6835),
    "medan tembung": (3.6100, 98.7200), "medan barat": (3.5940, 98.6721),
    "medan marelan": (3.7120, 98.6470), "medan belawan": (3.7745, 98.6852),
    "medan kota": (3.5820, 98.6903), "medan baru": (3.5623, 98.6553),
    "medan polonia": (3.5729, 98.6668), "medan johor": (3.5308, 98.6674),
    "medan sunggal": (3.5975, 98.6288), "medan labuhan": (3.7255, 98.6841),
    "medan perjuangan": (3.6135, 98.6903), "medan tuntungan": (3.5262, 98.6193),
    "medan": (3.5952, 98.6722),
}

def _fallback_lokal(address: str):
    """Mencari koordinat dari tabel bawaan: jalan+kecamatan, lalu jalan, lalu kecamatan."""
    low = address.lower()
    kecamatan = next((k for k in KECAMATAN_CENTROID if k != "medan" and k in low), None)

    if kecamatan:
        for jalan, coord in JALAN_CENTROID.items():
            if "|" in jalan:
                nama, kec = jalan.split("|")
                if nama in low and kec == kecamatan:
                    return coord, f"Ruas jalan ({nama.title()}, {kec.title()})", "Fallback"

    for jalan, coord in JALAN_CENTROID.items():
        if "|" not in jalan and jalan in low:
            return coord, f"Ruas jalan ({jalan.title()})", "Fallback"

    if kecamatan:
        return KECAMATAN_CENTROID[kecamatan], f"Centroid kecamatan ({kecamatan.title()})", "Fallback"

    if "medan" in low:
        return KECAMATAN_CENTROID["medan"], "Centroid kota (Medan)", "Fallback"

    return None


def _query_nominatim(address: str, timeout: int = 10):
    """Satu permintaan ke Nominatim. Mengembalikan (lat, lon) atau None."""
    # Nomor rumah jarang ada di OSM Indonesia, jadi dicoba dua bentuk query.
    kandidat = [address, re.sub(r"No\.?\s*\d+,?\s*", "", address, flags=re.I)]
    for q in kandidat:
        params = urlencode({"q": q, "format": "json", "limit": 1, "countrycodes": "id"})
        req = Request(f"{NOMINATIM_URL}?{params}", headers={"User-Agent": USER_AGENT})
        try:
            with urlopen(req, timeout=timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except Exception:
            return None
        if data:
            return float(data[0]["lat"]), float(data[0]["lon"])
        time.sleep(1.1)  # kebijakan Nominatim: maksimal 1 permintaan per detik
    return None


def geocode_address(address: str, use_online: bool = True, cache: dict | None = None):
    """Mengembalikan dict dengan latitude, longitude, source, status."""
    cache = _load_cache() if cache is None else cache
    key = address.strip().lower()

    if key in cache:
        c = cache[key]
        return {"latitude": c["lat"], "longitude": c["lon"],
                "source": c.get("source", "Cache"), "status": "Cache"}

    if use_online:
        hit = _query_nominatim(address)
        time.sleep(1.1)
        if hit:
            cache[key] = {"lat": hit[0], "lon": hit[1], "source": "Nominatim OSM"}
            _save_cache(cache)
            return {"latitude": hit[0], "longitude": hit[1],
                    "source": "Nominatim OSM", "status": "Berhasil"}

    fb = _fallback_lokal(address)
    if fb:
        (lat, lon), src, st = fb
        return {"latitude": lat, "longitude": lon, "source": src, "status": st}

    return {"latitude": None, "longitude": None, "source": "-", "status": "Gagal"}


def geocode_dataframe(df: pd.DataFrame, address_col: str = "address",
                      use_online: bool = True, overwrite: bool = False) -> pd.DataFrame:
    """Mengisi kolom latitude/longitude pada sebuah DataFrame."""
    df = df.copy()
    for col in ("latitude", "longitude", "geocode_source", "geocode_status"):
        if col not in df.columns:
            df[col] = None

    cache = _load_cache()
    for i, row in df.iterrows():
        sudah_ada = pd.notna(row.get("latitude")) and pd.notna(row.get("longitude"))
        if sudah_ada and not overwrite:
            continue
        hasil = geocode_address(str(row[address_col]), use_online=use_online, cache=cache)
        df.at[i, "latitude"] = hasil["latitude"]
        df.at[i, "longitude"] = hasil["longitude"]
        df.at[i, "geocode_source"] = hasil["source"]
        df.at[i, "geocode_status"] = hasil["status"]
    _save_cache(cache)
    return df


def geocode_missing(df: pd.DataFrame, address_col: str = "address",
                    use_online: bool = True, progress_cb=None) -> pd.DataFrame:
    """Mengisi HANYA baris yang latitude/longitude-nya masih kosong.

    Dipakai oleh tombol "Isi koordinat otomatis" di aplikasi. progress_cb(i, n, label)
    dipanggil setelah tiap baris selesai, agar UI bisa menampilkan progress bar.
    """
    df = df.copy()
    for col in ("latitude", "longitude", "geocode_source", "geocode_status"):
        if col not in df.columns:
            df[col] = None

    perlu = df[df.latitude.isna() | df.longitude.isna()]
    cache = _load_cache()
    for n, (i, row) in enumerate(perlu.iterrows(), start=1):
        alamat = str(row.get(address_col, "")).strip()
        if not alamat or alamat.lower() == "nan":
            df.at[i, "geocode_status"] = "Gagal (alamat kosong)"
        else:
            hasil = geocode_address(alamat, use_online=use_online, cache=cache)
            df.at[i, "latitude"] = hasil["latitude"]
            df.at[i, "longitude"] = hasil["longitude"]
            df.at[i, "geocode_source"] = hasil["source"]
            df.at[i, "geocode_status"] = hasil["status"]
        if progress_cb:
            progress_cb(n, len(perlu), alamat)
    _save_cache(cache)
    return df


def geocode_workbook(path: str, use_online: bool = True, overwrite: bool = False) -> None:
    """Memperbarui sheet Customers dan Depot pada file Excel di tempat."""
    sheets = pd.read_excel(path, sheet_name=None)
    for nama in ("Customers", "Depot"):
        if nama in sheets:
            sheets[nama] = geocode_dataframe(sheets[nama], use_online=use_online,
                                             overwrite=overwrite)
            print(f"[{nama}] selesai, {sheets[nama]['latitude'].notna().sum()} titik terisi")
    with pd.ExcelWriter(path, engine="openpyxl") as xl:
        for nama, df in sheets.items():
            df.to_excel(xl, sheet_name=nama, index=False)
    print("Tersimpan:", path)


if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else "data/Data_Operasional_Rute.xlsx"
    online = "--offline" not in sys.argv
    geocode_workbook(target, use_online=online, overwrite="--overwrite" in sys.argv)
