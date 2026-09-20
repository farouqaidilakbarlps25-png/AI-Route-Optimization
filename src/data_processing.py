"""
data_processing.py
------------------
Memuat file Data_Operasional_Rute.xlsx, membersihkan tipe data, dan memvalidasinya
sebelum masuk ke tahap optimasi rute.

Pemakaian:
    from src.data_processing import load_data, validate_data
    data = load_data("data/Data_Operasional_Rute.xlsx")
    errors, warnings = validate_data(data)
"""
from __future__ import annotations

import io
import numbers
import re
from pathlib import Path

import pandas as pd

DEFAULT_PARAMS = {
    "detour_factor": 1.32,
    "traffic_factor": 1.18,
    "harga_solar_idr_per_liter": 6800,
    "penalti_keterlambatan_per_menit": 15000,
    "toleransi_tiba_awal_menit": 20,
    "emisi_co2_solar_per_liter": 2.68,
    "savings_lambda": 1.0,
    "max_iterasi_2opt": 200,
    "pusat_peta_lat": 3.5952,
    "pusat_peta_lon": 98.6722,
}

REQUIRED = {
    "Customers": ["customer_id", "customer_name", "address", "demand_kg",
                  "time_window_start", "time_window_end", "service_time_min"],
    "Vehicle": ["vehicle_id", "home_depot_id", "capacity_kg", "avg_speed_kmh",
                "variable_cost_per_km_idr", "fixed_cost_per_trip_idr",
                "shift_start", "shift_end", "max_stops_per_route"],
    "Depot": ["depot_id", "depot_name", "address", "open_time", "close_time"],
}


# Nama kolom alternatif yang otomatis dipetakan ke nama baku (huruf kecil, spasi -> "_").
# Bila data Anda memakai nama lain, cukup tambahkan pasangannya di sini.
ALIAS_KOLOM = {
    "Customers": {},
    "Vehicle": {
        "depot_id": "home_depot_id", "home_depot": "home_depot_id",
        "speed_kmph": "avg_speed_kmh", "speed_kmh": "avg_speed_kmh",
        "avg_speed_kmph": "avg_speed_kmh",
        "cost_per_km_idr": "variable_cost_per_km_idr",
        "fixed_cost_idr": "fixed_cost_per_trip_idr",
        "available_start": "shift_start", "available_end": "shift_end",
        "max_customers": "max_stops_per_route", "max_stops": "max_stops_per_route",
    },
    "Depot": {
        "operating_start": "open_time", "operating_end": "close_time",
        "capacity_kg": "storage_capacity_kg",
    },
}

# Kendaraan dengan status ini tidak diikutkan dalam perencanaan.
STATUS_TIDAK_SIAP = {"unavailable", "not available", "maintenance", "under maintenance",
                     "perawatan", "rusak", "inactive", "nonaktif", "tidak tersedia", "broken"}


# ---------------------------------------------------------------- util waktu
def to_minutes(value) -> float | None:
    """Mengubah '08:30', datetime.time, atau Timestamp menjadi menit sejak 00:00."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    if hasattr(value, "hour"):
        return value.hour * 60 + value.minute
    if isinstance(value, numbers.Real) and not isinstance(value, bool):
        v = float(value)
        if 0 <= v < 1:                      # pecahan hari bawaan Excel (0.25 = 06:00)
            return int(round(v * 1440))
        if 1 <= v < 24.0001:                # format jam.menit, mis. 6.3 = 06:30
            jam = int(v)
            menit = int(round((v - jam) * 100))
            return jam * 60 + menit if menit < 60 else None
        return None
    teks = str(value).strip()
    if not teks or teks.lower() == "nan":
        return None
    if re.fullmatch(r"\d{1,2}\.\d{2}", teks):     # "06.00" -> "06:00"
        teks = teks.replace(".", ":")
    bagian = teks.split(":")
    try:
        return int(bagian[0]) * 60 + int(bagian[1])
    except (ValueError, IndexError):
        return None


def to_hhmm(minutes) -> str:
    if minutes is None or pd.isna(minutes):
        return ""
    minutes = int(round(minutes))
    return f"{minutes // 60 % 24:02d}:{minutes % 60:02d}"


# ---------------------------------------------------------------- pemuatan
def _bersihkan(customers: pd.DataFrame, vehicles: pd.DataFrame, depots: pd.DataFrame,
              params_df: pd.DataFrame | None, mentah_vehicle: pd.DataFrame | None = None) -> dict:
    """Logika pembersihan tipe data, dipakai baik untuk file maupun DataFrame manual."""
    customers = customers.copy()
    vehicles = vehicles.copy()
    depots = depots.copy()
    params_df = params_df.copy() if params_df is not None else pd.DataFrame()
    catatan_default: list[str] = []

    for nama, df in (("Customers", customers), ("Vehicle", vehicles), ("Depot", depots)):
        df.columns = [str(c).strip() for c in df.columns]
        # samakan nama kolom alternatif dengan nama baku
        peta: dict[str, str] = {}
        for c in df.columns:
            baru = ALIAS_KOLOM[nama].get(c.lower().replace(" ", "_"))
            if baru and baru not in df.columns and baru not in peta.values():
                peta[c] = baru
        if peta:
            df.rename(columns=peta, inplace=True)
            catatan_default.append(f"Pemetaan kolom {nama}: "
                                   + ", ".join(f"{a} → {b}" for a, b in peta.items()))
        # buang baris yang seluruhnya kosong (umum terjadi pada data_editor Streamlit)
        df.dropna(how="all", inplace=True)

    # kendaraan yang sedang tidak siap pakai dikeluarkan dari perencanaan
    if "status" in vehicles.columns:
        tak_siap = vehicles["status"].astype(str).str.strip().str.lower().isin(STATUS_TIDAK_SIAP)
        if tak_siap.any():
            id_col = "vehicle_id" if "vehicle_id" in vehicles.columns else vehicles.columns[0]
            ids = vehicles.loc[tak_siap, id_col].astype(str).tolist()
            vehicles = vehicles[~tak_siap].copy()
            catatan_default.append(f"Vehicle: {len(ids)} kendaraan tidak diikutkan karena status "
                                   f"tidak tersedia ({', '.join(ids[:5])}).")

    # normalisasi jam menjadi teks HH:MM dan menit
    for df, cols in ((customers, ["time_window_start", "time_window_end"]),
                     (vehicles, ["shift_start", "shift_end"]),
                     (depots, ["open_time", "close_time"])):
        for c in cols:
            if c in df.columns:
                df[c + "_min"] = df[c].map(to_minutes)
                df[c] = df[c + "_min"].map(to_hhmm)

    # angka
    for df, cols in ((customers, ["demand_kg", "service_time_min", "priority_level",
                                  "latitude", "longitude"]),
                     (vehicles, ["capacity_kg", "avg_speed_kmh", "max_stops_per_route",
                                 "variable_cost_per_km_idr", "fixed_cost_per_trip_idr",
                                 "fuel_consumption_km_per_liter"]),
                     (depots, ["storage_capacity_kg", "num_loading_docks",
                               "avg_loading_time_min", "latitude", "longitude"])):
        for c in cols:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")

    # konsumsi BBM yang terlanjur tersimpan sebagai jam (mis. 07:05:00 -> 7.05)
    if mentah_vehicle is not None and "fuel_consumption_km_per_liter" in vehicles.columns \
            and "fuel_consumption_km_per_liter" in mentah_vehicle.columns:
        kolom = mentah_vehicle["fuel_consumption_km_per_liter"]
        perlu_perbaikan = vehicles["fuel_consumption_km_per_liter"].isna()
        if perlu_perbaikan.any() and len(kolom) == len(vehicles):
            def dari_jam(v):
                m = to_minutes(v)
                return None if m is None else round(m // 60 + (m % 60) / 100, 2)
            vehicles.loc[perlu_perbaikan, "fuel_consumption_km_per_liter"] = \
                kolom[perlu_perbaikan].map(dari_jam)

    # kolom opsional yang kosong / bukan angka -> pakai nilai bawaan agar solver tidak error
    for df, kol, bawaan, kol_id, label, layak in (
            (vehicles, "fuel_consumption_km_per_liter", 8.0, "vehicle_id", "Vehicle",
             lambda x: x.notna() & (x > 0)),
            (depots, "avg_loading_time_min", 30, "depot_id", "Depot",
             lambda x: x.notna() & (x > 0)),
            (customers, "priority_level", 2, "customer_id", "Customers",
             lambda x: x.notna() & (x >= 1))):
        if kol in df.columns:
            rusak = ~layak(df[kol])
            if rusak.any():
                ids = df.loc[rusak, kol_id].astype(str).tolist() if kol_id in df.columns else []
                df.loc[rusak, kol] = bawaan
                catatan_default.append(
                    f"{label}: kolom {kol} kosong/bukan angka pada {int(rusak.sum())} baris "
                    f"({', '.join(ids[:5])}{'...' if len(ids) > 5 else ''}); dipakai nilai bawaan {bawaan}.")

    # parameter
    params = dict(DEFAULT_PARAMS)
    if not params_df.empty and {"param_name", "param_value"} <= set(params_df.columns):
        for _, r in params_df.iterrows():
            nilai = pd.to_numeric(r["param_value"], errors="coerce")
            params[str(r["param_name"])] = r["param_value"] if pd.isna(nilai) else float(nilai)

    return {"customers": customers.reset_index(drop=True),
            "vehicles": vehicles.reset_index(drop=True),
            "depots": depots.reset_index(drop=True),
            "params": params,
            "catatan_default": catatan_default}


def load_data(path: str | Path = "data/Data_Operasional_Rute.xlsx") -> dict:
    """Membaca file Excel dan mengembalikan dict berisi DataFrame + parameter."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File data tidak ditemukan: {path}")

    sheets = pd.read_excel(path, sheet_name=None)
    nama_map = {s.lower(): s for s in sheets}

    def ambil(nama):
        kunci = nama_map.get(nama.lower())
        return sheets[kunci].copy() if kunci else pd.DataFrame()

    hasil = _bersihkan(ambil("Customers"), ambil("Vehicle"), ambil("Depot"),
                       ambil("Parameters"), mentah_vehicle=ambil("Vehicle"))
    hasil["source_path"] = str(path)
    return hasil


def load_data_from_frames(customers: pd.DataFrame, vehicles: pd.DataFrame,
                          depots: pd.DataFrame, params_df: pd.DataFrame | None = None,
                          source_label: str = "Input manual / diedit pengguna") -> dict:
    """Sama seperti load_data(), tapi menerima DataFrame langsung.

    Dipakai untuk data hasil unggahan (upload_excel_bytes) atau hasil input
    manual pengguna di antarmuka (mis. st.data_editor pada Streamlit).
    """
    hasil = _bersihkan(customers, vehicles, depots, params_df)
    hasil["source_path"] = source_label
    return hasil


def read_uploaded_excel(file_bytes: bytes) -> dict[str, pd.DataFrame]:
    """Membaca bytes file Excel yang diunggah pengguna menjadi dict sheet mentah.

    Nama sheet dicocokkan tanpa memandang huruf besar/kecil terhadap
    Customers / Vehicle / Depot / Parameters, sehingga pengguna lain dengan
    file serupa (nama sheet sedikit berbeda kapitalisasinya) tetap terbaca.
    """
    sheets = pd.read_excel(io.BytesIO(file_bytes), sheet_name=None)
    nama_map = {s.lower(): s for s in sheets}
    hasil = {}
    for target, alias in (("Customers", ["customers", "customer", "pelanggan"]),
                          ("Vehicle", ["vehicle", "vehicles", "kendaraan"]),
                          ("Depot", ["depot", "depots", "gudang"]),
                          ("Parameters", ["parameters", "parameter", "params"])):
        kunci = next((nama_map[a] for a in alias if a in nama_map), None)
        hasil[target] = sheets[kunci].copy() if kunci else pd.DataFrame()
    return hasil


# --------------------------------------------------------------- template
def contoh_kosong() -> dict[str, pd.DataFrame]:
    """Template kosong (header + satu baris contoh) untuk diunduh dan diisi pengguna.

    Berlaku untuk kasus apa pun, bukan hanya Medan -- kolom alamat dan
    koordinat bebas diisi sesuai kota/wilayah masing-masing pengguna.
    """
    customers = pd.DataFrame([{
        "customer_id": "C001", "customer_name": "Nama Pelanggan/Toko Contoh",
        "address": "Alamat lengkap termasuk kota", "latitude": "", "longitude": "",
        "demand_kg": 500, "priority_level": 2, "time_window_start": "08:00",
        "time_window_end": "12:00", "service_time_min": 20,
    }])
    vehicles = pd.DataFrame([{
        "vehicle_id": "V001", "plate_number": "Contoh B 1234 CD",
        "vehicle_type": "Medium Truck", "home_depot_id": "D001", "capacity_kg": 5000,
        "avg_speed_kmh": 40, "fuel_consumption_km_per_liter": 7.0,
        "fixed_cost_per_trip_idr": 250000, "variable_cost_per_km_idr": 3500,
        "shift_start": "06:00", "shift_end": "18:00", "max_stops_per_route": 8,
        "cold_chain_capable": "Tidak",
    }])
    depots = pd.DataFrame([{
        "depot_id": "D001", "depot_name": "Nama Depot/Gudang Contoh",
        "address": "Alamat lengkap termasuk kota", "latitude": "", "longitude": "",
        "open_time": "06:00", "close_time": "20:00", "storage_capacity_kg": 15000,
        "num_loading_docks": 3, "avg_loading_time_min": 30,
    }])
    params = pd.DataFrame([
        {"param_id": f"P{i+1:02d}", "param_name": k, "param_value": v}
        for i, (k, v) in enumerate(DEFAULT_PARAMS.items())
    ])
    return {"Customers": customers, "Vehicle": vehicles, "Depot": depots, "Parameters": params}


# ---------------------------------------------------------------- validasi
def validate_data(data: dict) -> tuple[list[str], list[str]]:
    """Mengembalikan (errors, warnings). Errors membuat optimasi tidak bisa jalan."""
    errors: list[str] = []
    warnings: list[str] = list(data.get("catatan_default", []))

    cust, veh, dep = data["customers"], data["vehicles"], data["depots"]

    for nama, df in (("Customers", cust), ("Vehicle", veh), ("Depot", dep)):
        if df.empty:
            errors.append(f"Sheet {nama} kosong atau tidak ditemukan.")
            continue
        hilang = [c for c in REQUIRED[nama] if c not in df.columns]
        if hilang:
            errors.append(f"Sheet {nama} kekurangan kolom: {', '.join(hilang)}")

    if errors:
        return errors, warnings

    # ID duplikat
    for nama, df, kol in (("Customers", cust, "customer_id"),
                          ("Vehicle", veh, "vehicle_id"),
                          ("Depot", dep, "depot_id")):
        dup = df[df[kol].duplicated()][kol].tolist()
        if dup:
            errors.append(f"{nama}: {kol} duplikat -> {dup}")

    # koordinat
    for nama, df, kol in (("Customers", cust, "customer_id"), ("Depot", dep, "depot_id")):
        if "latitude" not in df.columns or "longitude" not in df.columns:
            errors.append(f"{nama}: kolom latitude/longitude belum ada. "
                          f"Isi manual atau gunakan tombol 'Isi koordinat otomatis' di tab Data.")
            continue
        kosong = df[df.latitude.isna() | df.longitude.isna()][kol].tolist()
        if kosong:
            errors.append(f"{nama}: koordinat kosong untuk {kosong}. "
                          f"Isi manual di tabel atau gunakan tombol 'Isi koordinat otomatis' di tab Data.")
        luar = df[(df.latitude.notna()) &
                  ((df.latitude < -90) | (df.latitude > 90) |
                   (df.longitude < -180) | (df.longitude > 180))][kol].tolist()
        if luar:
            errors.append(f"{nama}: nilai latitude/longitude tidak valid (di luar -90..90 / -180..180) untuk {luar}.")
        di_luar_id = df[(df.latitude.notna()) &
                        ((df.latitude < -11) | (df.latitude > 6) |
                         (df.longitude < 95) | (df.longitude > 141))][kol].tolist()
        if di_luar_id and not luar:
            warnings.append(f"{nama}: koordinat di luar wilayah Indonesia untuk {di_luar_id}. "
                            f"Ini normal bila kasus Anda berada di negara lain; abaikan bila memang begitu.")

    # relasi depot
    if "home_depot_id" in veh.columns:
        asing = set(veh.home_depot_id) - set(dep.depot_id)
        if asing:
            errors.append(f"Vehicle: home_depot_id tidak ada di sheet Depot -> {sorted(asing)}")

    # kolom angka/jam wajib tidak boleh kosong
    wajib_angka = (
        ("Vehicle", veh, "vehicle_id",
         ["capacity_kg", "avg_speed_kmh", "variable_cost_per_km_idr",
          "fixed_cost_per_trip_idr", "max_stops_per_route", "shift_start_min", "shift_end_min"]),
        ("Customers", cust, "customer_id", ["service_time_min"]),
        ("Depot", dep, "depot_id", ["open_time_min", "close_time_min"]),
    )
    for nama, df, kol_id, kolom in wajib_angka:
        for k in kolom:
            if k in df.columns and df[k].isna().any():
                errors.append(f"{nama}: kolom {k.removesuffix('_min')} kosong atau bukan angka/jam "
                              f"yang valid untuk {df.loc[df[k].isna(), kol_id].tolist()}")

    # demand vs kapasitas
    if cust.demand_kg.isna().any():
        errors.append("Customers: ada demand_kg yang kosong atau bukan angka.")
    else:
        maks_kendaraan = veh.capacity_kg.max()
        kebesaran = cust[cust.demand_kg > maks_kendaraan].customer_id.tolist()
        if kebesaran:
            errors.append(f"Customers: demand melebihi kapasitas kendaraan terbesar "
                          f"({maks_kendaraan:.0f} kg) -> {kebesaran}")
        total_demand = cust.demand_kg.sum()
        total_kapasitas = veh.capacity_kg.sum()
        if total_demand > total_kapasitas:
            warnings.append(f"Total permintaan {total_demand:,.0f} kg melebihi total kapasitas "
                            f"armada {total_kapasitas:,.0f} kg, sebagian order berpotensi tidak terlayani.")
        elif total_demand > 0.9 * total_kapasitas:
            warnings.append(f"Utilisasi armada akan sangat tinggi "
                            f"({total_demand / total_kapasitas:.0%} dari kapasitas).")

    # time window
    if {"time_window_start_min", "time_window_end_min"} <= set(cust.columns):
        aneh = cust[(cust.time_window_start_min.isna()) |
                    (cust.time_window_end_min.isna()) |
                    (cust.time_window_end_min <= cust.time_window_start_min)].customer_id.tolist()
        if aneh:
            errors.append(f"Customers: time window tidak valid (akhir <= awal) -> {aneh}")

        sempit = cust[(cust.time_window_end_min - cust.time_window_start_min) <
                      cust.service_time_min].customer_id.tolist()
        if sempit:
            warnings.append(f"Customers: time window lebih pendek dari service time -> {sempit}")

    # jam kerja kendaraan vs jam operasional depot
    if {"shift_start_min", "shift_end_min"} <= set(veh.columns):
        buka = dep.set_index("depot_id")["open_time_min"].to_dict()
        tutup = dep.set_index("depot_id")["close_time_min"].to_dict()
        for _, v in veh.iterrows():
            b, t = buka.get(v.home_depot_id), tutup.get(v.home_depot_id)
            if b is not None and v.shift_start_min < b:
                warnings.append(f"{v.vehicle_id}: shift mulai {to_hhmm(v.shift_start_min)} "
                                f"sebelum depot buka {to_hhmm(b)}.")
            if t is not None and v.shift_end_min > t:
                warnings.append(f"{v.vehicle_id}: shift selesai {to_hhmm(v.shift_end_min)} "
                                f"setelah depot tutup {to_hhmm(t)}.")

    # informasi tambahan
    if "geocode_status" in cust.columns:
        perkiraan = cust[cust.geocode_status.isin(["Perkiraan", "Fallback"])].customer_id.tolist()
        if perkiraan:
            warnings.append(f"Koordinat masih perkiraan untuk {len(perkiraan)} pelanggan "
                            f"({', '.join(perkiraan[:5])}{'...' if len(perkiraan) > 5 else ''}). "
                            f"Jarak hasil hitungan bisa meleset ratusan meter.")

    return errors, warnings


def summary_table(data: dict) -> pd.DataFrame:
    """Ringkasan singkat untuk ditampilkan di aplikasi."""
    c, v, d = data["customers"], data["vehicles"], data["depots"]
    baris = [
        ("Jumlah pelanggan", str(len(c))),
        ("Jumlah depot", str(len(d))),
        ("Jumlah kendaraan", str(len(v))),
        ("Total permintaan (kg)", f"{c.demand_kg.sum():,.0f}"),
        ("Total kapasitas armada (kg)", f"{v.capacity_kg.sum():,.0f}"),
        ("Rata-rata service time (menit)", f"{c.service_time_min.mean():.1f}"),
        ("Rentang time window", f"{c.time_window_start.min()} - {c.time_window_end.max()}"),
    ]
    return pd.DataFrame(baris, columns=["Metrik", "Nilai"])


if __name__ == "__main__":
    data = load_data()
    err, warn = validate_data(data)
    print("ERRORS:", err or "tidak ada")
    print("WARNINGS:", warn or "tidak ada")
    print(summary_table(data).to_string(index=False))