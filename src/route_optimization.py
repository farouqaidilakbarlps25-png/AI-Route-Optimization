"""
route_optimization.py
---------------------
Optimasi rute (Capacitated VRP with Time Windows, multi-depot, armada heterogen)
TANPA Google Maps API. Jarak dihitung sendiri dari koordinat memakai haversine
dikali faktor tortuositas, sehingga project bisa jalan sepenuhnya offline.

Alur:
    matriks jarak -> Clarke-Wright savings -> perbaikan lokal 2-opt & relokasi
    -> simulasi jadwal (time window) -> perhitungan biaya
Beberapa kombinasi parameter dijalankan sebagai "kandidat", lalu yang terbaik dipilih.

Pemakaian:
    from src.route_optimization import optimize_from_data
    result = optimize_from_data(data)
    result["candidate_summary"]   # DataFrame perbandingan kandidat
    result["route_tables"]        # dict nama_rute -> DataFrame rincian per stop
    result["routes"]              # list rute mentah (untuk peta)
    result["stops"]               # DataFrame semua stop digabung
"""
from __future__ import annotations

import math
from itertools import count

import pandas as pd

from .data_processing import to_hhmm

EARTH_R = 6371.0


def _angka_atau(nilai, bawaan: float) -> float:
    """float(nilai); bila None/NaN/bukan angka/<=0 dipakai nilai bawaan."""
    try:
        v = float(nilai)
    except (TypeError, ValueError):
        return bawaan
    return v if v == v and v > 0 else bawaan


# ------------------------------------------------------------------ jarak
def haversine_km(lat1, lon1, lat2, lon2) -> float:
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * EARTH_R * math.asin(math.sqrt(a))


def build_nodes(data: dict) -> pd.DataFrame:
    """Menggabungkan depot dan pelanggan menjadi satu daftar node."""
    c, d = data["customers"], data["depots"]
    dep = pd.DataFrame({
        "node_id": d.depot_id, "node_name": d.depot_name, "node_type": "DEPOT",
        "latitude": d.latitude, "longitude": d.longitude, "demand_kg": 0.0,
        "service_time_min": d.get("avg_loading_time_min", 30),
        "tw_start": d.open_time_min, "tw_end": d.close_time_min,
        "priority_level": 0, "address": d.address,
    })
    cus = pd.DataFrame({
        "node_id": c.customer_id, "node_name": c.customer_name, "node_type": "CUSTOMER",
        "latitude": c.latitude, "longitude": c.longitude, "demand_kg": c.demand_kg,
        "service_time_min": c.service_time_min,
        "tw_start": c.time_window_start_min, "tw_end": c.time_window_end_min,
        "priority_level": c.get("priority_level", 2), "address": c.address,
    })
    return pd.concat([dep, cus], ignore_index=True)


def build_distance_matrix(nodes: pd.DataFrame, detour_factor: float = 1.32) -> pd.DataFrame:
    ids = nodes.node_id.tolist()
    lat = nodes.latitude.tolist()
    lon = nodes.longitude.tolist()
    n = len(ids)
    mat = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            km = round(haversine_km(lat[i], lon[i], lat[j], lon[j]) * detour_factor, 3)
            mat[i][j] = mat[j][i] = km
    return pd.DataFrame(mat, index=ids, columns=ids)


# ------------------------------------------------------------ Clarke-Wright
def _savings_routes(depot_id, pelanggan, dist, kapasitas, max_stops, demand, lam=1.0):
    """Membentuk rute awal dengan metode savings Clarke-Wright."""
    rute = {c: [c] for c in pelanggan}
    beban = {c: demand[c] for c in pelanggan}

    savings = []
    for a in pelanggan:
        for b in pelanggan:
            if a < b:
                s = dist.at[depot_id, a] + dist.at[depot_id, b] - lam * dist.at[a, b]
                savings.append((s, a, b))
    savings.sort(reverse=True)

    induk = {c: c for c in pelanggan}

    def cari(c):
        while induk[c] != c:
            induk[c] = induk[induk[c]]
            c = induk[c]
        return c

    for _s, a, b in savings:
        ra, rb = cari(a), cari(b)
        if ra == rb:
            continue
        if beban[ra] + beban[rb] > kapasitas:
            continue
        if len(rute[ra]) + len(rute[rb]) > max_stops:
            continue
        # gabung hanya bila a dan b berada di ujung rutenya masing-masing
        if rute[ra][-1] == a and rute[rb][0] == b:
            gabung = rute[ra] + rute[rb]
        elif rute[ra][0] == a and rute[rb][-1] == b:
            gabung = rute[rb] + rute[ra]
        elif rute[ra][0] == a and rute[rb][0] == b:
            gabung = rute[ra][::-1] + rute[rb]
        elif rute[ra][-1] == a and rute[rb][-1] == b:
            gabung = rute[ra] + rute[rb][::-1]
        else:
            continue
        rute[ra] = gabung
        beban[ra] += beban[rb]
        induk[rb] = ra
        del rute[rb]
    return [r for k, r in rute.items() if cari(k) == k]


def _panjang(seq, depot_id, dist) -> float:
    titik = [depot_id] + seq + [depot_id]
    return sum(dist.at[titik[i], titik[i + 1]] for i in range(len(titik) - 1))


def _two_opt(seq, depot_id, dist, max_iter=200):
    """Perbaikan lokal 2-opt pada satu rute."""
    if len(seq) < 3:
        return seq
    best = seq[:]
    best_len = _panjang(best, depot_id, dist)
    for _ in range(max_iter):
        membaik = False
        for i in range(len(best) - 1):
            for j in range(i + 1, len(best)):
                kandidat = best[:i] + best[i:j + 1][::-1] + best[j + 1:]
                pjg = _panjang(kandidat, depot_id, dist)
                if pjg < best_len - 1e-9:
                    best, best_len, membaik = kandidat, pjg, True
        if not membaik:
            break
    return best


def _urut_prioritas(seq, nodes_idx):
    """Menggeser pelanggan prioritas 1 ke depan bila jaraknya tidak banyak berubah."""
    return sorted(seq, key=lambda c: (nodes_idx[c]["tw_end"], nodes_idx[c]["priority_level"]))


# ------------------------------------------------------- simulasi & biaya
def _simulasi_rute(seq, kendaraan, depot, dist, nodes_idx, params):
    """Menghitung jadwal, jarak, biaya, dan keterlambatan satu rute."""
    traffic = float(params.get("traffic_factor", 1.18))
    harga_bbm = float(params.get("harga_solar_idr_per_liter", 6800))
    penalti_menit = float(params.get("penalti_keterlambatan_per_menit", 15000))
    tol_awal = float(params.get("toleransi_tiba_awal_menit", 20))
    emisi_per_liter = float(params.get("emisi_co2_solar_per_liter", 2.68))

    depot_id = depot["depot_id"]
    kecepatan = float(kendaraan["avg_speed_kmh"]) or 40.0
    waktu = max(float(kendaraan["shift_start_min"]), float(depot["open_time_min"]))
    muat = _angka_atau(depot.get("avg_loading_time_min"), 30.0)

    baris = []
    total_km = total_telat = total_tunggu = 0.0
    beban = sum(nodes_idx[c]["demand_kg"] for c in seq)
    sisa = beban

    baris.append({
        "stop_sequence": 1, "node_id": depot_id, "node_name": depot["depot_name"],
        "node_type": "Depot (Berangkat)", "jarak_dari_sebelumnya_km": 0.0,
        "waktu_tempuh_menit": 0.0, "tiba": to_hhmm(waktu), "mulai_layanan": to_hhmm(waktu),
        "selesai": to_hhmm(waktu + muat), "service_menit": muat,
        "muatan_turun_kg": 0.0, "muatan_di_atas_kg": beban,
        "tunggu_menit": 0.0, "telat_menit": 0.0,
        "time_window": f"{to_hhmm(depot['open_time_min'])}-{to_hhmm(depot['close_time_min'])}",
        "status": "OK",
    })
    waktu += muat
    sekarang = depot_id

    for urut, c in enumerate(seq, start=2):
        km = dist.at[sekarang, c]
        menit = km / kecepatan * 60 * traffic
        tiba = waktu + menit
        info = nodes_idx[c]
        tunggu = max(0.0, info["tw_start"] - tiba) if info["tw_start"] is not None else 0.0
        if tunggu > tol_awal:
            tunggu = max(0.0, info["tw_start"] - tiba)  # tetap menunggu, dicatat sebagai idle
        mulai = tiba + tunggu
        telat = max(0.0, mulai - info["tw_end"]) if info["tw_end"] is not None else 0.0
        selesai = mulai + info["service_time_min"]
        sisa -= info["demand_kg"]

        baris.append({
            "stop_sequence": urut, "node_id": c, "node_name": info["node_name"],
            "node_type": "Pelanggan", "jarak_dari_sebelumnya_km": round(km, 2),
            "waktu_tempuh_menit": round(menit, 1), "tiba": to_hhmm(tiba),
            "mulai_layanan": to_hhmm(mulai), "selesai": to_hhmm(selesai),
            "service_menit": info["service_time_min"],
            "muatan_turun_kg": info["demand_kg"], "muatan_di_atas_kg": round(sisa, 1),
            "tunggu_menit": round(tunggu, 1), "telat_menit": round(telat, 1),
            "time_window": f"{to_hhmm(info['tw_start'])}-{to_hhmm(info['tw_end'])}",
            "status": "Terlambat" if telat > 0 else ("Menunggu" if tunggu > 0 else "OK"),
        })
        total_km += km
        total_telat += telat
        total_tunggu += tunggu
        waktu = selesai
        sekarang = c

    km_pulang = dist.at[sekarang, depot_id]
    menit_pulang = km_pulang / kecepatan * 60 * traffic
    waktu += menit_pulang
    total_km += km_pulang
    baris.append({
        "stop_sequence": len(seq) + 2, "node_id": depot_id, "node_name": depot["depot_name"],
        "node_type": "Depot (Kembali)", "jarak_dari_sebelumnya_km": round(km_pulang, 2),
        "waktu_tempuh_menit": round(menit_pulang, 1), "tiba": to_hhmm(waktu),
        "mulai_layanan": to_hhmm(waktu), "selesai": to_hhmm(waktu),
        "service_menit": 0, "muatan_turun_kg": 0.0, "muatan_di_atas_kg": round(sisa, 1),
        "tunggu_menit": 0.0, "telat_menit": 0.0,
        "time_window": f"{to_hhmm(depot['open_time_min'])}-{to_hhmm(depot['close_time_min'])}",
        "status": "Lewat jam tutup depot" if waktu > depot["close_time_min"] else "OK",
    })

    konsumsi = _angka_atau(kendaraan.get("fuel_consumption_km_per_liter"), 8.0)
    liter = total_km / konsumsi if konsumsi else 0.0
    biaya_variabel = total_km * float(kendaraan["variable_cost_per_km_idr"])
    biaya_tetap = float(kendaraan["fixed_cost_per_trip_idr"])
    biaya_penalti = total_telat * penalti_menit
    kapasitas = float(kendaraan["capacity_kg"])

    ringkas = {
        "route_id": None,
        "vehicle_id": kendaraan["vehicle_id"],
        "depot_id": depot_id,
        "jumlah_stop": len(seq),
        "muatan_kg": round(beban, 1),
        "kapasitas_kg": kapasitas,
        "utilisasi": round(beban / kapasitas, 4) if kapasitas else 0.0,
        "total_jarak_km": round(total_km, 2),
        "durasi_menit": round(waktu - max(float(kendaraan["shift_start_min"]),
                                          float(depot["open_time_min"])), 1),
        "waktu_tunggu_menit": round(total_tunggu, 1),
        "keterlambatan_menit": round(total_telat, 1),
        "konsumsi_bbm_liter": round(liter, 2),
        "biaya_bbm_idr": round(liter * harga_bbm),
        "biaya_variabel_idr": round(biaya_variabel),
        "biaya_tetap_idr": round(biaya_tetap),
        "biaya_penalti_idr": round(biaya_penalti),
        "total_biaya_idr": round(biaya_variabel + biaya_tetap + biaya_penalti),
        "emisi_co2_kg": round(liter * emisi_per_liter, 2),
        "selesai_pukul": to_hhmm(waktu),
        "urutan": seq,
    }
    return ringkas, pd.DataFrame(baris)


# ------------------------------------------------------------------ solver
def _bangun_solusi(data, dist, nodes_idx, lam, pakai_2opt, urut_tw, params):
    cust, veh, dep = data["customers"], data["vehicles"], data["depots"]
    depot_rec = dep.set_index("depot_id").to_dict("index")
    for k, v in depot_rec.items():
        v["depot_id"] = k

    # setiap pelanggan ditugaskan ke depot terdekat
    tugas = {}
    for _, c in cust.iterrows():
        terdekat = min(dep.depot_id, key=lambda d: dist.at[d, c.customer_id])
        tugas.setdefault(terdekat, []).append(c.customer_id)

    demand = dict(zip(cust.customer_id, cust.demand_kg))
    hasil, tabel = [], {}
    penghitung = count(1)
    tidak_terlayani = []

    for depot_id, pelanggan in tugas.items():
        armada = veh[veh.home_depot_id == depot_id].sort_values("capacity_kg", ascending=False)
        if armada.empty:
            tidak_terlayani.extend(pelanggan)
            continue
        kapasitas = float(armada.capacity_kg.max())
        max_stops = int(armada.max_stops_per_route.max())

        rute_awal = _savings_routes(depot_id, pelanggan, dist, kapasitas, max_stops,
                                    demand, lam)
        rute_awal.sort(key=lambda r: -sum(demand[c] for c in r))

        daftar_kendaraan = list(armada.to_dict("records"))
        for i, seq in enumerate(rute_awal):
            if i >= len(daftar_kendaraan):
                tidak_terlayani.extend(seq)
                continue
            kend = daftar_kendaraan[i]
            seq = [c for c in seq]
            # buang pelanggan yang tidak muat di kendaraan ini
            while sum(demand[c] for c in seq) > float(kend["capacity_kg"]) and seq:
                tidak_terlayani.append(seq.pop())
            while len(seq) > int(kend["max_stops_per_route"]) and seq:
                tidak_terlayani.append(seq.pop())
            if not seq:
                continue
            if urut_tw:
                seq = _urut_prioritas(seq, nodes_idx)
            if pakai_2opt:
                seq = _two_opt(seq, depot_id, dist, int(params.get("max_iterasi_2opt", 200)))

            ringkas, df = _simulasi_rute(seq, kend, depot_rec[depot_id], dist,
                                         nodes_idx, params)
            rid = f"RT-{next(penghitung):02d}"
            ringkas["route_id"] = rid
            df.insert(0, "route_id", rid)
            df.insert(1, "vehicle_id", kend["vehicle_id"])
            hasil.append(ringkas)
            tabel[rid] = df

    return hasil, tabel, tidak_terlayani


def _skor(rute, tidak_terlayani, params):
    """Fungsi objektif: biaya total + penalti pelanggan tidak terlayani."""
    biaya = sum(r["total_biaya_idr"] for r in rute)
    return biaya + len(tidak_terlayani) * 750_000


# ----------------------------------------------------------------- publik
def optimize_from_data(data: dict, kandidat: list[dict] | None = None) -> dict:
    """Menjalankan beberapa kandidat konfigurasi lalu memilih yang terbaik."""
    params = data.get("params", {})
    nodes = build_nodes(data)

    if nodes.latitude.isna().any() or nodes.longitude.isna().any():
        raise ValueError("Masih ada node tanpa koordinat. Jalankan 'python src/geocoding.py' "
                         "atau isi kolom latitude/longitude secara manual.")

    dist = build_distance_matrix(nodes, float(params.get("detour_factor", 1.32)))
    nodes_idx = nodes.set_index("node_id").to_dict("index")

    if kandidat is None:
        kandidat = [
            {"nama": "Savings murni", "lam": 1.0, "pakai_2opt": False, "urut_tw": False},
            {"nama": "Savings + 2-opt", "lam": 1.0, "pakai_2opt": True, "urut_tw": False},
            {"nama": "Savings + 2-opt + urut time window", "lam": 1.0, "pakai_2opt": True, "urut_tw": True},
            {"nama": "Savings lambda 0.8 + 2-opt", "lam": 0.8, "pakai_2opt": True, "urut_tw": False},
            {"nama": "Savings lambda 1.4 + 2-opt", "lam": 1.4, "pakai_2opt": True, "urut_tw": False},
        ]

    semua = []
    for k in kandidat:
        rute, tabel, sisa = _bangun_solusi(data, dist, nodes_idx, k["lam"],
                                           k["pakai_2opt"], k["urut_tw"], params)
        semua.append({
            "nama": k["nama"], "config": k, "routes": rute,
            "route_tables": tabel, "unserved": sisa, "skor": _skor(rute, sisa, params),
        })

    semua.sort(key=lambda s: s["skor"])
    terbaik = semua[0]
    baseline = next((s for s in semua if s["nama"] == "Savings murni"), semua[-1])

    ringkasan = pd.DataFrame([{
        "kandidat": s["nama"],
        "jumlah_rute": len(s["routes"]),
        "total_jarak_km": round(sum(r["total_jarak_km"] for r in s["routes"]), 2),
        "total_biaya_idr": sum(r["total_biaya_idr"] for r in s["routes"]),
        "keterlambatan_menit": round(sum(r["keterlambatan_menit"] for r in s["routes"]), 1),
        "rata_utilisasi": round(
            sum(r["utilisasi"] for r in s["routes"]) / len(s["routes"]), 4) if s["routes"] else 0,
        "pelanggan_tidak_terlayani": len(s["unserved"]),
        "skor_objektif": round(s["skor"]),
        "terpilih": "YA" if s is terbaik else "",
    } for s in semua])

    stops = pd.concat(terbaik["route_tables"].values(), ignore_index=True) \
        if terbaik["route_tables"] else pd.DataFrame()
    if not stops.empty:
        stops = stops.merge(nodes[["node_id", "latitude", "longitude", "address"]],
                            on="node_id", how="left")

    jarak_baseline = sum(r["total_jarak_km"] for r in baseline["routes"])
    jarak_terbaik = sum(r["total_jarak_km"] for r in terbaik["routes"])
    biaya_baseline = sum(r["total_biaya_idr"] for r in baseline["routes"])
    biaya_terbaik = sum(r["total_biaya_idr"] for r in terbaik["routes"])

    return {
        "candidate_summary": ringkasan,
        "best_candidate": terbaik["nama"],
        "routes": terbaik["routes"],
        "route_tables": terbaik["route_tables"],
        "route_summary": pd.DataFrame(
            [{k: v for k, v in r.items() if k != "urutan"} for r in terbaik["routes"]]),
        "stops": stops,
        "unserved": terbaik["unserved"],
        "nodes": nodes,
        "distance_matrix": dist,
        "penghematan": {
            "jarak_baseline_km": round(jarak_baseline, 2),
            "jarak_optimal_km": round(jarak_terbaik, 2),
            "hemat_km": round(jarak_baseline - jarak_terbaik, 2),
            "hemat_jarak_persen": round((jarak_baseline - jarak_terbaik) / jarak_baseline * 100, 2)
            if jarak_baseline else 0.0,
            "biaya_baseline_idr": biaya_baseline,
            "biaya_optimal_idr": biaya_terbaik,
            "hemat_biaya_idr": biaya_baseline - biaya_terbaik,
        },
    }


if __name__ == "__main__":
    from .data_processing import load_data
    d = load_data()
    r = optimize_from_data(d)
    print(r["candidate_summary"].to_string(index=False))
    print("\nKandidat terpilih:", r["best_candidate"])
    print(r["route_summary"].to_string(index=False))