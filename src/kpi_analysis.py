"""
kpi_analysis.py
---------------
Menghitung indikator kinerja dari hasil optimasi dan menyiapkan tabel
yang siap ditampilkan di Streamlit atau diekspor ke Excel.
"""
from __future__ import annotations

import pandas as pd


def hitung_kpi(result: dict, data: dict) -> dict:
    """Mengembalikan dict KPI utama."""
    rute = result["route_summary"]
    stops = result["stops"]
    cust = data["customers"]

    if rute.empty:
        return {}

    total_stop = int(rute.jumlah_stop.sum())
    telat_stop = int((stops.telat_menit > 0).sum()) if not stops.empty else 0
    armada_total = len(data["vehicles"])

    return {
        "jumlah_rute": len(rute),
        "kendaraan_dipakai": rute.vehicle_id.nunique(),
        "kendaraan_menganggur": armada_total - rute.vehicle_id.nunique(),
        "pelanggan_terlayani": total_stop,
        "pelanggan_tidak_terlayani": len(result.get("unserved", [])),
        "tingkat_layanan": total_stop / len(cust) if len(cust) else 0,
        "total_jarak_km": float(rute.total_jarak_km.sum()),
        "jarak_per_pelanggan_km": float(rute.total_jarak_km.sum()) / total_stop if total_stop else 0,
        "total_durasi_jam": float(rute.durasi_menit.sum()) / 60,
        "total_waktu_tunggu_menit": float(rute.waktu_tunggu_menit.sum()),
        "total_keterlambatan_menit": float(rute.keterlambatan_menit.sum()),
        "stop_terlambat": telat_stop,
        "on_time_rate": 1 - (telat_stop / total_stop) if total_stop else 0,
        "total_muatan_kg": float(rute.muatan_kg.sum()),
        "rata_utilisasi": float(rute.utilisasi.mean()),
        "utilisasi_terendah": float(rute.utilisasi.min()),
        "total_biaya_idr": float(rute.total_biaya_idr.sum()),
        "biaya_per_km_idr": float(rute.total_biaya_idr.sum()) / float(rute.total_jarak_km.sum())
        if rute.total_jarak_km.sum() else 0,
        "biaya_per_kg_idr": float(rute.total_biaya_idr.sum()) / float(rute.muatan_kg.sum())
        if rute.muatan_kg.sum() else 0,
        "total_bbm_liter": float(rute.konsumsi_bbm_liter.sum()),
        "total_emisi_co2_kg": float(rute.emisi_co2_kg.sum()),
    }


def tabel_kpi(kpi: dict) -> pd.DataFrame:
    """Versi tabel yang enak dibaca manusia."""
    if not kpi:
        return pd.DataFrame(columns=["Indikator", "Nilai", "Satuan"])
    baris = [
        ("Jumlah rute", f"{kpi['jumlah_rute']}", "rute"),
        ("Kendaraan dipakai", f"{kpi['kendaraan_dipakai']}", "unit"),
        ("Kendaraan menganggur", f"{kpi['kendaraan_menganggur']}", "unit"),
        ("Pelanggan terlayani", f"{kpi['pelanggan_terlayani']}", "titik"),
        ("Pelanggan tidak terlayani", f"{kpi['pelanggan_tidak_terlayani']}", "titik"),
        ("Tingkat layanan", f"{kpi['tingkat_layanan']:.1%}", ""),
        ("Ketepatan waktu (on time)", f"{kpi['on_time_rate']:.1%}", ""),
        ("Total jarak tempuh", f"{kpi['total_jarak_km']:,.2f}", "km"),
        ("Jarak per pelanggan", f"{kpi['jarak_per_pelanggan_km']:,.2f}", "km"),
        ("Total durasi operasi", f"{kpi['total_durasi_jam']:,.2f}", "jam"),
        ("Total waktu tunggu", f"{kpi['total_waktu_tunggu_menit']:,.0f}", "menit"),
        ("Total keterlambatan", f"{kpi['total_keterlambatan_menit']:,.0f}", "menit"),
        ("Total muatan terkirim", f"{kpi['total_muatan_kg']:,.0f}", "kg"),
        ("Rata-rata utilisasi kendaraan", f"{kpi['rata_utilisasi']:.1%}", ""),
        ("Utilisasi terendah", f"{kpi['utilisasi_terendah']:.1%}", ""),
        ("Total biaya operasional", f"Rp {kpi['total_biaya_idr']:,.0f}", ""),
        ("Biaya per km", f"Rp {kpi['biaya_per_km_idr']:,.0f}", ""),
        ("Biaya per kg terkirim", f"Rp {kpi['biaya_per_kg_idr']:,.0f}", ""),
        ("Konsumsi BBM", f"{kpi['total_bbm_liter']:,.2f}", "liter"),
        ("Emisi CO2", f"{kpi['total_emisi_co2_kg']:,.2f}", "kg"),
    ]
    return pd.DataFrame(baris, columns=["Indikator", "Nilai", "Satuan"])


def tabel_per_kendaraan(result: dict) -> pd.DataFrame:
    r = result["route_summary"].copy()
    if r.empty:
        return r
    kolom = ["route_id", "vehicle_id", "depot_id", "jumlah_stop", "muatan_kg", "kapasitas_kg",
             "utilisasi", "total_jarak_km", "durasi_menit", "keterlambatan_menit",
             "konsumsi_bbm_liter", "total_biaya_idr", "emisi_co2_kg", "selesai_pukul"]
    return r[[c for c in kolom if c in r.columns]]


def temuan_otomatis(kpi: dict, result: dict, data: dict) -> list[str]:
    """Catatan analitis sederhana berbasis aturan, tanpa perlu API LLM."""
    if not kpi:
        return ["Belum ada rute yang terbentuk."]
    catatan = []
    hemat = result.get("penghematan", {})

    if hemat.get("hemat_jarak_persen", 0) > 0:
        catatan.append(
            f"Kandidat '{result['best_candidate']}' memangkas jarak "
            f"{hemat['hemat_km']:.2f} km ({hemat['hemat_jarak_persen']:.1f}%) dan biaya "
            f"Rp {hemat['hemat_biaya_idr']:,.0f} dibanding savings murni.")
    else:
        catatan.append("Perbaikan lokal tidak menemukan rute yang lebih pendek dari solusi awal, "
                       "artinya solusi savings sudah cukup baik untuk data sekecil ini.")

    if kpi["pelanggan_tidak_terlayani"] > 0:
        catatan.append(
            f"{kpi['pelanggan_tidak_terlayani']} pelanggan belum masuk rute manapun. "
            f"Penyebab tersering: kapasitas atau batas stop per kendaraan sudah penuh.")

    if kpi["rata_utilisasi"] < 0.6:
        catatan.append(
            f"Rata-rata utilisasi hanya {kpi['rata_utilisasi']:.0%}. Pertimbangkan menggabungkan "
            f"rute atau memakai kendaraan berkapasitas lebih kecil agar biaya tetap per trip turun.")
    elif kpi["rata_utilisasi"] > 0.9:
        catatan.append(
            f"Utilisasi rata-rata {kpi['rata_utilisasi']:.0%}, sangat padat. "
            f"Tambahan order mendadak berisiko tidak tertampung.")

    if kpi["stop_terlambat"] > 0:
        telat = result["stops"]
        daftar = telat[telat.telat_menit > 0].node_id.tolist()
        catatan.append(
            f"{kpi['stop_terlambat']} kunjungan melewati time window: {', '.join(daftar[:6])}. "
            f"Coba majukan jam berangkat atau longgarkan time window pelanggan tersebut.")

    if kpi["total_waktu_tunggu_menit"] > 60:
        catatan.append(
            f"Total waktu menunggu {kpi['total_waktu_tunggu_menit']:.0f} menit karena kendaraan "
            f"tiba sebelum pelanggan buka. Jam berangkat bisa digeser mundur.")

    if kpi["kendaraan_menganggur"] > 0:
        catatan.append(
            f"{kpi['kendaraan_menganggur']} kendaraan tidak terpakai pada perencanaan ini.")

    return catatan


def ekspor_excel(result: dict, kpi: dict, path: str = "outputs/hasil_optimasi.xlsx") -> str:
    """Menyimpan seluruh hasil ke satu file Excel."""
    from pathlib import Path
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(path, engine="openpyxl") as xl:
        tabel_kpi(kpi).to_excel(xl, sheet_name="KPI", index=False)
        result["candidate_summary"].to_excel(xl, sheet_name="Perbandingan_Kandidat", index=False)
        tabel_per_kendaraan(result).to_excel(xl, sheet_name="Ringkasan_Rute", index=False)
        if not result["stops"].empty:
            result["stops"].to_excel(xl, sheet_name="Detail_Stop", index=False)
        result["distance_matrix"].to_excel(xl, sheet_name="Matriks_Jarak")
    return path
