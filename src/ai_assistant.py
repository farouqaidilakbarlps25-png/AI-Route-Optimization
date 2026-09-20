"""
ai_assistant.py
---------------
Integrasi Google Gemini API ke project optimasi rute.

Prinsip desain (penting untuk modul pembelajaran):
  * SOLVER menghitung, AI menafsirkan. Semua angka (jarak, biaya, telat, utilisasi)
    berasal dari route_optimization.py, lalu dikirim ke Gemini sebagai konteks JSON.
    Gemini diminta tidak mengarang angka -> mengurangi halusinasi.
  * Empat kemampuan AI:
      1. analisis_hasil()        -> laporan analis logistik dari hasil optimasi
      2. tanya_rute()            -> tanya jawab (chat) yang berpijak pada data rute
      3. bantu_perbaiki_data()   -> menjelaskan & memperbaiki error validasi data
      4. jelaskan_konsep()       -> tutor algoritma memakai KODE ASLI project
         (+ buat_pertanyaan_latihan() dan nilai_jawaban() untuk uji pemahaman)
  * API key dibaca dari file .env (GEMINI_API_KEY), tidak pernah ditulis di kode.
  * Alamat pelanggan TIDAK dikirim ke Gemini (hanya ID, nama, dan angka operasional).

Uji cepat tanpa Streamlit:
    python test_ai.py
"""
from __future__ import annotations

import inspect
import json
import os
import time
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from google.genai import errors, types

from . import route_optimization as ro
from .kpi_analysis import temuan_otomatis

# .env berada di root project (satu tingkat di atas folder src)
load_dotenv(Path(__file__).resolve().parents[1] / ".env")

# Nama model bisa diganti lewat .env tanpa mengubah kode: GEMINI_MODEL=...
MODEL = os.getenv("GEMINI_MODEL", "gemini-3.6-flash")


class AIError(RuntimeError):
    """Kesalahan yang pesannya aman ditampilkan langsung ke pengguna."""


# ==================================================================
# KLIEN & PEMANGGILAN API
# ==================================================================
_client = None


def _api_key() -> str | None:
    return os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")


def api_key_tersedia() -> bool:
    return bool(_api_key())


def set_api_key(key: str) -> None:
    """Dipakai kolom input di sidebar bila .env belum diisi (hanya untuk sesi ini)."""
    global _client
    os.environ["GEMINI_API_KEY"] = key.strip()
    _client = None


def _get_client() -> genai.Client:
    global _client
    if _client is None:
        if not api_key_tersedia():
            raise AIError("GEMINI_API_KEY belum diatur. Isi file .env atau masukkan lewat sidebar.")
        _client = genai.Client(api_key=_api_key())
    return _client


def _pesan_ramah(kode: int | None) -> str:
    peta = {
        400: "Permintaan ditolak Gemini (400). Periksa nama model di GEMINI_MODEL dan format API key.",
        401: "API key tidak valid. Periksa GEMINI_API_KEY.",
        403: "API key tidak punya akses ke model ini, atau layanan belum diaktifkan untuk akun Anda.",
        404: f"Model '{MODEL}' tidak ditemukan. Ganti GEMINI_MODEL di file .env.",
        429: "Kuota atau batas permintaan Gemini terlampaui. Tunggu sekitar satu menit lalu coba lagi.",
    }
    if kode in peta:
        return peta[kode]
    if kode and kode >= 500:
        return "Server Gemini sedang sibuk. Coba lagi beberapa saat."
    return f"Gagal memanggil Gemini (kode {kode})."


def _panggil(isi, system: str, temperature: float = 0.3, percobaan: int = 3) -> str:
    """Satu pintu semua panggilan ke Gemini: retry untuk 429/5xx + pesan galat ramah."""
    client = _get_client()
    config = types.GenerateContentConfig(system_instruction=system, temperature=temperature)
    for i in range(percobaan):
        try:
            resp = client.models.generate_content(model=MODEL, contents=isi, config=config)
        except errors.APIError as e:
            kode = getattr(e, "code", None)
            if kode in (429, 500, 502, 503, 504) and i < percobaan - 1:
                time.sleep(2 * 2 ** i)  # 2 dtk, 4 dtk, ...
                continue
            raise AIError(_pesan_ramah(kode)) from e
        except Exception as e:  # noqa: BLE001  (mis. tidak ada internet)
            raise AIError(f"Tidak bisa terhubung ke Gemini: {e}") from e
        teks = (resp.text or "").strip()
        if not teks:
            raise AIError("Gemini mengembalikan jawaban kosong (mungkin terblokir filter keamanan). "
                          "Coba ubah pertanyaan.")
        return teks
    raise AIError("Gemini tidak merespons setelah beberapa kali percobaan.")


# ==================================================================
# KONTEKS: mengubah hasil solver menjadi JSON ringkas untuk Gemini
# ==================================================================
def _ke_python(o):
    return o.item() if hasattr(o, "item") else str(o)


def bangun_konteks(result: dict, kpi: dict, data: dict, maks_stop: int = 30) -> str:
    """Ringkasan hasil optimasi dalam JSON. Sengaja tanpa alamat pelanggan."""
    p = data.get("params", {})
    stops = result["stops"]

    bermasalah = []
    if not stops.empty:
        kolom = ["route_id", "node_id", "node_name", "tiba", "time_window",
                 "tunggu_menit", "telat_menit", "status"]
        kolom = [c for c in kolom if c in stops.columns]
        mask = (stops.telat_menit > 0) | (stops.tunggu_menit > 0)
        bermasalah = stops.loc[mask, kolom].head(maks_stop).to_dict("records")

    kolom_rute = ["route_id", "vehicle_id", "depot_id", "jumlah_stop", "muatan_kg",
                  "kapasitas_kg", "utilisasi", "total_jarak_km", "durasi_menit",
                  "waktu_tunggu_menit", "keterlambatan_menit", "total_biaya_idr",
                  "selesai_pukul", "urutan"]
    rute = [{k: r[k] for k in kolom_rute if k in r} for r in result["routes"]]

    konteks = {
        "parameter": {k: p.get(k) for k in ("detour_factor", "traffic_factor",
                                            "harga_solar_idr_per_liter",
                                            "penalti_keterlambatan_per_menit",
                                            "toleransi_tiba_awal_menit")},
        "skala": {"pelanggan": len(data["customers"]), "kendaraan": len(data["vehicles"]),
                  "depot": len(data["depots"])},
        "kpi": {k: (round(v, 3) if isinstance(v, float) else v) for k, v in kpi.items()},
        "kandidat_solusi": result["candidate_summary"].to_dict("records"),
        "kandidat_terpilih": result["best_candidate"],
        "penghematan_vs_savings_murni": result["penghematan"],
        "rute": rute,
        "stop_terlambat_atau_menunggu": bermasalah,
        "pelanggan_tidak_terlayani": result["unserved"],
        "temuan_berbasis_aturan": temuan_otomatis(kpi, result, data),
    }
    return json.dumps(konteks, ensure_ascii=False, default=_ke_python)


# ==================================================================
# 1. ANALISIS HASIL
# ==================================================================
SYSTEM_ANALIS = (
    "Anda adalah analis logistik senior yang membantu manajer operasional membaca hasil "
    "optimasi rute distribusi. Solver sudah menghitung semua angka; tugas Anda menafsirkan, "
    "bukan menghitung ulang. Aturan: (1) hanya pakai angka dan fakta yang ada di DATA, jangan "
    "mengarang angka, pelanggan, atau penyebab yang tidak didukung data; (2) bila data tidak "
    "cukup untuk menyimpulkan sesuatu, katakan terus terang; (3) bahasa Indonesia, ringkas dan "
    "langsung; (4) tulis rupiah dengan format Rp 1.234.567."
)


def analisis_hasil(result: dict, kpi: dict, data: dict) -> str:
    konteks = bangun_konteks(result, kpi, data)
    prompt = (
        f"DATA HASIL OPTIMASI (JSON):\n{konteks}\n\n"
        "Susun laporan untuk manajer operasional dengan format markdown berikut:\n"
        "### Ringkasan\n2-3 kalimat: seberapa baik rencana rute ini.\n"
        "### Yang sudah baik\nMaksimal 3 poin, sertakan angkanya.\n"
        "### Masalah & kemungkinan penyebab\nBerdasarkan stop terlambat/menunggu, utilisasi, "
        "pelanggan tidak terlayani. Sebut ID rute/pelanggan yang relevan.\n"
        "### Rekomendasi tindakan\nMaksimal 4 poin. Tiap poin harus konkret (mis. geser jam "
        "berangkat, longgarkan time window pelanggan tertentu, tambah/ganti kendaraan, ubah "
        "parameter) dan sebut dampak yang diharapkan.\n"
        "### Risiko operasional\n1-2 poin."
    )
    return _panggil(prompt, SYSTEM_ANALIS, temperature=0.3)


# ==================================================================
# 2. TANYA JAWAB BERPIJAK PADA DATA
# ==================================================================
SYSTEM_TANYA = (
    "Anda adalah asisten perencana rute distribusi. Jawab pertanyaan pengguna HANYA berdasarkan "
    "DATA HASIL OPTIMASI di bawah. Jika jawabannya tidak ada di data, katakan 'data tidak "
    "memuat informasi itu' dan sarankan langkah untuk mencarinya. Jangan mengarang angka. "
    "Bahasa Indonesia, jawaban singkat (maksimal sekitar 150 kata) kecuali diminta rinci."
)


def tanya_rute(pertanyaan: str, riwayat: list[dict], konteks: str) -> str:
    """riwayat: list dict {"role": "user"|"assistant", "content": str}."""
    isi = [types.Content(role="user" if m["role"] == "user" else "model",
                         parts=[types.Part(text=m["content"])])
           for m in riwayat[-8:]]  # hanya 8 pesan terakhir agar hemat token
    isi.append(types.Content(role="user", parts=[types.Part(text=pertanyaan)]))
    system = f"{SYSTEM_TANYA}\n\nDATA HASIL OPTIMASI (JSON):\n{konteks}"
    return _panggil(isi, system, temperature=0.2)


# ==================================================================
# 3. BANTU PERBAIKI DATA (problem solving)
# ==================================================================
SYSTEM_DATA = (
    "Anda membantu pengguna non-teknis memperbaiki data Excel untuk aplikasi optimasi rute. "
    "Jelaskan setiap masalah dengan bahasa sederhana, sebut sheet dan kolom yang harus diubah, "
    "dan beri langkah perbaikan yang konkret. Bahasa Indonesia, gunakan daftar bernomor."
)


def bantu_perbaiki_data(errors_: list[str], warnings_: list[str], data: dict) -> str:
    struktur = {nama: list(df.columns) for nama, df in
                (("Customers", data["customers"]), ("Vehicle", data["vehicles"]),
                 ("Depot", data["depots"]))}
    prompt = (
        f"KOLOM YANG ADA PER SHEET:\n{json.dumps(struktur, ensure_ascii=False)}\n\n"
        f"ERROR VALIDASI (menghalangi optimasi):\n{json.dumps(errors_, ensure_ascii=False)}\n\n"
        f"WARNING:\n{json.dumps(warnings_, ensure_ascii=False)}\n\n"
        "Untuk setiap error: (a) jelaskan artinya dan kenapa itu masalah bagi optimasi rute, "
        "(b) beri langkah perbaikan. Urutkan dari yang paling menghalangi."
    )
    return _panggil(prompt, SYSTEM_DATA, temperature=0.2)


# ==================================================================
# 4. TUTOR ALGORITMA — menjelaskan KODE ASLI project
# ==================================================================
KONSEP = {
    "Clarke-Wright Savings": [ro._savings_routes],
    "Perbaikan lokal 2-opt": [ro._two_opt, ro._panjang],
    "Time window & simulasi jadwal": [ro._simulasi_rute],
    "Jarak haversine & faktor detour": [ro.haversine_km, ro.build_distance_matrix],
    "Alur solver (CVRP multi-depot)": [ro._bangun_solusi],
}

SYSTEM_TUTOR = (
    "Anda adalah dosen pemrograman dan riset operasi yang sabar. Jelaskan konsep dengan analogi "
    "sehari-hari, lalu hubungkan ke kode yang diberikan (sebut nama fungsi dan bagian barisnya). "
    "Jangan mengubah kode. Bahasa Indonesia."
)


def _kode_sumber(topik: str) -> str:
    return "\n\n".join(inspect.getsource(f) for f in KONSEP[topik])


def jelaskan_konsep(topik: str, level: str = "Pemula", konteks: str | None = None) -> str:
    bagian_data = (f"\n\nHASIL NYATA DARI PROJECT PENGGUNA (JSON):\n{konteks}\n"
                   "Rujuk angka-angka ini sebagai contoh konkret bila relevan."
                   if konteks else "")
    prompt = (
        f"Level pembelajar: {level}\nKonsep: {topik}\n\n"
        f"KODE PROJECT:\n```python\n{_kode_sumber(topik)}\n```{bagian_data}\n\n"
        "Format jawaban (markdown):\n"
        "### Ide dasarnya\nAnalogi sederhana + tujuan algoritma.\n"
        "### Cara kerja langkah demi langkah\nBerurutan, tiap langkah menunjuk bagian kode.\n"
        "### Contoh dari project ini\nGunakan angka dari hasil nyata bila ada; bila tidak ada, "
        "buat contoh mini 3-4 pelanggan.\n"
        "### Kelebihan & keterbatasan\n2-3 poin.\n"
        "Sesuaikan kedalaman dengan level pembelajar."
    )
    return _panggil(prompt, SYSTEM_TUTOR, temperature=0.4)


def buat_pertanyaan_latihan(topik: str, level: str = "Pemula") -> str:
    prompt = (
        f"Buat SATU pertanyaan latihan uraian singkat tentang '{topik}' untuk level {level}, "
        f"berdasarkan kode berikut:\n```python\n{_kode_sumber(topik)}\n```\n"
        "Pertanyaan harus menguji pemahaman (bukan hafalan), bisa dijawab 2-4 kalimat. "
        "Keluarkan HANYA teks pertanyaannya."
    )
    return _panggil(prompt, SYSTEM_TUTOR, temperature=0.7)


def nilai_jawaban(topik: str, pertanyaan: str, jawaban: str) -> str:
    prompt = (
        f"Konsep: {topik}\nPertanyaan: {pertanyaan}\nJawaban pembelajar: {jawaban}\n\n"
        f"Kode acuan:\n```python\n{_kode_sumber(topik)}\n```\n\n"
        "Nilai jawaban secara adil dan suportif. Format:\n"
        "**Skor: X/100**\n### Yang sudah tepat\n### Yang perlu diperbaiki\n"
        "### Jawaban ideal (singkat)"
    )
    return _panggil(prompt, SYSTEM_TUTOR, temperature=0.2)
