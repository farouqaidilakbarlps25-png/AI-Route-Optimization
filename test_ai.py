"""Uji cepat integrasi Gemini tanpa Streamlit: python test_ai.py"""
from src import ai_assistant as ai
from src.data_processing import load_data, validate_data
from src.kpi_analysis import hitung_kpi
from src.route_optimization import optimize_from_data

if not ai.api_key_tersedia():
    raise SystemExit("GEMINI_API_KEY belum diatur di file .env")

print(f"Model: {ai.MODEL}")

# 1) Tes koneksi paling sederhana
print("\n=== TES KONEKSI ===")
print(ai._panggil("Balas satu kata saja: siap", "Anda asisten singkat.", 0.0))

# 2) Jalankan pipeline solver seperti biasa
data = load_data("data/Data_Operasional_Rute.xlsx")
errors, _ = validate_data(data)
if errors:
    raise SystemExit(f"Data tidak valid: {errors}")
result = optimize_from_data(data)
kpi = hitung_kpi(result, data)

# 3) Analisis AI atas hasil solver
print("\n=== ANALISIS GEMINI ===")
print(ai.analisis_hasil(result, kpi, data))

# 4) Tanya jawab
print("\n=== TANYA JAWAB ===")
print(ai.tanya_rute("Rute mana yang utilisasinya paling rendah dan kenapa?", [],
                    ai.bangun_konteks(result, kpi, data)))

# 5) Tutor algoritma
print("\n=== TUTOR: 2-opt ===")
print(ai.jelaskan_konsep("Perbaikan lokal 2-opt", "Pemula", ai.bangun_konteks(result, kpi, data)))
