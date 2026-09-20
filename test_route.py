"""Uji cepat pipeline tanpa Streamlit: python test_route.py"""
from src.data_processing import load_data, validate_data, summary_table
from src.route_optimization import optimize_from_data
from src.kpi_analysis import hitung_kpi, tabel_kpi, tabel_per_kendaraan, temuan_otomatis

data = load_data("data/Data_Operasional_Rute.xlsx")
errors, warnings = validate_data(data)

print("\n=== ERRORS ===")
print(errors or "tidak ada")
print("\n=== WARNINGS ===")
print(warnings or "tidak ada")

if errors:
    raise ValueError("Data tidak valid. Perbaiki error terlebih dahulu.")

print("\n=== RINGKASAN DATA ===")
print(summary_table(data).to_string(index=False))

print("\n=== MENJALANKAN ROUTING & OPTIMIZATION ===")
result = optimize_from_data(data)

print("\n=== CANDIDATE SUMMARY ===")
print(result["candidate_summary"].to_string(index=False))

print("\n=== RINGKASAN PER RUTE ===")
print(tabel_per_kendaraan(result).to_string(index=False))

print("\n=== ROUTE TABLE ===")
for name, table in result["route_tables"].items():
    print(f"\n--- {name} ---")
    print(table[["stop_sequence", "node_id", "node_name", "jarak_dari_sebelumnya_km",
                 "tiba", "selesai", "muatan_turun_kg", "telat_menit", "status"]].to_string(index=False))

kpi = hitung_kpi(result, data)
print("\n=== KPI ===")
print(tabel_kpi(kpi).to_string(index=False))

print("\n=== TEMUAN ===")
for t in temuan_otomatis(kpi, result, data):
    print("-", t)
