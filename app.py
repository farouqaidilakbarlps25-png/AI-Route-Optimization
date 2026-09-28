"""
app.py
------
Aplikasi Streamlit untuk optimasi rute distribusi.
Terbuka untuk kasus apa pun (bukan hanya contoh Medan) -- pengguna bisa:
  1. Mengunggah file Excel sendiri (struktur bebas, mengikuti template), atau
  2. Mengisi data langsung di tabel dalam aplikasi (tambah/hapus baris), atau
  3. Memulai dari data contoh lalu mengubahnya.

Jalankan dari folder project:
    streamlit run app.py
"""
from __future__ import annotations

import io
from pathlib import Path

import pandas as pd
import plotly.express as px
import pydeck as pdk
import streamlit as st

from src.data_processing import (FileUploadError, contoh_kosong, load_data,
                                 load_data_from_frames, read_uploaded_excel,
                                 summary_table, validate_data)
from src.geocoding import geocode_missing
from src.kpi_analysis import hitung_kpi, tabel_kpi, tabel_per_kendaraan, temuan_otomatis
from src.route_optimization import optimize_from_data, polyline_rute_terpilih
from src import ai_assistant as ai

DATA_CONTOH = Path("data/Data_Operasional_Rute.xlsx")
WARNA = [[15, 118, 110], [200, 134, 43], [37, 84, 160], [176, 74, 58], [110, 70, 140],
         [88, 120, 60], [214, 90, 120], [90, 96, 110]]
WARNA_TAHAP = {"Sebelum": "#93C5FD", "Sesudah": "#1D4ED8"}  # biru muda, biru tua

# Citra satelit gratis tanpa API key (Esri World Imagery). Dipakai sebagai layer
# TileLayer paling bawah di peta bila pengguna menyalakan toggle "Latar satelit".
URL_SATELIT = ("https://server.arcgisonline.com/ArcGIS/rest/services/"
              "World_Imagery/MapServer/tile/{z}/{y}/{x}")

st.set_page_config(page_title="Rute Pintar — Optimasi Rute Distribusi", page_icon="🧭", layout="wide")


def muat_css() -> None:
    """Menyuntikkan tema dari assets/style.css (font, warna, kartu, tab, tombol)."""
    css = Path(__file__).parent / "assets" / "style.css"
    if css.exists():
        st.markdown(f"<style>{css.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)


muat_css()


# ==================================================================
# STATE
# ==================================================================
def _frame_kosong(nama: str) -> pd.DataFrame:
    return contoh_kosong()[nama].iloc[0:0].copy()


def init_state():
    """Aplikasi selalu dimulai tanpa data. Pengguna mengisi manual, mengunggah file,
    atau (opsional) memuat data contoh lewat tombol di tab Data & Input."""
    if "customers" in st.session_state:
        return
    template = contoh_kosong()
    st.session_state.customers = template["Customers"]
    st.session_state.vehicles = template["Vehicle"]
    st.session_state.depots = template["Depot"]
    st.session_state.params = {}
    st.session_state.sumber_data = "Belum ada data — silakan isi atau unggah"


init_state()
st.session_state.setdefault("result", None)
st.session_state.setdefault("kpi", None)
st.session_state.setdefault("polylines", None)
st.session_state.setdefault("tampilkan_satelit", False)
st.session_state.setdefault("editor_ver", 0)
st.session_state.setdefault("file_diproses", None)
st.session_state.setdefault("info_unggah", None)
st.session_state.setdefault("info_geo", None)
st.session_state.setdefault("chat_riwayat", [])
st.session_state.setdefault("saran_data", None)
for _k in ("tutor_penjelasan", "tutor_topik", "tutor_soal", "tutor_nilai"):
    st.session_state.setdefault(_k, None)


def data_sekarang() -> dict:
    """Membangun ulang data dari tabel + PARAMETER YANG SEDANG DIATUR DI SIDEBAR.

    Sebelumnya fungsi ini memanggil load_data_from_frames() tanpa params_df, sehingga
    params selalu kembali ke nilai bawaan (DEFAULT_PARAMS) dan slider di sidebar
    (faktor jarak, faktor lalu lintas, harga BBM, penalti keterlambatan) tidak
    benar-benar mempengaruhi hasil optimasi walau terlihat bisa digeser.
    """
    data = load_data_from_frames(
        st.session_state.customers, st.session_state.vehicles,
        st.session_state.depots, source_label=st.session_state.sumber_data)
    data["params"] = {**data["params"], **st.session_state.get("params", {})}
    return data


def segarkan_editor():
    """Membuat ulang semua st.data_editor.

    Editor menyimpan 'perubahan' (baris tambahan/edit) di state widget berdasarkan key-nya.
    Bila data diganti lewat kode (geocoding, unggah, reset) tanpa key baru, perubahan lama
    ditimpakan lagi ke data baru sehingga hasil geocoding tampak tidak masuk.
    """
    st.session_state.editor_ver += 1


def siapkan_koordinat(df: pd.DataFrame) -> pd.DataFrame:
    """Memastikan latitude/longitude bertipe angka; sel kosong/teks kosong menjadi NaN."""
    df = df.copy()
    for c in ("latitude", "longitude"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def ringkas_geocode(df: pd.DataFrame, label: str) -> str:
    sisa = int((df["latitude"].isna() | df["longitude"].isna()).sum())
    status = df["geocode_status"].value_counts().to_dict() if "geocode_status" in df.columns else {}
    rincian = ", ".join(f"{k}: {v}" for k, v in status.items() if k) or "-"
    return f"{label} selesai diproses ({rincian}). Baris yang masih tanpa koordinat: {sisa}."


def warna(i):
    return WARNA[i % len(WARNA)]


def template_excel_bytes() -> bytes:
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as xl:
        for nama, df in contoh_kosong().items():
            df.to_excel(xl, sheet_name=nama, index=False)
    return buf.getvalue()


def buat_layer_peta(result, nodes, polylines=None, tampilkan_satelit=False):
    layers = []
    if tampilkan_satelit:
        layers.append(pdk.Layer("TileLayer", data=URL_SATELIT,
                                min_zoom=0, max_zoom=19, tile_size=256))

    titik = nodes.copy()
    titik["radius"] = titik.node_type.map({"DEPOT": 320, "CUSTOMER": 170})
    titik["warna"] = titik.node_type.map({"DEPOT": [27, 36, 48], "CUSTOMER": [120, 130, 145]})
    layers.append(pdk.Layer(
        "ScatterplotLayer", data=titik, get_position=["longitude", "latitude"],
        get_radius="radius", get_fill_color="warna", pickable=True, opacity=0.85))

    koord = nodes.set_index("node_id")[["latitude", "longitude"]].to_dict("index")
    garis = []
    for i, (rid, tabel) in enumerate(result["route_tables"].items()):
        # bila polyline OSRM tersedia untuk rute ini, ikuti jalan sungguhan; bila tidak
        # (OSRM belum dihitung / gagal), tetap jatuh ke garis lurus antar stop (perilaku lama)
        if polylines and rid in polylines:
            path = polylines[rid]["geometry"]
        else:
            path = [[koord[n]["longitude"], koord[n]["latitude"]] for n in tabel.node_id]
        garis.append({"route_id": rid, "path": path, "warna": warna(i),
                      "vehicle_id": tabel.vehicle_id.iloc[0]})
    if garis:
        layers.append(pdk.Layer("PathLayer", data=garis, get_path="path",
                                get_color="warna", width_min_pixels=3, pickable=True))
    return layers


def render_perbandingan_chart(h: dict, key_prefix: str = "") -> None:
    """Grafik batang Jarak (km) & Biaya (Rp): sebelum vs sesudah, berdampingan.

    key_prefix wajib diisi berbeda tiap kali fungsi ini dipanggil di halaman yang sama
    (mis. "rute", "kpi"), supaya masing-masing st.plotly_chart punya ID unik dan tidak
    bentrok dengan StreamlitDuplicateElementId saat fungsi ini dipanggil lebih dari
    sekali dengan data yang identik.
    """
    col_jarak, col_biaya = st.columns(2)
    with col_jarak:
        st.markdown("**Jarak (km)**")
        df_jarak = pd.DataFrame({
            "Tahap": ["Sebelum", "Sesudah"],
            "Nilai": [h["jarak_baseline_km"], h["jarak_optimal_km"]],
        })
        fig = px.bar(df_jarak, x="Tahap", y="Nilai", color="Tahap",
                     color_discrete_map=WARNA_TAHAP, text_auto=".2f")
        fig.update_layout(showlegend=False, yaxis_title="km", xaxis_title="")
        st.plotly_chart(fig, use_container_width=True, key=f"{key_prefix}_chart_jarak")
    with col_biaya:
        st.markdown("**Biaya (Rp)**")
        df_biaya = pd.DataFrame({
            "Tahap": ["Sebelum", "Sesudah"],
            "Nilai": [h["biaya_baseline_idr"], h["biaya_optimal_idr"]],
        })
        fig = px.bar(df_biaya, x="Tahap", y="Nilai", color="Tahap",
                     color_discrete_map=WARNA_TAHAP, text_auto=",.0f")
        fig.update_layout(showlegend=False, yaxis_title="Rp", xaxis_title="")
        st.plotly_chart(fig, use_container_width=True, key=f"{key_prefix}_chart_biaya")


# ==================================================================
# SIDEBAR - PARAMETER
# ==================================================================
st.sidebar.title("🚚 Optimasi Rute")
st.sidebar.info(f"Sumber data saat ini: **{st.session_state.sumber_data}**")

st.sidebar.subheader("Parameter")
p = st.session_state.params or {}
p["detour_factor"] = st.sidebar.slider(
    "Faktor jarak jalan", 1.0, 2.0, float(p.get("detour_factor", 1.32)), 0.01,
    help="Pengali manual (bukan data GPS/peta real-time) untuk mengubah jarak garis lurus "
         "menjadi perkiraan jarak jalan sebenarnya. Sesuaikan sendiri: 1.2-1.3 untuk kota "
         "dengan jalan relatif lurus, 1.4-1.6 untuk area berkelok/banyak satu arah.")
p["traffic_factor"] = st.sidebar.slider(
    "Faktor lalu lintas", 1.0, 2.5, float(p.get("traffic_factor", 1.18)), 0.01,
    help="Pengali manual (bukan data lalu lintas real-time) untuk memperlambat waktu tempuh "
         "dibanding kondisi jalan lengang. Naikkan untuk simulasi jam sibuk/macet, turunkan "
         "mendekati 1.0 untuk simulasi malam hari/lengang.")
p["harga_solar_idr_per_liter"] = st.sidebar.number_input(
    "Harga BBM (Rp/liter)", 0, 50000, int(p.get("harga_solar_idr_per_liter", 6800)), 100)
p["penalti_keterlambatan_per_menit"] = st.sidebar.number_input(
    "Penalti keterlambatan (Rp/menit)", 0, 500000,
    int(p.get("penalti_keterlambatan_per_menit", 15000)), 1000)

# Rute selalu memakai OSRM (jalan sungguhan) — pengguna tidak perlu menyalakan/mematikannya,
# hanya memilih sumber datanya: online (internet) atau cache lokal saja.
p["pakai_rute_jalan_asli"] = True
st.sidebar.subheader("🛣️ Rute jalan asli (OSRM)")
p["rute_online"] = st.sidebar.checkbox(
    "Ambil dari internet (matikan untuk pakai cache lokal saja)",
    value=bool(p.get("rute_online", True)), key="rute_online_cb",
    help="Rute selalu mengikuti jalan sungguhan (OSRM). Nyalakan untuk mengambil data "
         "rute terbaru dari internet; matikan untuk memakai cache lokal saja — berguna "
         "saat tidak ada koneksi internet.")
st.sidebar.caption("Bila OSRM tidak terjangkau, aplikasi otomatis jatuh ke estimasi garis "
                   "lurus supaya tetap bisa jalan.")

st.session_state.tampilkan_satelit = st.sidebar.checkbox(
    "🛰️ Latar peta citra satelit", value=st.session_state.tampilkan_satelit,
    help="Esri World Imagery — gratis, tanpa API key. Ditampilkan di tab Peta.")
st.session_state.params = p

# Key dari Streamlit Secrets (bila belum ada di environment)
try:
    if not ai.api_key_tersedia():
        for _nama in ("AI_API_KEY", "GEMINI_API_KEY", "GOOGLE_API_KEY"):
            if _nama in st.secrets:
                ai.atur_kunci_sesi(str(st.secrets[_nama]))
                break
except Exception:  # noqa: BLE001  (tidak ada file secrets saat dijalankan lokal)
    pass
if st.session_state.get("kunci_sesi"):
    ai.atur_kunci_sesi(st.session_state["kunci_sesi"])

st.sidebar.subheader("🤖 Asisten AI")
if ai.api_key_tersedia():
    st.sidebar.success("AI siap digunakan")
else:
    st.sidebar.warning("Fitur AI belum aktif untuk sesi ini.")

st.markdown(
    '<div class="hero">'
    '<div class="eyebrow">Logistics Intelligence</div>'
    '<h1>RutePintar</h1>'
    '<p>Rencanakan rute pengiriman Anda secara otomatis — lebih hemat, lebih cepat, dan tepat waktu.</p>'
    '</div>', unsafe_allow_html=True)

tab_data, tab_peta, tab_rute, tab_kpi, tab_ai, tab_tutor, tab_ekspor = st.tabs(
    ["📋 Data & Input", "🗺️ Peta", "🛣️ Rute", "📊 KPI", "🤖 Asisten AI",
     "🎓 Tutor Algoritma", "💾 Ekspor"])

# ==================================================================
# TAB 1 — DATA & INPUT
# ==================================================================
with tab_data:
    st.subheader("1. Pilih sumber data")
    st.download_button(
        "⬇️ Unduh template Excel kosong", template_excel_bytes(),
        file_name="template_data_rute.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        help="Berisi 4 sheet (Customers, Vehicle, Depot, Parameters) dengan satu baris "
             "contoh dan header baku. Isi sesuai data Anda sendiri lalu unggah kembali.")

    col1, col2 = st.columns(2)
    with col1:
        unggah = st.file_uploader("Unggah file Excel Anda (.xlsx)", type=["xlsx"])
        if unggah is None:
            st.session_state.file_diproses = None
        else:
            id_file = f"{unggah.name}:{unggah.size}"
            # file_uploader tetap 'berisi file' di setiap rerun. Tanpa penanda ini, data hasil
            # geocoding/edit langsung ditimpa lagi oleh isi file asli setiap halaman dimuat ulang.
            if st.session_state.file_diproses != id_file:
                try:
                    mentah = read_uploaded_excel(unggah.getvalue())
                    bersih = load_data_from_frames(
                        mentah["Customers"], mentah["Vehicle"], mentah["Depot"],
                        mentah["Parameters"], source_label=f"Unggahan: {unggah.name}")
                    st.session_state.customers = bersih["customers"]
                    st.session_state.vehicles = bersih["vehicles"]
                    st.session_state.depots = bersih["depots"]
                    st.session_state.params = bersih["params"]
                    st.session_state.sumber_data = f"Unggahan: {unggah.name}"
                    st.session_state.result = None
                    st.session_state.file_diproses = id_file
                    st.session_state.info_unggah = (
                        f"File '{unggah.name}' berhasil dimuat: {len(bersih['customers'])} pelanggan, "
                        f"{len(bersih['vehicles'])} kendaraan, {len(bersih['depots'])} depot.")
                    peta_kolom = [c for c in bersih.get("catatan_default", [])
                                  if c.startswith(("Pemetaan kolom", "Vehicle: "))
                                  and ("→" in c or "tidak diikutkan" in c)]
                    if peta_kolom:
                        st.session_state.info_unggah += "\n\n" + "\n\n".join("• " + c for c in peta_kolom)
                    segarkan_editor()
                    st.rerun()
                except FileUploadError as e:
                    st.session_state.file_diproses = None
                    st.error(f"⚠️ File tidak sesuai: {e}")
                except Exception as e:  # noqa: BLE001  (error tak terduga lain)
                    st.session_state.file_diproses = None
                    st.error(f"Gagal membaca file: {e}. Pastikan strukturnya mengikuti template.")
            elif st.session_state.info_unggah:
                st.success(st.session_state.info_unggah)
    with col2:
        if st.button("🗑️ Mulai dari kosong (input manual dari nol)"):
            st.session_state.customers = _frame_kosong("Customers")
            st.session_state.vehicles = _frame_kosong("Vehicle")
            st.session_state.depots = _frame_kosong("Depot")
            st.session_state.params = {}
            st.session_state.sumber_data = "Input manual"
            st.session_state.result = None
            st.session_state.info_unggah = None
            segarkan_editor()
            st.rerun()
        if DATA_CONTOH.exists() and st.button("↩️ Muat ulang data contoh"):
            d = load_data(DATA_CONTOH)
            st.session_state.customers = d["customers"]
            st.session_state.vehicles = d["vehicles"]
            st.session_state.depots = d["depots"]
            st.session_state.params = d["params"]
            st.session_state.sumber_data = f"Data contoh ({DATA_CONTOH.name})"
            st.session_state.result = None
            st.session_state.info_unggah = None
            segarkan_editor()
            st.rerun()

    st.divider()
    st.subheader("2. Isi atau edit data langsung di tabel")
    st.caption("Klik sel untuk mengubah, klik baris kosong paling bawah untuk menambah, "
               "atau tandai baris lalu tekan tombol hapus (ikon tempat sampah) untuk menghapus.")

    st.markdown("**Pelanggan (Customers)**")
    st.session_state.customers = st.data_editor(
        st.session_state.customers, num_rows="dynamic", use_container_width=True,
        key=f"editor_customers_{st.session_state.editor_ver}",
        column_config={
            "demand_kg": st.column_config.NumberColumn("demand_kg", min_value=0),
            "latitude": st.column_config.NumberColumn("latitude", format="%.6f"),
            "longitude": st.column_config.NumberColumn("longitude", format="%.6f"),
            "priority_level": st.column_config.NumberColumn("priority_level", min_value=1, max_value=5),
        })

    gcol1, gcol2, gcol3 = st.columns([1, 1, 2])
    online = gcol1.checkbox("Pakai internet (Nominatim OSM)", value=True,
                            help="Matikan bila tidak ada koneksi internet; akan memakai tabel "
                                 "perkiraan lokal (saat ini hanya mencakup Kota Medan).")
    if gcol2.button("📍 Isi koordinat otomatis"):
        df_c = siapkan_koordinat(st.session_state.customers)
        n_kosong = int(df_c["latitude"].isna().sum()) if "latitude" in df_c else len(df_c)
        if n_kosong == 0:
            st.info("Semua pelanggan sudah punya koordinat.")
        else:
            progres = st.progress(0.0, text="Memulai geocoding...")

            def _cb(i, n, label):
                progres.progress(i / max(n, 1), text=f"({i}/{n}) {label[:60]}")

            hasil_geo = siapkan_koordinat(geocode_missing(
                df_c, use_online=online, progress_cb=_cb))
            progres.empty()
            st.session_state.customers = hasil_geo
            st.session_state.info_geo = ringkas_geocode(hasil_geo, "Pelanggan")
            segarkan_editor()
            st.rerun()
    gcol3.caption("Geocoding memakai Nominatim OpenStreetMap (gratis, tanpa API key). "
                 "Untuk kota di luar Medan, koreksi manual di kolom latitude/longitude bila hasilnya kurang tepat.")

    if st.session_state.info_geo:
        st.info("📍 " + st.session_state.info_geo)
    st.markdown("**Kendaraan (Vehicle)**")
    st.session_state.vehicles = st.data_editor(
        st.session_state.vehicles, num_rows="dynamic", use_container_width=True,
        key=f"editor_vehicles_{st.session_state.editor_ver}",
        column_config={
            "capacity_kg": st.column_config.NumberColumn("capacity_kg", min_value=0),
            "avg_speed_kmh": st.column_config.NumberColumn("avg_speed_kmh", min_value=1),
        })

    st.markdown("**Depot**")
    st.session_state.depots = st.data_editor(
        st.session_state.depots, num_rows="dynamic", use_container_width=True,
        key=f"editor_depots_{st.session_state.editor_ver}",
        column_config={
            "latitude": st.column_config.NumberColumn("latitude", format="%.6f"),
            "longitude": st.column_config.NumberColumn("longitude", format="%.6f"),
        })

    if st.button("📍 Isi koordinat otomatis (Depot)"):
        df_d = siapkan_koordinat(st.session_state.depots)
        progres = st.progress(0.0, text="Memulai geocoding depot...")

        def _cb2(i, n, label):
            progres.progress(i / max(n, 1), text=f"({i}/{n}) {label[:60]}")

        hasil_geo = siapkan_koordinat(geocode_missing(df_d, use_online=online, progress_cb=_cb2))
        progres.empty()
        st.session_state.depots = hasil_geo
        st.session_state.info_geo = ringkas_geocode(hasil_geo, "Depot")
        segarkan_editor()
        st.rerun()

    st.divider()
    st.subheader("3. Periksa data")
    data = data_sekarang()
    errors, warnings_ = validate_data(data)

    if not data["customers"].empty:
        st.dataframe(summary_table(data), hide_index=True, use_container_width=True)

    if errors:
        st.error("Data belum lengkap/valid — optimasi belum bisa dijalankan:")
        for e in errors:
            st.write("- ", e)
        if st.button("🤖 Minta AI menjelaskan & memperbaiki", key="ai_fix_data"):
            with st.spinner("AI menganalisis masalah data..."):
                try:
                    st.session_state.saran_data = ai.bantu_perbaiki_data(errors, warnings_, data)
                except ai.AIError as err:
                    st.error(str(err))
        if st.session_state.saran_data:
            st.info(st.session_state.saran_data)
    else:
        if warnings_:
            st.warning("Catatan (optimasi tetap bisa dijalankan):")
            for w in warnings_:
                st.write("- ", w)
        else:
            st.success("Data lengkap dan valid.")

        st.divider()
        if st.button("🚀 Jalankan Optimasi", type="primary", use_container_width=True):
            with st.spinner("Menghitung rute..."):
                st.session_state.result = optimize_from_data(data)
                st.session_state.kpi = hitung_kpi(st.session_state.result, data)
                st.session_state.chat_riwayat = []
                st.session_state.polylines = None
                if p.get("pakai_rute_jalan_asli", False):
                    with st.spinner("Mengambil bentuk rute mengikuti jalan dari OSRM..."):
                        st.session_state.polylines = polyline_rute_terpilih(
                            st.session_state.result["routes"], st.session_state.result["nodes"],
                            use_online=bool(p.get("rute_online", True)))
            st.success("Optimasi selesai. Buka tab Peta, Rute, atau KPI untuk melihat hasilnya.")

result = st.session_state.result
kpi = st.session_state.kpi

if result is None:
    for tab in (tab_peta, tab_rute, tab_kpi, tab_ai, tab_ekspor):
        with tab:
            st.info("Belum ada hasil. Lengkapi data pada tab **Data & Input**, lalu tekan "
                   "tombol **🚀 Jalankan Optimasi**.")
else:
    data = data_sekarang()
    h = result["penghematan"]

    # ---------------------------------------------------------------- peta
    with tab_peta:
        nodes = result["nodes"]
        polylines = st.session_state.polylines
        st.pydeck_chart(pdk.Deck(
            map_style=None,
            initial_view_state=pdk.ViewState(
                latitude=float(p.get("pusat_peta_lat", nodes.latitude.mean())),
                longitude=float(p.get("pusat_peta_lon", nodes.longitude.mean())),
                zoom=10.5, pitch=0),
            layers=buat_layer_peta(result, nodes, polylines, st.session_state.tampilkan_satelit),
            tooltip={"text": "{node_name}{route_id} {vehicle_id}"}))
        st.caption("Titik gelap = depot, titik abu-abu = pelanggan, garis berwarna = rute kendaraan.")

        sumber_jarak = result.get("sumber_jarak", "Estimasi (garis lurus x faktor jarak jalan)")
        if sumber_jarak.startswith("OSRM"):
            st.caption(f"📍 Jarak & bentuk rute mengikuti jalan sungguhan (sumber: {sumber_jarak}).")
        else:
            st.caption(f"📍 Sumber jarak saat ini: {sumber_jarak}.")
        if p.get("pakai_rute_jalan_asli", False) and polylines:
            sumber_polyline = {v["sumber"] for v in polylines.values()}
            if any(s.startswith("Fallback") for s in sumber_polyline):
                st.warning("Sebagian bentuk rute di peta jatuh ke garis lurus (OSRM tidak "
                          "terjangkau untuk rute tersebut saat itu). Jalankan ulang optimasi "
                          "saat koneksi lebih stabil bila perlu bentuk jalan yang akurat.")

        legenda = pd.DataFrame([
            {"Rute": r["route_id"], "Kendaraan": r["vehicle_id"], "Depot": r["depot_id"],
             "Stop": r["jumlah_stop"], "Jarak (km)": r["total_jarak_km"],
             "Selesai": r["selesai_pukul"]} for r in result["routes"]])
        st.dataframe(legenda, hide_index=True, use_container_width=True)

    # ---------------------------------------------------------------- rute
    with tab_rute:
        st.subheader("Perbandingan kandidat solusi")
        st.dataframe(result["candidate_summary"], hide_index=True, use_container_width=True)

        st.success(
            f"✅ **Keputusan rute terbaik: kandidat '{result['best_candidate']}'.** "
            f"Dipilih karena skor objektifnya (total biaya + penalti pelanggan tak terlayani) "
            f"paling rendah di antara semua kandidat yang diuji. Dibandingkan solusi awal "
            f"sebelum perbaikan lokal (baris **Savings murni** di tabel di atas — "
            f"ini yang dipakai sebagai pembanding **'sebelum'**), rute ini (**'sesudah'**) "
            f"lebih hemat **{h['hemat_km']:.2f} km ({h['hemat_jarak_persen']:.1f}%)** jarak dan "
            f"**Rp {h['hemat_biaya_idr']:,.0f}** biaya.")

        st.markdown("**Perbandingan sebelum vs sesudah optimasi**")
        k1, k2, k3 = st.columns(3)
        k1.metric("Jarak — sebelum → sesudah",
                  f"{h['jarak_optimal_km']:,.2f} km",
                  f"{-h['hemat_km']:+,.2f} km ({-h['hemat_jarak_persen']:+.1f}%)",
                  delta_color="inverse",
                  help=f"Sebelum (Savings murni): {h['jarak_baseline_km']:,.2f} km. "
                       f"Sesudah ({result['best_candidate']}): {h['jarak_optimal_km']:,.2f} km.")
        k2.metric("Biaya — sebelum → sesudah",
                  f"Rp {h['biaya_optimal_idr']:,.0f}",
                  f"{-h['hemat_biaya_idr']:+,.0f}", delta_color="inverse",
                  help=f"Sebelum: Rp {h['biaya_baseline_idr']:,.0f}. "
                       f"Sesudah: Rp {h['biaya_optimal_idr']:,.0f}.")
        k3.metric("Rute terbentuk", f"{len(result['routes'])}")

        with st.expander("Lihat perbandingan jarak & biaya sebelum vs sesudah sebagai grafik",
                          expanded=True):
            render_perbandingan_chart(h, key_prefix="rute")

        if result["unserved"]:
            st.warning(f"Belum terlayani: {', '.join(result['unserved'])}")

        st.subheader("Rincian per rute")
        for rid, tabel in result["route_tables"].items():
            ringkas = next(r for r in result["routes"] if r["route_id"] == rid)
            with st.expander(
                    f"{rid} · {ringkas['vehicle_id']} · {ringkas['jumlah_stop']} stop · "
                    f"{ringkas['total_jarak_km']:.2f} km · utilisasi {ringkas['utilisasi']:.0%}",
                    expanded=(rid == list(result["route_tables"])[0])):
                st.dataframe(tabel, hide_index=True, use_container_width=True)

    # ---------------------------------------------------------------- kpi
    with tab_kpi:
        c = st.columns(4)
        c[0].metric("Total jarak", f"{kpi['total_jarak_km']:,.1f} km")
        c[1].metric("Total biaya", f"Rp {kpi['total_biaya_idr']:,.0f}")
        c[2].metric("Ketepatan waktu", f"{kpi['on_time_rate']:.0%}")
        c[3].metric("Utilisasi rata-rata", f"{kpi['rata_utilisasi']:.0%}")

        c = st.columns(4)
        c[0].metric("Pelanggan terlayani", f"{kpi['pelanggan_terlayani']}/{len(data['customers'])}")
        c[1].metric("Kendaraan dipakai", f"{kpi['kendaraan_dipakai']}/{len(data['vehicles'])}")
        c[2].metric("Konsumsi BBM", f"{kpi['total_bbm_liter']:,.1f} L")
        c[3].metric("Emisi CO₂", f"{kpi['total_emisi_co2_kg']:,.1f} kg")

        kiri, kanan = st.columns([1, 1])
        with kiri:
            st.subheader("Tabel KPI")
            st.dataframe(tabel_kpi(kpi), hide_index=True, use_container_width=True)
        with kanan:
            st.subheader("Temuan")
            for t in temuan_otomatis(kpi, result, data):
                st.write("- ", t)
            st.subheader("Jarak per rute")
            st.bar_chart(tabel_per_kendaraan(result).set_index("route_id")[["total_jarak_km"]],
                         color="#0F766E")

        st.subheader("Perbandingan sebelum vs sesudah optimasi")
        render_perbandingan_chart(h, key_prefix="kpi")

        st.subheader("Ringkasan per kendaraan")
        st.dataframe(tabel_per_kendaraan(result), hide_index=True, use_container_width=True)

    # ------------------------------------------------------------- asisten AI
    with tab_ai:
        if not ai.api_key_tersedia():
            st.warning("Fitur AI belum aktif untuk sesi ini.")
        else:
            st.subheader("Tanya jawab tentang rute Anda")
            st.caption("Contoh: “Kenapa RT-02 terlambat?”, “Rute mana yang paling boros?”, "
                       "“Apa yang terjadi kalau jam berangkat dimajukan?”")
            for m in st.session_state.chat_riwayat:
                with st.chat_message(m["role"]):
                    st.markdown(m["content"])
            if tanya := st.chat_input("Tanyakan sesuatu tentang hasil optimasi..."):
                st.session_state.chat_riwayat.append({"role": "user", "content": tanya})
                with st.spinner("AI berpikir..."):
                    try:
                        jawab = ai.tanya_rute(tanya, st.session_state.chat_riwayat[:-1],
                                              ai.bangun_konteks(result, kpi, data))
                    except ai.AIError as err:
                        jawab = f"⚠️ {err}"
                st.session_state.chat_riwayat.append({"role": "assistant", "content": jawab})
                st.rerun()
            if st.session_state.chat_riwayat and st.button("🧹 Hapus percakapan"):
                st.session_state.chat_riwayat = []
                st.rerun()

    # ------------------------------------------------------------- ekspor
    with tab_ekspor:
        st.write("Unduh hasil perencanaan untuk dibagikan ke tim operasional.")
        buffer = io.BytesIO()
        with pd.ExcelWriter(buffer, engine="openpyxl") as xl:
            tabel_kpi(kpi).to_excel(xl, sheet_name="KPI", index=False)
            result["candidate_summary"].to_excel(xl, sheet_name="Perbandingan_Kandidat", index=False)
            tabel_per_kendaraan(result).to_excel(xl, sheet_name="Ringkasan_Rute", index=False)
            result["stops"].to_excel(xl, sheet_name="Detail_Stop", index=False)
            result["distance_matrix"].to_excel(xl, sheet_name="Matriks_Jarak")
        st.download_button("⬇️ Unduh hasil optimasi (.xlsx)", buffer.getvalue(),
                           file_name="hasil_optimasi_rute.xlsx",
                           mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
        st.download_button("⬇️ Unduh detail stop (.csv)",
                           result["stops"].to_csv(index=False).encode("utf-8"),
                           file_name="detail_stop.csv", mime="text/csv")
        st.dataframe(result["stops"], hide_index=True, use_container_width=True)


# ==================================================================
# TAB TUTOR ALGORITMA — AI sebagai media belajar
# ==================================================================
with tab_tutor:
    st.subheader("🎓 Tutor Algoritma")
    st.caption("Pilih konsep. AI menjelaskan memakai kode asli di `src/route_optimization.py` "
               "dan, bila optimasi sudah dijalankan, angka dari hasil Anda sendiri.")
    if not ai.api_key_tersedia():
        st.warning("Fitur AI belum aktif untuk sesi ini.")
    else:
        c1, c2 = st.columns([2, 1])
        topik = c1.selectbox("Konsep", list(ai.KONSEP))
        level = c2.selectbox("Level", ["Pemula", "Menengah", "Lanjut"])

        if st.button("📖 Jelaskan", key="btn_tutor"):
            with st.spinner("AI menyiapkan penjelasan..."):
                try:
                    konteks = (ai.bangun_konteks(result, kpi, data_sekarang())
                               if result is not None else None)
                    st.session_state.tutor_penjelasan = ai.jelaskan_konsep(topik, level, konteks)
                    st.session_state.tutor_topik = topik
                    st.session_state.tutor_soal = None
                    st.session_state.tutor_nilai = None
                except ai.AIError as err:
                    st.error(str(err))

        if st.session_state.tutor_penjelasan:
            st.markdown(f"#### {st.session_state.tutor_topik}")
            st.markdown(st.session_state.tutor_penjelasan)

            st.divider()
            st.markdown("**Uji pemahaman**")
            if st.button("📝 Buat pertanyaan latihan", key="btn_soal"):
                with st.spinner("AI membuat pertanyaan..."):
                    try:
                        st.session_state.tutor_soal = ai.buat_pertanyaan_latihan(
                            st.session_state.tutor_topik, level)
                        st.session_state.tutor_nilai = None
                    except ai.AIError as err:
                        st.error(str(err))
            if st.session_state.tutor_soal:
                st.info(st.session_state.tutor_soal)
                jawaban = st.text_area("Jawaban Anda", key="tutor_jawaban")
                if st.button("✅ Nilai jawaban saya", key="btn_nilai") and jawaban.strip():
                    with st.spinner("AI menilai..."):
                        try:
                            st.session_state.tutor_nilai = ai.nilai_jawaban(
                                st.session_state.tutor_topik, st.session_state.tutor_soal, jawaban)
                        except ai.AIError as err:
                            st.error(str(err))
                if st.session_state.tutor_nilai:
                    st.markdown(st.session_state.tutor_nilai)