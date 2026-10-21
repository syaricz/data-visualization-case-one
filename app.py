"""
Langkah 3 - Dashboard Streamlit: Dataset Operasional Penjualan (Bedah Kasus 1)
Jalankan:  streamlit run app.py
"""
import os
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from scipy import stats

st.set_page_config(page_title="Dashboard Penjualan", page_icon="📊", layout="wide")

rp = lambda x: f"Rp{x:,.0f}".replace(",", ".")
def singkat(x):
    for batas, s in [(1e12, "T"), (1e9, "M"), (1e6, "jt")]:
        if abs(x) >= batas:
            return f"Rp{x/batas:,.1f} {s}".replace(",", "X").replace(".", ",").replace("X", ".")
    return rp(x)

# ----------------------------------------------------------------- DATA
@st.cache_data
def load(path="data_penjualan.csv"):
    if not os.path.exists(path):
        import subprocess, sys
        subprocess.run([sys.executable, "1_generate_data.py"], check=True)
    d = pd.read_csv(path, parse_dates=["Tanggal"])
    d["Omzet"] = d["Harga (Rp)"] * d["Jumlah Terjual"]
    d["Bulan"] = d["Tanggal"].dt.to_period("M").dt.to_timestamp()
    d["Hari"] = d["Tanggal"].dt.day_name()
    return d

df = load()

# ----------------------------------------------------------------- SIDEBAR
st.sidebar.header("🔎 Filter Data")
tgl = st.sidebar.date_input("Rentang tanggal", (df["Tanggal"].min().date(), df["Tanggal"].max().date()),
                            min_value=df["Tanggal"].min().date(), max_value=df["Tanggal"].max().date())
kat = st.sidebar.multiselect("Kategori", sorted(df["Kategori"].unique()), default=sorted(df["Kategori"].unique()))
prod_opsi = sorted(df[df["Kategori"].isin(kat)]["Produk"].unique())
prod = st.sidebar.multiselect("Produk", prod_opsi, default=prod_opsi)
hmin, hmax = int(df["Harga (Rp)"].min()), int(df["Harga (Rp)"].max())
rentang_harga = st.sidebar.slider("Rentang harga (Rp)", hmin, hmax, (hmin, hmax), step=1000)

if len(tgl) != 2:
    st.info("Pilih tanggal awal dan akhir.")
    st.stop()

f = df[(df["Tanggal"].dt.date >= tgl[0]) & (df["Tanggal"].dt.date <= tgl[1]) &
       (df["Kategori"].isin(kat)) & (df["Produk"].isin(prod)) &
       (df["Harga (Rp)"].between(*rentang_harga))]

st.caption("Nama: Syamsurizal NIM: 241051004")
st.title("📊 Dashboard Operasional Penjualan")
st.caption("Bedah Kasus 1 · Data sintetis 1.000 transaksi (dikembangkan dari 2 data contoh: Laptop & Buku)")


if f.empty:
    st.warning("Tidak ada data untuk filter yang dipilih.")
    st.stop()

# ----------------------------------------------------------------- KPI
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Total Omzet", singkat(f["Omzet"].sum()))
c2.metric("Unit Terjual", f"{f['Jumlah Terjual'].sum():,}".replace(",", "."))
c3.metric("Jumlah Transaksi", f"{len(f):,}".replace(",", "."))
c4.metric("Rata-rata Omzet/Transaksi", singkat(f["Omzet"].mean()))
c5.metric("Harga Rata-rata/Unit", singkat(f["Omzet"].sum() / f["Jumlah Terjual"].sum()))

tab1, tab2, tab3, tab4, tab5 = st.tabs(["🏷️ Kategorik (Nominal)", "💰 Harga (Kontinu)", "🔢 Jumlah (Diskrit)",
                                         "📈 Tren Waktu", "🧪 Analisis Statistik"])

# ----------------------------------------------------------------- TAB 1: KATEGORIK
with tab1:
    a, b = st.columns(2)
    k = f.groupby("Kategori").agg(Transaksi=("ID", "count"), Omzet=("Omzet", "sum")).reset_index()
    a.plotly_chart(px.bar(k.sort_values("Transaksi"), x="Transaksi", y="Kategori", orientation="h", text="Transaksi",
                          color="Kategori", title="Transaksi per Kategori (Bar Chart)").update_layout(showlegend=False),
                   use_container_width=True)
    b.plotly_chart(px.pie(k, names="Kategori", values="Omzet", hole=.45,
                          title="Kontribusi Omzet per Kategori (Pie/Donut Chart)"), use_container_width=True)
    p = f.groupby(["Produk", "Kategori"]).agg(Transaksi=("ID", "count"), Unit=("Jumlah Terjual", "sum"),
                                              Omzet=("Omzet", "sum")).reset_index().sort_values("Omzet", ascending=False)
    p["Kumulatif (%)"] = p["Omzet"].cumsum() / p["Omzet"].sum() * 100
    a, b = st.columns(2)
    a.plotly_chart(px.bar(p.sort_values("Unit"), x="Unit", y="Produk", orientation="h", color="Kategori",
                          title="Unit Terjual per Produk"), use_container_width=True)
    pareto = go.Figure()
    pareto.add_bar(x=p["Produk"], y=p["Omzet"], name="Omzet")
    pareto.add_scatter(x=p["Produk"], y=p["Kumulatif (%)"], name="Kumulatif %", yaxis="y2", mode="lines+markers")
    pareto.update_layout(title="Pareto Omzet per Produk", yaxis2=dict(overlaying="y", side="right", range=[0, 105]),
                         legend=dict(orientation="h"))
    b.plotly_chart(pareto, use_container_width=True)
    st.markdown("**Ringkasan per produk**")
    st.dataframe(p.style.format({"Omzet": "{:,.0f}", "Unit": "{:,.0f}", "Kumulatif (%)": "{:.1f}"}), use_container_width=True, hide_index=True)

# ----------------------------------------------------------------- TAB 2: HARGA
with tab2:
    a, b = st.columns(2)
    a.plotly_chart(px.histogram(f, x="Harga (Rp)", nbins=40, marginal="box", title="Histogram Harga"), use_container_width=True)
    b.plotly_chart(px.histogram(f, x=np.log10(f["Harga (Rp)"]), nbins=40, labels={"x": "log10(Harga)"},
                                title="Histogram Harga (skala log10) - lebih simetris"), use_container_width=True)
    st.plotly_chart(px.box(f, x="Harga (Rp)", y="Produk", color="Kategori", log_x=True,
                           title="Sebaran Harga per Produk (Boxplot, skala log)"), use_container_width=True)
    d = f["Harga (Rp)"].describe().to_frame().T
    d["skewness"] = f["Harga (Rp)"].skew(); d["CV (%)"] = f["Harga (Rp)"].std() / f["Harga (Rp)"].mean() * 100
    st.dataframe(d.round(2), use_container_width=True)

# ----------------------------------------------------------------- TAB 3: JUMLAH
with tab3:
    a, b = st.columns(2)
    a.plotly_chart(px.histogram(f, x="Jumlah Terjual", nbins=30, color="Kategori", title="Distribusi Jumlah Terjual"),
                   use_container_width=True)
    b.plotly_chart(px.box(f, x="Kategori", y="Jumlah Terjual", color="Kategori", title="Jumlah Terjual per Kategori"),
                   use_container_width=True)
    st.plotly_chart(px.scatter(f, x="Harga (Rp)", y="Jumlah Terjual", color="Kategori", log_x=True, hover_data=["Produk"],
                               opacity=.65, title="Harga vs Jumlah Terjual (tiap titik = 1 transaksi)"), use_container_width=True)

# ----------------------------------------------------------------- TAB 4: TREN
with tab4:
    metrik = st.radio("Metrik", ["Omzet", "Jumlah Terjual", "Transaksi"], horizontal=True)
    gran = st.radio("Granularitas", ["Bulanan", "Mingguan", "Harian (MA 7 hari)"], horizontal=True)
    kol = {"Omzet": ("Omzet", "sum"), "Jumlah Terjual": ("Jumlah Terjual", "sum"), "Transaksi": ("ID", "count")}[metrik]
    if gran == "Bulanan":
        t = f.groupby("Bulan")[kol[0]].agg(kol[1])
    elif gran == "Mingguan":
        t = f.set_index("Tanggal")[kol[0]].resample("W").agg(kol[1])
    else:
        t = f.set_index("Tanggal")[kol[0]].resample("D").agg(kol[1]).rolling(7, min_periods=1).mean()
    t = t.rename(metrik).reset_index().rename(columns={"index": "Tanggal", "Bulan": "Tanggal"})
    fig = px.line(t, x="Tanggal", y=metrik, markers=(gran != "Harian (MA 7 hari)"), title=f"Tren {metrik} - {gran} (Line Chart)")
    if len(t) > 2:
        reg = stats.linregress(np.arange(len(t)), t[metrik])
        fig.add_scatter(x=t["Tanggal"], y=reg.intercept + reg.slope * np.arange(len(t)), mode="lines",
                        line=dict(dash="dash", color="gray"), name=f"Tren linear (R²={reg.rvalue**2:.2f})")
    st.plotly_chart(fig, use_container_width=True)

    a, b = st.columns(2)
    stack = f.groupby(["Bulan", "Kategori"])["Omzet"].sum().reset_index()
    a.plotly_chart(px.area(stack, x="Bulan", y="Omzet", color="Kategori", title="Omzet Bulanan per Kategori"), use_container_width=True)
    urut = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    h = f.groupby("Hari")["Omzet"].mean().reindex(urut).reset_index()
    b.plotly_chart(px.bar(h, x="Hari", y="Omzet", title="Rata-rata Omzet per Transaksi menurut Hari"), use_container_width=True)

    hm = f.assign(Minggu=f["Tanggal"].dt.isocalendar().week.astype(int), Bln=f["Tanggal"].dt.month) \
          .pivot_table(index="Kategori", columns="Bln", values="Omzet", aggfunc="sum", fill_value=0)
    st.plotly_chart(px.imshow(hm, aspect="auto", color_continuous_scale="Blues", labels=dict(x="Bulan", color="Omzet"),
                              title="Heatmap Omzet: Kategori × Bulan"), use_container_width=True)

# ----------------------------------------------------------------- TAB 5: STATISTIK
with tab5:
    if len(f) < 10:
        st.info("Data terlalu sedikit untuk uji statistik.")
    else:
        a, b = st.columns(2)
        corr = f[["Harga (Rp)", "Jumlah Terjual", "Omzet"]].corr(method="spearman")
        a.plotly_chart(px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
                                 title="Korelasi Spearman"), use_container_width=True)
        r, pv = stats.spearmanr(f["Harga (Rp)"], f["Jumlah Terjual"])
        with b:
            st.markdown("### Hasil Uji")
            st.write(f"**Harga vs Jumlah** (Spearman): ρ = `{r:.3f}`, p = `{pv:.3g}` → "
                     f"{'hubungan negatif signifikan' if (pv < .05 and r < 0) else 'hubungan positif signifikan' if (pv < .05) else 'tidak signifikan'}")
            grup = [g["Omzet"].values for _, g in f.groupby("Kategori") if len(g) > 1]
            if len(grup) >= 2:
                H, p2 = stats.kruskal(*grup)
                st.write(f"**Omzet antar kategori** (Kruskal-Wallis): H = `{H:.1f}`, p = `{p2:.3g}` → "
                         f"{'ada perbedaan signifikan' if p2 < .05 else 'tidak ada perbedaan signifikan'}")
            st.write(f"**Skewness harga**: `{f['Harga (Rp)'].skew():.2f}` (>1 berarti sangat condong ke kanan)")
            q1, q3 = f["Omzet"].quantile([.25, .75]); iqr = q3 - q1
            n_out = ((f["Omzet"] < q1 - 1.5 * iqr) | (f["Omzet"] > q3 + 1.5 * iqr)).sum()
            st.write(f"**Outlier omzet (IQR, global)**: `{n_out}` transaksi ({n_out/len(f):.1%})")
        st.markdown("**Elastisitas per produk** - korelasi Spearman harga vs jumlah di dalam tiap produk")
        el = [(p, stats.spearmanr(g["Harga (Rp)"], g["Jumlah Terjual"])[0], len(g)) for p, g in f.groupby("Produk") if len(g) > 10]
        if el:
            el = pd.DataFrame(el, columns=["Produk", "Spearman", "n"]).sort_values("Spearman")
            st.plotly_chart(px.bar(el, x="Spearman", y="Produk", orientation="h", color="Spearman",
                                   color_continuous_scale="RdBu_r", range_color=[-1, 1]), use_container_width=True)

# ----------------------------------------------------------------- DATA & DOWNLOAD
with st.expander("📄 Lihat & unduh data (hasil filter)"):
    st.dataframe(f.drop(columns=["Bulan", "Hari"]), use_container_width=True, hide_index=True)
    st.download_button("⬇️ Unduh CSV", f.drop(columns=["Bulan", "Hari"]).to_csv(index=False).encode("utf-8"),
                       "penjualan_filter.csv", "text/csv")
