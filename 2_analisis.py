"""
Langkah 2 - Analisis detail + visualisasi (matplotlib/seaborn).
Visualisasi mengikuti rekomendasi pada Bedah Kasus 1:
  Nominal (Produk, Kategori)  -> Bar Chart / Pie Chart
  Numerik Kontinu (Harga)     -> Histogram
  Numerik Diskrit (Jumlah)    -> Histogram/Bar (hasil hitung bulat)
  Waktu (Tanggal)             -> Line Chart (tren)
Hasil gambar disimpan di folder ./hasil_analisis
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

sns.set_theme(style="whitegrid", palette="deep")
OUT = "hasil_analisis"
os.makedirs(OUT, exist_ok=True)
rp = lambda x, _=None: f"Rp{x:,.0f}".replace(",", ".")

def judul(teks):
    print("\n" + "=" * 70 + f"\n{teks}\n" + "=" * 70)

# ------------------------------------------------------------------ 1. LOAD
df = pd.read_csv("data_penjualan.csv", parse_dates=["Tanggal"])
df["Omzet"] = df["Harga (Rp)"] * df["Jumlah Terjual"]
df["Bulan"] = df["Tanggal"].dt.to_period("M").dt.to_timestamp()
df["Hari"] = df["Tanggal"].dt.day_name()

# ------------------------------------------------------------------ 2. KUALITAS DATA
judul("1. KUALITAS DATA")
print(f"Jumlah baris/kolom : {df.shape}")
print("\nTipe data:\n", df.dtypes.to_string())
print("\nMissing value per kolom:\n", df.isna().sum().to_string())
print(f"\nBaris duplikat     : {df.duplicated(subset=['Produk','Harga (Rp)','Jumlah Terjual','Tanggal']).sum()}")
print(f"Rentang tanggal    : {df['Tanggal'].min().date()} s/d {df['Tanggal'].max().date()}")

# ------------------------------------------------------------------ 3. KLASIFIKASI VARIABEL
judul("2. KLASIFIKASI SKALA DATA")
print(pd.DataFrame({
    "Variabel": ["Produk", "Kategori", "Harga (Rp)", "Jumlah Terjual", "Tanggal"],
    "Jenis": ["Kategorik", "Kategorik", "Numerik Kontinu", "Numerik Diskrit", "Waktu"],
    "Skala": ["Nominal", "Nominal", "Rasio", "Rasio", "Interval"],
    "Visualisasi": ["Bar/Pie", "Bar/Pie", "Histogram", "Histogram/Bar", "Line Chart"],
}).to_string(index=False))

# ------------------------------------------------------------------ 4. STATISTIK DESKRIPTIF
judul("3. STATISTIK DESKRIPTIF NUMERIK")
num = df[["Harga (Rp)", "Jumlah Terjual", "Omzet"]]
desk = num.describe().T
desk["skewness"] = num.skew()
desk["kurtosis"] = num.kurt()
desk["CV (%)"] = num.std() / num.mean() * 100
print(desk.round(2).to_string())

judul("4. STATISTIK DESKRIPTIF KATEGORIK")
for k in ["Kategori", "Produk"]:
    f = df[k].value_counts()
    print(f"\n{k}: modus = {f.index[0]} ({f.iloc[0]} transaksi, {f.iloc[0]/len(df):.1%})")
    print(pd.DataFrame({"Frekuensi": f, "Persen (%)": (f / len(df) * 100).round(1)}).to_string())

# ------------------------------------------------------------------ 5. KINERJA PENJUALAN
judul("5. KINERJA PENJUALAN")
total_omzet, total_unit = df["Omzet"].sum(), df["Jumlah Terjual"].sum()
print(f"Total omzet        : {rp(total_omzet)}")
print(f"Total unit terjual : {total_unit:,}".replace(",", "."))
print(f"Rata-rata omzet/transaksi : {rp(df['Omzet'].mean())}")
print(f"Rata-rata harga/unit (tertimbang) : {rp(total_omzet/total_unit)}")

per_kat = df.groupby("Kategori").agg(Transaksi=("ID", "count"), Unit=("Jumlah Terjual", "sum"),
                                     Omzet=("Omzet", "sum"), RataHarga=("Harga (Rp)", "mean"))
per_kat["Kontribusi Omzet (%)"] = per_kat["Omzet"] / total_omzet * 100
per_kat = per_kat.sort_values("Omzet", ascending=False)
print("\nPer kategori:\n", per_kat.round(1).to_string())

per_prod = df.groupby(["Produk", "Kategori"]).agg(Transaksi=("ID", "count"), Unit=("Jumlah Terjual", "sum"),
                                                  Omzet=("Omzet", "sum"), RataHarga=("Harga (Rp)", "mean")).reset_index()
per_prod = per_prod.sort_values("Omzet", ascending=False)
per_prod["Kumulatif (%)"] = per_prod["Omzet"].cumsum() / total_omzet * 100
print("\nPer produk (urut omzet) + analisis Pareto:\n", per_prod.round(1).to_string(index=False))
n80 = (per_prod["Kumulatif (%)"] < 80).sum() + 1
print(f"-> {n80} dari {len(per_prod)} produk menghasilkan >=80% omzet (Pareto).")

# ------------------------------------------------------------------ 6. TREN WAKTU
judul("6. TREN WAKTU")
bln = df.groupby("Bulan").agg(Omzet=("Omzet", "sum"), Unit=("Jumlah Terjual", "sum"), Transaksi=("ID", "count"))
bln["Pertumbuhan Omzet MoM (%)"] = bln["Omzet"].pct_change() * 100
print(bln.round(1).to_string())
print(f"\nBulan omzet tertinggi : {bln['Omzet'].idxmax():%B %Y} ({rp(bln['Omzet'].max())})")
print(f"Bulan omzet terendah  : {bln['Omzet'].idxmin():%B %Y} ({rp(bln['Omzet'].min())})")

x = np.arange(len(bln))
reg = stats.linregress(x, bln["Omzet"])
print(f"Tren linear omzet bulanan: slope = {rp(reg.slope)}/bulan, R² = {reg.rvalue**2:.3f}, p = {reg.pvalue:.4f}")

urut = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
hr = df.groupby("Hari")["Omzet"].agg(["count", "mean"]).reindex(urut)
print("\nRata-rata omzet per hari:\n", hr.round(0).to_string())

# ------------------------------------------------------------------ 7. KORELASI & UJI STATISTIK
judul("7. KORELASI & UJI STATISTIK")
r_p, p_p = stats.pearsonr(df["Harga (Rp)"], df["Jumlah Terjual"])
r_s, p_s = stats.spearmanr(df["Harga (Rp)"], df["Jumlah Terjual"])
print(f"Harga vs Jumlah (seluruh produk): Pearson r = {r_p:.3f} (p={p_p:.3g}), Spearman rho = {r_s:.3f} (p={p_s:.3g})")

# korelasi di dalam produk (kontrol efek kategori produk)
dalam = []
for p, g in df.groupby("Produk"):
    if len(g) > 10:
        dalam.append((p, stats.spearmanr(g["Harga (Rp)"], g["Jumlah Terjual"])[0]))
dalam = pd.DataFrame(dalam, columns=["Produk", "Spearman (Harga vs Jumlah)"]).sort_values("Spearman (Harga vs Jumlah)")
print("\nKorelasi harga-jumlah DI DALAM tiap produk:\n", dalam.round(3).to_string(index=False))
print("Catatan: Pearson gabungan lebih lemah dari Spearman karena hubungan non-linear & rentang harga antar produk sangat lebar; korelasi per produk lebih mencerminkan elastisitas harga.")

grup = [g["Omzet"].values for _, g in df.groupby("Kategori")]
H, pH = stats.kruskal(*grup)
print(f"\nKruskal-Wallis omzet antar kategori: H = {H:.2f}, p = {pH:.3g} -> "
      f"{'ada perbedaan signifikan' if pH < 0.05 else 'tidak ada perbedaan signifikan'}")

sw = stats.shapiro(df["Harga (Rp)"].sample(500, random_state=1))
print(f"Uji normalitas Shapiro-Wilk (harga, sampel 500): W = {sw.statistic:.3f}, p = {sw.pvalue:.3g} "
      f"-> {'normal' if sw.pvalue > 0.05 else 'tidak normal (right-skewed)'}")

judul("8. DETEKSI OUTLIER (IQR) PER PRODUK pada Omzet")
tot_out = 0
for p, g in df.groupby("Produk"):
    q1, q3 = g["Omzet"].quantile([.25, .75]); iqr = q3 - q1
    n = ((g["Omzet"] < q1 - 1.5 * iqr) | (g["Omzet"] > q3 + 1.5 * iqr)).sum()
    tot_out += n
    print(f"{p:<14}: {n} outlier dari {len(g)}")
print(f"Total outlier: {tot_out} ({tot_out/len(df):.1%})")

# ------------------------------------------------------------------ 9. VISUALISASI
fig, ax = plt.subplots(1, 2, figsize=(13, 5))
fk = df["Kategori"].value_counts()
sns.barplot(x=fk.values, y=fk.index, ax=ax[0]); ax[0].set_title("Jumlah Transaksi per Kategori (Bar Chart)")
for i, v in enumerate(fk.values): ax[0].text(v + 2, i, str(v), va="center")
ax[1].pie(per_kat["Omzet"], labels=per_kat.index, autopct="%1.1f%%", startangle=90,
          colors=sns.color_palette("deep"), wedgeprops={"edgecolor": "white"})
ax[1].set_title("Kontribusi Omzet per Kategori (Pie Chart)")
plt.tight_layout(); plt.savefig(f"{OUT}/01_kategori.png", dpi=130); plt.close()

fig, ax = plt.subplots(figsize=(10, 5))
fp = df["Produk"].value_counts()
sns.barplot(x=fp.values, y=fp.index, ax=ax, hue=fp.index, legend=False)
ax.set_title("Frekuensi Transaksi per Produk"); plt.tight_layout()
plt.savefig(f"{OUT}/02_produk.png", dpi=130); plt.close()

fig, ax = plt.subplots(1, 2, figsize=(13, 5))
sns.histplot(df["Harga (Rp)"], bins=40, kde=True, ax=ax[0]); ax[0].set_title("Histogram Harga (Kontinu)")
ax[0].xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x/1e6:.1f} jt"))
sns.histplot(np.log10(df["Harga (Rp)"]), bins=40, kde=True, ax=ax[1], color="tab:orange")
ax[1].set_title("Histogram Harga (skala log10)"); ax[1].set_xlabel("log10(Harga Rp)")
plt.tight_layout(); plt.savefig(f"{OUT}/03_hist_harga.png", dpi=130); plt.close()

fig, ax = plt.subplots(figsize=(8, 5))
sns.histplot(df["Jumlah Terjual"], bins=range(0, df["Jumlah Terjual"].max() + 10, 10), ax=ax, color="tab:green")
ax.set_title("Distribusi Jumlah Terjual (Diskrit)"); plt.tight_layout()
plt.savefig(f"{OUT}/04_hist_jumlah.png", dpi=130); plt.close()

fig, ax = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
ax[0].plot(bln.index, bln["Omzet"] / 1e6, marker="o"); ax[0].set_title("Tren Omzet Bulanan (Line Chart)")
ax[0].plot(bln.index, (reg.intercept + reg.slope * x) / 1e6, "--", color="gray", label="Tren linear"); ax[0].legend()
ax[0].set_ylabel("Omzet (juta Rp)")
ax[1].plot(bln.index, bln["Unit"], marker="s", color="tab:green"); ax[1].set_title("Tren Unit Terjual Bulanan")
ax[1].set_ylabel("Unit"); plt.tight_layout(); plt.savefig(f"{OUT}/05_tren.png", dpi=130); plt.close()

fig, ax = plt.subplots(figsize=(10, 5))
harian = df.groupby("Tanggal")["Omzet"].sum().rolling(7, min_periods=1).mean() / 1e6
ax.plot(harian.index, harian.values); ax.set_title("Omzet Harian (rata-rata bergerak 7 hari)")
ax.set_ylabel("Juta Rp"); plt.tight_layout(); plt.savefig(f"{OUT}/06_tren_harian.png", dpi=130); plt.close()

fig, ax = plt.subplots(figsize=(11, 5))
urut_p = df.groupby("Produk")["Harga (Rp)"].median().sort_values().index
sns.boxplot(data=df, y="Produk", x="Harga (Rp)", order=urut_p, ax=ax); ax.set_xscale("log")
ax.set_title("Sebaran Harga per Produk (Boxplot, skala log)"); plt.tight_layout()
plt.savefig(f"{OUT}/07_box_harga.png", dpi=130); plt.close()

fig, ax = plt.subplots(figsize=(9, 6))
sns.scatterplot(data=df, x="Harga (Rp)", y="Jumlah Terjual", hue="Kategori", alpha=.6, ax=ax)
ax.set_xscale("log"); ax.set_title("Harga vs Jumlah Terjual"); plt.tight_layout()
plt.savefig(f"{OUT}/08_scatter.png", dpi=130); plt.close()

fig, ax = plt.subplots(figsize=(6, 5))
sns.heatmap(df[["Harga (Rp)", "Jumlah Terjual", "Omzet"]].corr(method="spearman"), annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
ax.set_title("Korelasi Spearman"); plt.tight_layout(); plt.savefig(f"{OUT}/09_korelasi.png", dpi=130); plt.close()

fig, ax = plt.subplots(figsize=(10, 5))
pareto = per_prod.reset_index(drop=True)
ax.bar(pareto["Produk"], pareto["Omzet"] / 1e6); ax.tick_params(axis="x", rotation=45)
ax2 = ax.twinx(); ax2.plot(pareto["Produk"], pareto["Kumulatif (%)"], color="red", marker="o"); ax2.axhline(80, ls="--", color="gray")
ax.set_ylabel("Omzet (juta Rp)"); ax2.set_ylabel("Kumulatif (%)"); ax.set_title("Pareto Omzet per Produk")
plt.tight_layout(); plt.savefig(f"{OUT}/10_pareto.png", dpi=130); plt.close()

# ------------------------------------------------------------------ 10. INSIGHT
judul("9. RINGKASAN INSIGHT")
top_k = per_kat.index[0]; top_p = per_prod.iloc[0]["Produk"]
print(f"1. Kategori omzet terbesar: {top_k} ({per_kat.iloc[0]['Kontribusi Omzet (%)']:.1f}% dari total omzet).")
print(f"2. Produk omzet terbesar: {top_p}; frekuensi transaksi terbanyak: {fp.index[0]}.")
print(f"3. Distribusi harga sangat right-skewed (skew = {df['Harga (Rp)'].skew():.2f}) karena rentang harga {rp(df['Harga (Rp)'].min())} - {rp(df['Harga (Rp)'].max())}.")
print(f"4. Omzet bulanan {'naik' if reg.slope > 0 else 'turun'} (slope {rp(reg.slope)}/bulan), puncak {bln['Omzet'].idxmax():%B %Y}.")
print(f"5. Produk berharga tinggi terjual lebih sedikit unit tetapi menyumbang omzet besar -> {n80} produk penentu 80% omzet.")
print(f"\nGrafik tersimpan di folder '{OUT}/'")
