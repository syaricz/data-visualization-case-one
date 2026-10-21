"""
Langkah 1 - Membuat data sintetis 1000 baris dari 2 data contoh (Bedah Kasus 1).

Skema asli:
  Produk (Nominal) | Kategori (Nominal) | Harga (Rp) (Kontinu-Rasio)
  Jumlah Terjual (Diskrit-Rasio) | Tanggal (Waktu-Interval)

Dua data contoh:
  Laptop | Elektronik | 8.500.000 | 45  | 01/01/24
  Buku   | Pendidikan |    75.000 | 120 | 02/01/24

Logika sintetis:
  - Harga: lognormal di sekitar harga dasar produk (variasi merek/diskon -> kontinu).
  - Jumlah: Poisson, rata-rata lebih rendah untuk produk mahal (hubungan harga-jumlah negatif).
  - Tanggal: sepanjang 2024, ada tren naik + musiman (lonjakan Ramadan/Lebaran, 12.12, akhir tahun).
"""
import numpy as np
import pandas as pd

SEED = 42
N = 1000
rng = np.random.default_rng(SEED)

# katalog produk: (produk, kategori, harga_dasar, rata2_jumlah, bobot_kemunculan)
KATALOG = [
    ("Laptop",         "Elektronik", 8_500_000, 45,  0.08),
    ("Smartphone",     "Elektronik", 4_500_000, 60,  0.10),
    ("Headphone",      "Elektronik",   450_000, 90,  0.08),
    ("Buku",           "Pendidikan",    75_000, 120, 0.12),
    ("Alat Tulis",     "Pendidikan",    25_000, 200, 0.09),
    ("Tas Sekolah",    "Pendidikan",   180_000, 85,  0.06),
    ("Kemeja",         "Fashion",      150_000, 110, 0.09),
    ("Sepatu",         "Fashion",      650_000, 70,  0.07),
    ("Kopi Kemasan",   "Makanan",       55_000, 180, 0.09),
    ("Snack Box",      "Makanan",       35_000, 220, 0.07),
    ("Kursi Kantor",   "Perabot",    1_200_000, 40,  0.05),
    ("Meja Belajar",   "Perabot",      850_000, 38,  0.04),
]
kat = pd.DataFrame(KATALOG, columns=["Produk", "Kategori", "HargaDasar", "RataJumlah", "Bobot"])
kat["Bobot"] /= kat["Bobot"].sum()

# --- pilih produk per baris
idx = rng.choice(len(kat), size=N, p=kat["Bobot"].values)
df = kat.iloc[idx].reset_index(drop=True)

# --- tanggal: sampling harian dengan bobot tren + musiman
hari = pd.date_range("2024-01-01", "2024-12-31", freq="D")
t = np.arange(len(hari))
bobot = 1 + 0.6 * t / len(hari)                                    # tren naik
for puncak, kuat, lebar in [("2024-04-08", 0.9, 12),               # jelang Lebaran
                            ("2024-07-15", 0.4, 10),               # tahun ajaran baru
                            ("2024-11-11", 0.6, 4),                # 11.11
                            ("2024-12-12", 0.7, 4),                # 12.12
                            ("2024-12-25", 0.5, 6)]:               # Natal/akhir tahun
    d = (hari - pd.Timestamp(puncak)).days.values
    bobot += kuat * np.exp(-(d / lebar) ** 2)
bobot[hari.dayofweek >= 5] *= 1.15                                 # akhir pekan lebih ramai
bobot /= bobot.sum()
df["Tanggal"] = rng.choice(hari, size=N, p=bobot)

# --- harga (kontinu): lognormal sekitar harga dasar, dibulatkan ke Rp500
sigma = 0.12
harga = df["HargaDasar"] * rng.lognormal(mean=0, sigma=sigma, size=N)
df["Harga (Rp)"] = (np.round(harga / 500) * 500).astype(int)

# --- jumlah terjual (diskrit): Poisson, makin mahal dari harga dasar -> makin sedikit
elastis = (df["HargaDasar"] / df["Harga (Rp)"]) ** 0.8
lam = df["RataJumlah"] * elastis
lam *= 1 + 0.3 * (df["Tanggal"].dt.month.values / 12)             # permintaan naik sepanjang tahun
df["Jumlah Terjual"] = np.maximum(1, rng.poisson(lam)).astype(int)

# --- sisipkan 2 data asli di baris awal agar tetap menjadi bagian dataset
asli = pd.DataFrame({
    "Produk": ["Laptop", "Buku"], "Kategori": ["Elektronik", "Pendidikan"],
    "Harga (Rp)": [8_500_000, 75_000], "Jumlah Terjual": [45, 120],
    "Tanggal": pd.to_datetime(["2024-01-01", "2024-01-02"]),
})
df = df[["Produk", "Kategori", "Harga (Rp)", "Jumlah Terjual", "Tanggal"]]
df = pd.concat([asli, df.iloc[:N - 2]], ignore_index=True)
df = df.sort_values("Tanggal").reset_index(drop=True)
df.insert(0, "ID", [f"TRX{i:04d}" for i in range(1, len(df) + 1)])

df.to_csv("data_penjualan.csv", index=False)
print(f"Data tersimpan: data_penjualan.csv  ({len(df)} baris)")
print(df.head(8).to_string(index=False))
