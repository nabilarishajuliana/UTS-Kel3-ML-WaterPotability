"""
visualisasi_mushroom.py
Visualisasi EDA, komposisi cluster, dan heatmap
untuk dataset Mushrooms dengan metode K-Modes.
"""

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Menentukan lokasi folder proyek
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "mushrooms.csv"
RESULT_PATH = BASE_DIR / "hasil_clustering_kmodes.csv"
OUT_DIR = BASE_DIR / "visualisasi"

OUT_DIR.mkdir(exist_ok=True)

sns.set_theme(style="whitegrid")

# ==========================================
# 1. EDA: DISTRIBUSI KELAS JAMUR
# ==========================================

df = pd.read_csv(DATA_PATH)

class_labels = df["class"].map({
    "e": "Edible",
    "p": "Poisonous"
})

plt.figure(figsize=(7, 5))

ax = sns.countplot(
    x=class_labels,
    order=["Edible", "Poisonous"],
    color="steelblue"
)

ax.set_title("Distribusi Kelas pada Dataset Mushrooms")
ax.set_xlabel("Kelas Jamur")
ax.set_ylabel("Jumlah Sampel")

for bar in ax.patches:
    ax.annotate(
        f"{int(bar.get_height()):,}",
        (
            bar.get_x() + bar.get_width() / 2,
            bar.get_height()
        ),
        ha="center",
        va="bottom",
        xytext=(0, 4),
        textcoords="offset points"
    )

plt.tight_layout()
plt.savefig(
    OUT_DIR / "eda_distribusi_kelas.png",
    dpi=180,
    bbox_inches="tight"
)
plt.close()


# ==========================================
# 2. EDA: DISTRIBUSI FITUR KATEGORIKAL
# ==========================================

features = [
    "odor",
    "cap-color",
    "habitat",
    "population",
    "spore-print-color"
]

fig, axes = plt.subplots(2, 3, figsize=(15, 9))
axes = axes.flatten()

for ax, col in zip(axes, features):
    order = df[col].value_counts().index

    sns.countplot(
        data=df,
        x=col,
        order=order,
        ax=ax,
        color="steelblue"
    )

    ax.set_title(f"Distribusi {col}")
    ax.set_xlabel("Kategori")
    ax.set_ylabel("Jumlah Sampel")
    ax.tick_params(axis="x", rotation=35)

# Menyembunyikan subplot yang tidak digunakan
axes[-1].axis("off")

fig.suptitle(
    "Distribusi Fitur Kategorikal yang Digunakan pada K-Modes",
    fontsize=15
)

plt.tight_layout()
plt.savefig(
    OUT_DIR / "eda_distribusi_fitur.png",
    dpi=180,
    bbox_inches="tight"
)
plt.close()


# ==========================================
# 3. VISUALISASI KOMPOSISI CLUSTER
# ==========================================

result = pd.read_csv(RESULT_PATH)

# Menghitung jumlah kelas asli pada setiap cluster
ct = pd.crosstab(
    result["cluster"],
    result["true_class"]
)

# Memastikan urutan kelas edible dan poisonous
ct = ct.reindex(
    columns=["e", "p"],
    fill_value=0
)

ct.columns = ["Edible", "Poisonous"]

ax = ct.plot(
    kind="bar",
    stacked=True,
    figsize=(8, 5)
)

ax.set_title("Komposisi Kelas Asli pada Setiap Cluster")
ax.set_xlabel("Cluster K-Modes")
ax.set_ylabel("Jumlah Sampel")
ax.legend(title="Kelas Asli")

plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig(
    OUT_DIR / "komposisi_cluster.png",
    dpi=180,
    bbox_inches="tight"
)
plt.close()


# ==========================================
# 4. HEATMAP CLUSTER VS KELAS ASLI
# ==========================================

plt.figure(figsize=(7, 5))

sns.heatmap(
    ct,
    annot=True,
    fmt=",",
    cmap="YlGnBu",
    linewidths=0.5
)

plt.title("Heatmap Cluster vs Kelas Asli")
plt.xlabel("Kelas Asli")
plt.ylabel("Cluster K-Modes")

plt.tight_layout()
plt.savefig(
    OUT_DIR / "heatmap_cluster_class.png",
    dpi=180,
    bbox_inches="tight"
)
plt.close()


# ==========================================
# 5. MENAMPILKAN HASIL
# ==========================================

print("Visualisasi berhasil dibuat!")
print("Lokasi penyimpanan:", OUT_DIR)

for file in sorted(OUT_DIR.glob("*.png")):
    print("-", file.name)