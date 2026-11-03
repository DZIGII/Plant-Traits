import re
from itertools import combinations
from pathlib import Path
from scipy import stats

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler



INDEKS = "RN 101/2024"  # <-- upisite svoj broj indeksa

VIDLJIVE = ["X18_mean", "X3112_mean", "X11_mean"]
NEVIDLJIVE = ["X4_mean", "X50_mean", "X26_mean"]
GRUPE = ["WORLDCLIM", "SOIL", "MODIS", "VOD"]
PAROVI_OSOBINA = [(v, n) for v in VIDLJIVE for n in NEVIDLJIVE]  # 9 parova
PAROVI_GRUPA = list(combinations(GRUPE, 2))                      # 6 parova


def procitaj_indeks(indeks):
    m = re.fullmatch(r"\s*([A-Za-z]*)\s*(\d+)\s*/\s*(\d{2}|\d{4})\s*", indeks)
    if not m:
        raise ValueError(f"Indeks treba da bude u obliku 'RN 01/2026', a ne {indeks!r}")
    broj, godina = int(m.group(2)), int(m.group(3))
    if godina < 100:
        godina += 2000
    return broj, godina


BROJ, GODINA = procitaj_indeks(INDEKS)
SEED = int(f"{BROJ}{GODINA}")
CILJEVI = list(PAROVI_OSOBINA[BROJ % 9])
MOJE_GRUPE = list(PAROVI_GRUPA[(BROJ // 9) % 6])

print(f"Indeks: {INDEKS}  ->  k = {BROJ}, seed = {SEED}")
print(f"Osobine koje predvidjate: {CILJEVI}")
print(f"Grupe kolona koje koristite: {' + '.join(MOJE_GRUPE)}")


LOKALNI_DIR = Path("data")
df_sve = pd.read_csv(LOKALNI_DIR / "train.csv")
print(f"Ceo skup: {len(df_sve)} redova, {df_sve.shape[1]} kolona")


# Vas podskup -- ne menjati.
podskup = df_sve.sample(n=1500, random_state=SEED).reset_index(drop=True)

TAB_KOLONE = [c for c in podskup.columns if any(c.startswith(g + "_") for g in MOJE_GRUPE)]
print(f"Podskup: {len(podskup)} redova")
print(f"Tabelarne kolone koje model sme da koristi: {len(TAB_KOLONE)}")
for g in MOJE_GRUPE:
    print(f"  {g}: {sum(c.startswith(g + '_') for c in TAB_KOLONE)} kolona")

nedostaje = podskup[TAB_KOLONE + CILJEVI].isna().sum()
if nedostaje.any():
    print("\nPaznja: neke kolone imaju nedostajuce vrednosti -- obradite ih pre treniranja:")
    print(nedostaje[nedostaje > 0])


SLIKE_DIR = LOKALNI_DIR / "train_images"
nema = [i for i in podskup["id"] if not (SLIKE_DIR / f"{i}.jpeg").exists()]
print(f"Slika na disku: {len(podskup) - len(nema)} / {len(podskup)}")


fig, axes = plt.subplots(2, 4, figsize=(14, 7))
primeri = podskup.head(8)
for ax, id_, vrednosti in zip(axes.flat, primeri["id"], primeri[CILJEVI].to_numpy()):
    ax.imshow(Image.open(SLIKE_DIR / f"{id_}.jpeg"))
    ax.set_title("\n".join(f"{c} = {v:.3g}" for c, v in zip(CILJEVI, vrednosti)), fontsize=9)
    ax.axis("off")
plt.tight_layout()
plt.show()


class PlantTraitsDataset(Dataset):
    def __init__(self, df, slike_dir, tab_kolone, ciljevi, transform=None):
        self.ids = df["id"].to_numpy()
        self.tab = torch.tensor(df[tab_kolone].to_numpy(dtype=np.float32))
        self.y = torch.tensor(df[ciljevi].to_numpy(dtype=np.float32))
        self.slike_dir = Path(slike_dir)
        self.transform = transform or transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        slika = Image.open(self.slike_dir / f"{self.ids[i]}.jpeg").convert("RGB")
        return self.transform(slika), self.tab[i], self.y[i]


# Provera: jedan batch iz celog podskupa (ovo NIJE podela na trening/validaciju/test).
# ds = PlantTraitsDataset(podskup, SLIKE_DIR, TAB_KOLONE, CILJEVI)
# slike, tab, y = next(iter(DataLoader(ds, batch_size=16, shuffle=True)))
# print("slike:", tuple(slike.shape), "| tabela:", tuple(tab.shape), "| ciljevi:", tuple(y.shape))


#Analiza podataka


#podskup[CILJEVI].hist(bins=50, figsize=(10, 4))
#plt.show()


log_targets = np.log10(podskup[CILJEVI]).add_prefix("log10_")
all_targets = pd.concat([podskup[CILJEVI], log_targets], axis=1)

axes = all_targets.hist(bins=50, figsize=(14, 10), density=True, alpha=0.6, layout=(2, 2))
for ax in axes.flat:
    column = ax.get_title()

    mean_val = all_targets[column].mean()
    variance_val = all_targets[column].var()
    std_val = all_targets[column].std()
    skewness_val = stats.skew(all_targets[column])
    kurtosis_val = stats.kurtosis(all_targets[column])

    all_targets[column].plot.kde(ax=ax, color="blue", linewidth=2, label="KDE", alpha=0.6)

    ax.axvline(mean_val, color="red", linestyle="--", label=f"mean = {mean_val:.2f}")
    ax.axvline(mean_val - std_val, color="orange", linestyle=":", label=f"mean ± std (std = {std_val:.2f})")
    ax.axvline(mean_val + std_val, color="orange", linestyle=":")

    ax.text(0.02, 0.97,
            f"variance = {variance_val:.2f}\nskewness = {skewness_val:.2f}\nkurtosis = {kurtosis_val:.2f}",
            transform=ax.transAxes, va="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

    ax.set_xlim(all_targets[column].min(), all_targets[column].max())
    ax.set_xlabel("value")
    ax.set_ylabel("Density")
    ax.legend(loc="upper right")

plt.suptitle("Distribution of target variables: original vs log scale")
plt.tight_layout()
plt.show()


def show_extremes(column, n=8):
    sorted_df = podskup.sort_values(column)
    lowest = sorted_df.head(n)
    highest = sorted_df.tail(n)

    fig, axes = plt.subplots(2, n, figsize=(2.5 * n, 6))
    for row, (label, subset) in enumerate([("Lowest", lowest), ("Highest", highest)]):
        for ax, (_, r) in zip(axes[row], subset.iterrows()):
            ax.imshow(Image.open(SLIKE_DIR / f"{int(r['id'])}.jpeg"))
            ax.set_title(f"{r[column]:.3g}", fontsize=9)
            ax.axis("off")
        axes[row, 0].text(-0.15, 0.5, label, transform=axes[row, 0].transAxes,
                          rotation=90, va="center", ha="right", fontsize=12)
    plt.suptitle(f"{column}: {n} lowest vs {n} highest values")
    plt.tight_layout()
    plt.savefig(f"extremes_{column}.png", dpi=150)
    plt.show()



#izbacibanje nemogucih vrednosti
MAX_VISINA = 120

nemoguce = podskup["X18_mean"] > MAX_VISINA
print(f"Nemoguce vrednosti visine (> {MAX_VISINA} m): {nemoguce.sum()}")
print(podskup.loc[nemoguce, ["id"] + CILJEVI].to_string())

podskup_cist = podskup[~nemoguce].reset_index(drop=True)
print(f"\nPrimera posle izbacivanja: {len(podskup_cist)} / {len(podskup)}")

log_cist = np.log10(podskup_cist[CILJEVI])
z = (log_cist - log_cist.mean()) / log_cist.std()
ekstremi = (z.abs() > 3)


#show_extremes("X18_mean")
#show_extremes("X26_mean")


x = np.log10(podskup["X18_mean"])
y = np.log10(podskup["X26_mean"])

r = np.corrcoef(x, y)[0, 1]
print(f"Korelacija r = {r:.3f}")

plt.scatter(x, y, color="blue", s=10, alpha=0.5)
plt.xlabel("log10(X18_mean) - height")
plt.ylabel("log10(X26_mean) - seed mass")
plt.title("Height vs seed mass, r = {r:.2f}")

plt.show()


# print(podskup.describe().index.tolist())
# print(podskup.describe().columns.tolist())
# print(podskup[TAB_KOLONE].describe())

modis = [c for c in TAB_KOLONE if c.startswith("MODIS")]
vod = [c for c in TAB_KOLONE if c.startswith("VOD")]


def month(c):
    return int(re.search(r"m(\d+)$", c).group(1))


band1 = sorted([c for c in modis if "band_01" in c], key=month)
band2 = sorted([c for c in modis if "band_02" in c], key=month)

#crveni kanal - biljke ga apsorbuju
b1 = podskup_cist[band1].to_numpy()
#infracrveni kanal - biljke ga snazno reflektuju
b2 = podskup_cist[band2].to_numpy()

ndvi = (b2 - b1) / (b2 + b1)
ndvi_mean = ndvi.mean(axis=1)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for ax, c in zip(axes, CILJEVI):
    y = log_cist[c]
    r = np.corrcoef(ndvi_mean, y)[0, 1]
    ax.scatter(ndvi_mean, y, s=10, alpha=0.5)
    ax.set_xlabel("Mean annual NDVI")
    ax.set_ylabel(f"log10({c})")
    ax.set_title(f"NDVI vs {c}, r = {r:.2f}")

plt.suptitle("Vegetation density (NDVI) vs plant traits")
plt.tight_layout()
plt.savefig("ndvi_vs_traits.png", dpi=150)
plt.show()


tab_skew = podskup_cist[TAB_KOLONE].skew()
tab_kurt = podskup_cist[TAB_KOLONE].kurtosis()

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for ax, val, name in [(axes[0], tab_skew, "Skewness"), (axes[1], tab_kurt, "Kurtosis")]:
    ax.hist(val[modis], bins=30, alpha=0.6, label="MODIS")
    ax.hist(val[vod], bins=30, alpha=0.6, label="VOD")
    ax.axvline(0, color="black", linestyle="--", linewidth=1)
    ax.set_xlabel(name)
    ax.set_ylabel("Number of columns")
    ax.set_title(f"{name} of tabular columns")
    ax.legend()
axes[0].axvline(1, color="red", linestyle=":", linewidth=1)
axes[0].axvline(-1, color="red", linestyle=":", linewidth=1)
plt.tight_layout()
plt.savefig("tabular_moments.png", dpi=150)
plt.show()


def kanal(c):
    return "band " + re.search(r"band_(\d+)", c).group(1)


def frekvencija(c):
    return "VOD " + c.split("_")[1]


corr = pd.DataFrame({cilj: podskup_cist[TAB_KOLONE].corrwith(log_cist[cilj]) for cilj in CILJEVI})
edges = corr.abs().max().max()

fig, axes = plt.subplots(2, 2, figsize=(16, 8), gridspec_kw={"height_ratios": [5, 3]}, constrained_layout=True)

for j, cilj in enumerate(CILJEVI):
    for i, (kolone, ime_reda, grupa) in enumerate([(modis, kanal, "MODIS"), (vod, frekvencija, "VOD")]):
        tabela = pd.DataFrame({
            "red": [ime_reda(c) for c in kolone],
            "mesec": [month(c) for c in kolone],
            "r": corr.loc[kolone, cilj].values,
        }).pivot(index="red", columns="mesec", values="r")

        ax = axes[i, j]
        im = ax.imshow(tabela.values, cmap="coolwarm", vmin=-edges, vmax=edges, aspect="auto")
        ax.set_xticks(range(12))
        ax.set_xticklabels(tabela.columns)
        ax.set_yticks(range(len(tabela.index)))
        ax.set_yticklabels(tabela.index)
        ax.set_xlabel("Month")
        ax.set_title(f"{grupa} vs log10({cilj})")

        for y in range(tabela.shape[0]):
            for x in range(tabela.shape[1]):
                ax.text(x, y, f"{tabela.values[y, x]:.2f}", ha="center", va="center", fontsize=7)

fig.colorbar(im, ax=axes, label="Pearson r")
fig.suptitle("Correlation of MODIS and VOD columns with targets")
plt.savefig("corr_heatmap.png", dpi=150)
plt.show()

for cilj in CILJEVI:
    print(f"\nNajjace korelacije sa {cilj}:")
    print(corr[cilj].reindex(corr[cilj].abs().sort_values(ascending=False).index).head(5).round(3).to_string())


# rgb = []

# for id_ in podskup_cist["id"]:
#     img = Image.open(SLIKE_DIR / f"{int(id_)}.jpeg").convert("RGB").resize((128, 128))
#     arr = np.asarray(img, dtype=np.float32) / 255.0
#     rgb.append(arr.mean(axis=(0, 1)))

# rgb = np.array(rgb)
# green_ratio = rgb[:, 1] / rgb.sum(axis=1)# green_ratio = g / (r + g + b)

# print("Mean per channel (R, G, B):", rgb.mean(axis=0).round(3))
# print("Std per channel (R, G, B):", rgb.std(axis=0).round(3))


# fig, axes = plt.subplots(1, 2, figsize=(14, 5))
# for k, (name, color) in enumerate([("R", "red"), ("G", "green"), ("B", "blue")]):
#     axes[0].hist(rgb[:, k], bins=50, alpha=0.5, color=color, label=name)

# axes[0].set_xlabel("Mean channel value per image (0-1)")
# axes[0].set_ylabel("Number of images")
# axes[0].set_title("Distribution of mean RGB values")
# axes[0].legend()

# r = np.corrcoef(green_ratio, ndvi_mean)[0, 1]
# axes[1].scatter(ndvi_mean, green_ratio, s=10, alpha=0.5)
# axes[1].set_xlabel("Mean annual NDVI (location)")
# axes[1].set_ylabel("Green ratio G / (R+G+B) (photo)")
# axes[1].set_title(f"Photo greenness vs location NDVI, r = {r:.2f}")

# plt.tight_layout()
# plt.savefig("rgb_analysis.png", dpi=150)
# plt.show()




#### MODEL


#Podela podataka

groups = pd.qcut(log_cist["X18_mean"], q=10, labels=False)
train, temp = train_test_split(podskup_cist, test_size=0.30, stratify=groups, random_state=SEED)

groups_temp = groups.loc[temp.index]
val, test = train_test_split(temp, test_size=0.50, stratify=groups_temp, random_state=SEED)

for ime, skup in [("train", train), ("val", val), ("test", test)]:
    print(f"\n{ime} ({len(skup)}):")
    print(np.log10(skup[CILJEVI]).describe().round(3).to_string())




print("Min MODIS vrednost:", podskup_cist[modis].min().min())
print("Broj negativnih MODIS vrednosti:", (podskup_cist[modis] < 0).sum().sum())


def obradi_modis(df):
    df = df.copy()
    df[modis] = np.log1p(df[modis].clip(lower=0))
    return df

train_p = obradi_modis(train)
val_p = obradi_modis(val)
test_p = obradi_modis(test)

scaler_tab = StandardScaler().fit(train_p[TAB_KOLONE])

for df in (train_p, val_p, test_p):
    df[TAB_KOLONE] = scaler_tab.transform(df[TAB_KOLONE])

print("NaN posle obrade:", train_p[TAB_KOLONE].isna().sum().sum(),
      val_p[TAB_KOLONE].isna().sum().sum(),
      test_p[TAB_KOLONE].isna().sum().sum())


for df in (train_p, val_p, test_p):
    df[CILJEVI] = np.log10(df[CILJEVI])

print(train_p[CILJEVI].describe().round(3).to_string())