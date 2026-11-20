import re
from itertools import combinations
from pathlib import Path
from scipy import stats
import copy
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torchvision import transforms
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch.nn.functional as F
from torchvision.models import resnet18, ResNet18_Weights



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
#plt.show()


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
plt.savefig("ciljevi_raspodela.png", dpi=150)
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

#plt.show()


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
#plt.show()


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
#plt.show()


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
#plt.show()

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


#metrika

def r2_po_cilju(y_true, y_pred):
    ss_res = ((y_true - y_pred) ** 2).sum(axis=0)
    ss_tot = ((y_true - y_true.mean(axis=0)) ** 2).sum(axis=0)
    return 1 - ss_res / ss_tot

def predict(model, loader):
    model.eval()
    preds, ys = [], []
    with torch.no_grad():
        for x, y in loader:
            preds.append(model(x.to(device)).cpu().numpy())
            ys.append(y.numpy())
    return np.concatenate(ys), np.concatenate(preds)

#Model

class Net(nn.Module):
    def __init__(self, n_features, hidden_size=64, use_dropout=False):
        super().__init__()
        self.fc1 = nn.Linear(n_features, hidden_size)
        self.fc1_drop = nn.Dropout(0.3)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc2_drop = nn.Dropout(0.3)
        self.fc3 = nn.Linear(hidden_size, 2)

        self.activation_function = nn.ReLU()
        self.use_dropout = use_dropout

    def forward(self, x):
        x = self.activation_function(self.fc1(x))
        if self.use_dropout:
            x = self.fc1_drop(x)

        x = self.activation_function(self.fc2(x))
        if self.use_dropout:
            x = self.fc2_drop(x)

        x = self.fc3(x)
        return x

def train_one_epoch(model, train_loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0

    for x, y in train_loader:
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        pred = model(x)
        loss = criterion(pred, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * len(x)

    return total_loss / len(train_loader.dataset)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def make_loader(df, shuffle):
    X = torch.tensor(df[TAB_KOLONE].to_numpy(dtype=np.float32))
    y = torch.tensor(df[CILJEVI].to_numpy(dtype=np.float32))
    return DataLoader(TensorDataset(X,y), batch_size=32, shuffle=shuffle)

train_loader = make_loader(train_p, shuffle=True)
val_loader = make_loader(val_p, shuffle=True)


def evaluate(model, loader, criterion):
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            pred = model(x)
            total_loss += criterion(pred, y).item() * len(x)

    return total_loss / len(loader.dataset)



torch.manual_seed(SEED)
model = Net(n_features=len(TAB_KOLONE), use_dropout=True).to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.001)

best_val_loss = float("inf")
best_state = None
best_epoch = 0

for epoch in range(1, 51):
    train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
    val_loss = evaluate(model, val_loader, criterion)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_state = copy.deepcopy(model.state_dict())
        best_epoch = epoch

    print(f"Epoch {epoch:3d} | train loss {train_loss:.4f} | val loss {val_loss:.4f}")

model.load_state_dict(best_state)
print(f"\nNajbolja epoha: {best_epoch}, val loss {best_val_loss:.4f}")

y_val, p_val = predict(model, val_loader)
r2 = r2_po_cilju(y_val, p_val)
print(f"Val R2: X18 = {r2[0]:.3f}, X26 = {r2[1]:.3f}, prosek = {r2.mean():.3f}")


y_train = train_p[CILJEVI].to_numpy()
baseline = np.tile(y_train.mean(axis=0), (len(y_val), 1))
r2_base = r2_po_cilju(y_val, baseline)
print(f"Trivijalno R2: X18 = {r2_base[0]:.3f}, X26 = {r2_base[1]:.3f}, prosek = {r2_base.mean():.3f}")


#cnn


IMG_SIZE = 128

def load_images(df):
    return [Image.open(SLIKE_DIR / f"{int(i)}.jpeg").convert("RGB").resize((IMG_SIZE, IMG_SIZE)) for i in df["id"]]

train_imgs = load_images(train_p)
val_imgs = load_images(val_p)

arr = np.stack([np.asarray(img, dtype=np.float32) / 255.0 for img in train_imgs])
img_mean = arr.mean(axis=(0, 1, 2))
img_std = arr.std(axis=(0, 1, 2))
del arr
print("Image mean (R,G,B):", img_mean.round(3), "| std:", img_std.round(3))

train_tf = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(img_mean.tolist(), img_std.tolist()),
])
eval_tf = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(img_mean.tolist(), img_std.tolist()),
])


class ImageDataset(Dataset):
    def __init__(self, images, df, transform):
        self.images = images
        self.y = torch.tensor(df[CILJEVI].to_numpy(dtype=np.float32))
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, i):
        return self.transform(self.images[i]), self.y[i]


train_img_loader = DataLoader(ImageDataset(train_imgs, train_p, train_tf), batch_size=32, shuffle=True)
val_img_loader = DataLoader(ImageDataset(val_imgs, val_p, eval_tf), batch_size=32, shuffle=False)


class CNNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.max_pool = nn.MaxPool2d(2)
        self.gap = nn.AdaptiveAvgPool2d(1)

        self.fc1 = nn.Linear(64, 32)
        self.fc1_drop = nn.Dropout(0.3)
        self.fc2 = nn.Linear(32, 2)

    def forward(self, x):
        x = self.max_pool(F.relu(self.conv1(x)))
        x = self.max_pool(F.relu(self.conv2(x)))
        x = self.max_pool(F.relu(self.conv3(x)))

        x = self.gap(x)
        x = torch.flatten(x, 1)

        x = self.fc1_drop(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return x


torch.manual_seed(SEED)
cnn = CNNet().to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(cnn.parameters(), lr=0.001, weight_decay=0.0001)

best_val_loss = float("inf")
best_state = None
best_epoch = 0

for epoch in range(1, 101):
    train_loss = train_one_epoch(cnn, train_img_loader, optimizer, criterion, device)
    val_loss = evaluate(cnn, val_img_loader, criterion)
    if val_loss < best_val_loss:
        best_val_loss, best_state, best_epoch = val_loss, copy.deepcopy(cnn.state_dict()), epoch
    print(f"Epoch {epoch:3d} | train loss {train_loss:.4f} | val loss {val_loss:.4f}")


cnn.load_state_dict(best_state)
print(f"\nNajbolja epoha: {best_epoch}, val loss {best_val_loss:.4f}")

y_val_img, p_val_img = predict(cnn, val_img_loader)
r2_cnn = r2_po_cilju(y_val_img, p_val_img)
print(f"CNN Val R2: X18 = {r2_cnn[0]:.3f}, X26 = {r2_cnn[1]:.3f}, prosek = {r2_cnn.mean():.3f}")

#Model 3 (cnn + sve kolone)


resnet = resnet18(weights=ResNet18_Weights.DEFAULT)
resnet.fc = nn.Identity()
resnet = resnet.to(device).eval()

resnet_tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

@torch.no_grad()
def extract_features(df, batch_size=64):
    features = []
    ids = df["id"].to_numpy()
    for start in range(0, len(ids), batch_size):
        batch = torch.stack([
            resnet_tf(Image.open(SLIKE_DIR / f"{int(i)}.jpeg").convert("RGB"))
            for i in ids[start:start + batch_size]
        ]).to(device)
        features.append(resnet(batch).cpu().numpy())
    return np.concatenate(features)

img_train = extract_features(train_p)
img_val = extract_features(val_p)
img_test = extract_features(test_p)
print("Image features:", img_train.shape, img_val.shape, img_test.shape)


scaler_img = StandardScaler().fit(img_train)
img_train_s = scaler_img.transform(img_train)
img_val_s = scaler_img.transform(img_val)
img_test_s = scaler_img.transform(img_test)

tab_train = train_p[TAB_KOLONE].to_numpy(dtype=np.float32)
tab_val = val_p[TAB_KOLONE].to_numpy(dtype=np.float32)
tab_test = test_p[TAB_KOLONE].to_numpy(dtype=np.float32)

y_train_np = train_p[CILJEVI].to_numpy(dtype=np.float32)
y_val_np = val_p[CILJEVI].to_numpy(dtype=np.float32)
y_test_np = test_p[CILJEVI].to_numpy(dtype=np.float32)

fused_train = np.hstack([img_train_s, tab_train])
fused_val = np.hstack([img_val_s, tab_val])
fused_test = np.hstack([img_test_s, tab_test])


def to_loader(X, y, shuffle):
    return DataLoader(TensorDataset(torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)), batch_size=32, shuffle=shuffle)


def train_mlp(X_tr, y_tr, X_va, y_va, epochs=100):
    tr_loader = to_loader(X_tr, y_tr, shuffle=True)
    va_loader = to_loader(X_va, y_va, shuffle=False)

    torch.manual_seed(SEED)
    model = Net(n_features=X_tr.shape[1], use_dropout=True).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-3)

    best_val_loss, best_state, best_epoch = float("inf"), None, 0
    for epoch in range(1, epochs + 1):
        train_one_epoch(model, tr_loader, optimizer, criterion, device)
        val_loss = evaluate(model, va_loader, criterion)
        if val_loss < best_val_loss:
            best_val_loss, best_state, best_epoch = val_loss, copy.deepcopy(model.state_dict()), epoch

    model.load_state_dict(best_state)
    y_true, y_pred = predict(model, va_loader)
    return model, r2_po_cilju(y_true, y_pred), best_epoch


model_img, r2_img, ep_img = train_mlp(img_train_s, y_train_np, img_val_s, y_val_np)
print(f"ResNet slika:   X18 = {r2_img[0]:.3f}, X26 = {r2_img[1]:.3f}, prosek = {r2_img.mean():.3f} (epoha {ep_img})")

model_v3, r2_v3, ep_v3 = train_mlp(fused_train, y_train_np, fused_val, y_val_np)
print(f"V3 slika+tabela: X18 = {r2_v3[0]:.3f}, X26 = {r2_v3[1]:.3f}, prosek = {r2_v3.mean():.3f} (epoha {ep_v3})")



assert model.fc1.in_features == len(TAB_KOLONE), "`model` vise nije V1 (MLP nad tabelom)"

test_imgs = load_images(test_p)

modeli = {
    "V1: MLP, tabela": (
        model,
        make_loader(val_p, shuffle=False),
        make_loader(test_p, shuffle=False),
    ),
    "V2: CNN od nule, slika": (
        cnn,
        DataLoader(ImageDataset(val_imgs, val_p, eval_tf), batch_size=32, shuffle=False),
        DataLoader(ImageDataset(test_imgs, test_p, eval_tf), batch_size=32, shuffle=False),
    ),
    "ResNet18, slika": (
        model_img,
        to_loader(img_val_s, y_val_np, shuffle=False),
        to_loader(img_test_s, y_test_np, shuffle=False),
    ),
    "V3: ResNet18 + tabela": (
        model_v3,
        to_loader(fused_val, y_val_np, shuffle=False),
        to_loader(fused_test, y_test_np, shuffle=False),
    ),
}

redovi = []

mean_train = y_train_np.mean(axis=0)
r2_v = r2_po_cilju(y_val_np, np.tile(mean_train, (len(y_val_np), 1)))
r2_t = r2_po_cilju(y_test_np, np.tile(mean_train, (len(y_test_np), 1)))
redovi.append({"Model": "Trivijalno", "Val X18": r2_v[0], "Val X26": r2_v[1],
               "Test X18": r2_t[0], "Test X26": r2_t[1]})

for ime, (m, val_loader_m, test_loader_m) in modeli.items():
    r2_v = r2_po_cilju(*predict(m, val_loader_m))
    r2_t = r2_po_cilju(*predict(m, test_loader_m))
    redovi.append({"Model": ime, "Val X18": r2_v[0], "Val X26": r2_v[1],
                   "Test X18": r2_t[0], "Test X26": r2_t[1]})

rezultati = pd.DataFrame(redovi).set_index("Model")
rezultati["Val prosek"] = rezultati[["Val X18", "Val X26"]].mean(axis=1)
rezultati["Test prosek"] = rezultati[["Test X18", "Test X26"]].mean(axis=1)

print("\nR2 (log10 skala), validacija i test:")
print(rezultati.round(3).to_string())
rezultati.round(3).to_csv("rezultati.csv")