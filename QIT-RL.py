# %%
import os, random, csv, json
from typing import List, Tuple, Dict, Optional
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
import math

from tqdm import tqdm
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import (
    confusion_matrix, accuracy_score,
    silhouette_score, davies_bouldin_score,
    roc_auc_score
)

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler, Subset
from torch.optim.lr_scheduler import StepLR
from torchvision import datasets, transforms, models
from PIL import Image

import time
from mpl_toolkits.mplot3d import Axes3D

import seaborn as sns
# %%
RANDOM_SEED    = 42

BATCH_SIZE     = 128
EPOCHS         = 50
LR             = 1e-3 
WEIGHT_DECAY   = 5e-4

BACKBONE = "resnet18"  

DATASET_NAME = "cifar10"

MAMMO_ROOT = "./Dataset_Mammogram"

CIFAR_ROOT     = "./cifar_data"
TINY_ROOT      = "./tiny-imagenet-200"

SELECTED_CIFAR10_CLASSES: List[int] = list(range(10))

SELECTED_CIFAR100_CLASSES = list(range(100))

TINY_MAX_CLASSES: Optional[int] = 20   

PRETTY_PLOTS = True
SHOW_LEGENDS = False
PLOT_DPI = 300
MARKER_SIZE = 18
MARKER_ALPHA = 0.75
MARKER_EDGE_WIDTH = 0.30

TSNE_MAX_POINTS = 5000   
METRICS_MAX_POINTS = 8000 

USE_QIT_LOSS              = True
LAMBDA_QIT                = 0.5
QIT_MIN_CLASS_SAMPLES     = 2   
QIT_TRACE_NORMALIZE       = True

HEAD_TYPE = "cosine" 
COS_M              = 0.20
LEARNABLE_S        = True
S_INIT             = 16.0

USE_FEW_SHOT_EVAL        = True
FEW_SHOT_KS              = [1, 5, 10]
FEW_SHOT_EPISODES        = 100
FEW_SHOT_N_WAY           = 5   
FEW_SHOT_QUERY_PER_CLASS = 15
FEW_SHOT_USE_NORMALIZED  = True

def dataset_tag() -> str:
    return DATASET_NAME.lower()

RUN_NAME = f"{HEAD_TYPE}"
if HEAD_TYPE == "cosine" and LEARNABLE_S:
    RUN_NAME += "_ls"
RUN_NAME = ("qit_on_" if USE_QIT_LOSS else "qit_off_") + RUN_NAME

RUN_DIR = None  

def set_seed(seed=RANDOM_SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed()

def build_class_map(selected: List[int]) -> Dict[int,int]:
    return {orig: new for new, orig in enumerate(selected)}

class MammogramDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.samples = []
        self.transform = transform

        for label_name in ["normal", "mass"]:
            label = 0 if label_name == "normal" else 1
            folder = os.path.join(root_dir, label_name)

            if not os.path.isdir(folder):
                continue

            for fname in os.listdir(folder):
                path = os.path.join(folder, fname)
                if os.path.isfile(path):
                    self.samples.append((path, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")

        if self.transform:
            img = self.transform(img)

        return img, torch.tensor(label, dtype=torch.long)

class RemappedSubset(Dataset):
    def __init__(self, base: Dataset, selected: List[int], base_targets: List[int], transform=None):
        self.base = base
        self.selected = selected
        self.class_map = build_class_map(selected)
        self.transform = transform

        self.indices: List[int] = []
        self.labels: List[int] = []
        for idx, y in enumerate(base_targets):
            if int(y) in self.class_map:
                self.indices.append(idx)
                self.labels.append(self.class_map[int(y)])

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        base_idx = self.indices[idx]
        x, _ = self.base[base_idx]
        y = self.labels[idx]
        return x, torch.tensor(y, dtype=torch.long)

class IndexMappedDataset(Dataset):
    def __init__(self, base: Dataset, indices: List[int]):
        self.base = base
        self.indices = list(indices)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        return self.base[self.indices[idx]]

def create_weighted_sampler(labels: List[int], num_classes: int) -> WeightedRandomSampler:
    labels_np = np.asarray(labels, dtype=int)
    cls_cnt = np.bincount(labels_np, minlength=num_classes)
    cls_w   = 1.0 / (cls_cnt + 1e-6)
    samp_w  = torch.as_tensor(cls_w[labels_np], dtype=torch.double)
    return WeightedRandomSampler(samp_w, num_samples=len(samp_w), replacement=True)  # type: ignore

def make_transforms():
    base_tf = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    ])
    aug_tf = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    ])
    return base_tf, aug_tf

def get_dataset_and_loaders(batch_size=BATCH_SIZE, val_split=0.2) -> Tuple[DataLoader, DataLoader, List[str], int, str]:
    base_tf, aug_tf = make_transforms()

    if DATASET_NAME.lower() == "mammo":
        full_eval_ds = MammogramDataset(MAMMO_ROOT, transform=base_tf)
        full_aug_ds  = MammogramDataset(MAMMO_ROOT, transform=aug_tf)

        n_val = int(len(full_eval_ds) * val_split)
        idxs = torch.randperm(len(full_eval_ds)).tolist()
        val_indices = idxs[:n_val]
        tr_indices  = idxs[n_val:]

        tr_ds  = IndexMappedDataset(full_aug_ds,  tr_indices)
        val_ds = IndexMappedDataset(full_eval_ds, val_indices)

        tr_labels = [full_eval_ds.samples[i][1] for i in tr_indices]

        tr_loader = DataLoader(
            tr_ds,
            batch_size=batch_size,
            sampler=create_weighted_sampler(tr_labels, 2)
        )
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

        class_names = ["normal", "mass"]
        num_classes = 2
        run_suffix = "mammo_binary"
        return tr_loader, val_loader, class_names, num_classes, run_suffix

    if DATASET_NAME.lower() == "cifar10":
        base_train_eval = datasets.CIFAR10(root=CIFAR_ROOT, train=True, download=True, transform=base_tf)
        base_train_aug  = datasets.CIFAR10(root=CIFAR_ROOT, train=True, download=True, transform=aug_tf)
        selected = SELECTED_CIFAR10_CLASSES
        class_names_full = ["plane","car","bird","cat","deer","dog","frog","horse","ship","truck"]
        class_names = [class_names_full[i] for i in selected]

        full_eval_ds = RemappedSubset(base_train_eval, selected, base_train_eval.targets, transform=base_tf)
        full_aug_ds  = RemappedSubset(base_train_aug,  selected, base_train_aug.targets,  transform=aug_tf)
        num_classes = len(class_names)

        n_val = int(len(full_eval_ds) * val_split)
        idxs = torch.randperm(len(full_eval_ds)).tolist()
        val_indices = idxs[:n_val]
        tr_indices  = idxs[n_val:]

        tr_ds  = IndexMappedDataset(full_aug_ds,  tr_indices)
        val_ds = IndexMappedDataset(full_eval_ds, val_indices)

        tr_labels = [full_eval_ds.labels[i] for i in tr_indices]
        tr_loader = DataLoader(tr_ds, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

        run_suffix = f"cifar10_{num_classes}cls"
        return tr_loader, val_loader, class_names, num_classes, run_suffix

    if DATASET_NAME.lower() == "cifar100":
        base_train_eval = datasets.CIFAR100(root=CIFAR_ROOT, train=True, download=True, transform=base_tf)
        base_train_aug  = datasets.CIFAR100(root=CIFAR_ROOT, train=True, download=True, transform=aug_tf)
        selected = SELECTED_CIFAR100_CLASSES
        class_names_full = base_train_eval.classes
        class_names = [class_names_full[i] for i in selected]

        full_eval_ds = RemappedSubset(base_train_eval, selected, base_train_eval.targets, transform=base_tf)
        full_aug_ds  = RemappedSubset(base_train_aug,  selected, base_train_aug.targets,  transform=aug_tf)
        num_classes = len(class_names)

        n_val = int(len(full_eval_ds) * val_split)
        idxs = torch.randperm(len(full_eval_ds)).tolist()
        val_indices = idxs[:n_val]
        tr_indices  = idxs[n_val:]

        tr_ds  = IndexMappedDataset(full_aug_ds,  tr_indices)
        val_ds = IndexMappedDataset(full_eval_ds, val_indices)

        tr_labels = [full_eval_ds.labels[i] for i in tr_indices]
        tr_loader = DataLoader(tr_ds, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

        run_suffix = f"cifar100_{num_classes}cls"
        return tr_loader, val_loader, class_names, num_classes, run_suffix

    if DATASET_NAME.lower() == "tinyimagenet":
        train_dir = os.path.join(TINY_ROOT, "train")
        val_dir   = os.path.join(TINY_ROOT, "val")
        if not (os.path.isdir(train_dir) and os.path.isdir(val_dir)):
            raise FileNotFoundError(
                f"TinyImageNet not found. Expected:\n  {train_dir}\n  {val_dir}\n"
                f"Download/unzip tiny-imagenet-200 into {TINY_ROOT}."
            )

        base_train = datasets.ImageFolder(train_dir, transform=aug_tf)
        base_val   = datasets.ImageFolder(val_dir,   transform=base_tf)

        if TINY_MAX_CLASSES is not None:
            allowed_class_ids = list(range(min(TINY_MAX_CLASSES, len(base_train.classes))))
            selected = allowed_class_ids
            class_names = [base_train.classes[i] for i in selected]
            num_classes = len(class_names)

            def filter_indices(imgfolder: datasets.ImageFolder, selected_ids: List[int]):
                selected_set = set(selected_ids)
                targets = [int(t) for _, t in imgfolder.samples]
                return [i for i, t in enumerate(targets) if t in selected_set]

            tr_keep = filter_indices(base_train, selected)
            va_keep = filter_indices(base_val, selected)

            tr_ds = Subset(base_train, tr_keep)  
            val_ds = Subset(base_val, va_keep)   

            class_map = build_class_map(selected)

            class TinyRemap(Dataset):
                def __init__(self, subset: Subset, class_map: Dict[int, int]):
                    self.subset = subset
                    self.class_map = class_map

                def __len__(self):
                    return len(self.subset)

                def __getitem__(self, idx):
                    x, y = self.subset[idx]   # type: ignore
                    return x, torch.tensor(self.class_map[int(y)], dtype=torch.long)

            tr_ds2 = TinyRemap(tr_ds, class_map)
            val_ds2 = TinyRemap(val_ds, class_map)

            tr_labels = []
            for i in range(len(tr_ds2)):
                _, y = tr_ds2[i]
                tr_labels.append(int(y))
            tr_loader = DataLoader(tr_ds2, batch_size=batch_size, sampler=create_weighted_sampler(tr_labels, num_classes))
            val_loader = DataLoader(val_ds2, batch_size=batch_size, shuffle=False)
        else:
            class_names = base_train.classes
            num_classes = len(class_names)
            tr_labels = [int(t) for _, t in base_train.samples]
            tr_loader = DataLoader(base_train, batch_size=batch_size, sampler=create_weighted_sampler(tr_labels, num_classes))
            val_loader = DataLoader(base_val, batch_size=batch_size, shuffle=False)

        run_suffix = f"tinyimagenet_{num_classes}cls"
        return tr_loader, val_loader, class_names, num_classes, run_suffix

    raise ValueError(f"Unknown DATASET_NAME={DATASET_NAME}. Choose from: cifar10, cifar100, tinyimagenet")

class Backbone(nn.Module):
    def __init__(self):
        super().__init__()

        if BACKBONE == "resnet18":
            m = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        elif BACKBONE == "resnet50":
            m = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        else:
            raise ValueError("Unknown backbone")

        self.in_feats = m.fc.in_features
        m.fc = nn.Identity()  # type: ignore
        self.body = m

    def forward(self, x):
        return self.body(x)

class CosineSoftmaxHead(nn.Module):
    def __init__(self, in_feats, num_classes: int, m=COS_M, s_init=S_INIT, learn_s=LEARNABLE_S):
        super().__init__()
        self.W = nn.Parameter(torch.randn(num_classes, in_feats))
        nn.init.xavier_normal_(self.W)
        self.m = float(m)
        self.learn_s = bool(learn_s)
        if self.learn_s:
            self.log_s = nn.Parameter(torch.log(torch.tensor(float(s_init))))
        else:
            self.s = float(s_init)

    def forward(self, feats, y=None, training=False):
        x = F.normalize(feats, p=2, dim=1, eps=1e-8)
        W = F.normalize(self.W,  p=2, dim=1, eps=1e-8)
        cos = F.linear(x, W)
        if training and y is not None:
            cos = cos.clone()
            cos[torch.arange(len(y), device=cos.device), y] -= self.m
        s = torch.exp(self.log_s) if self.learn_s else self.s
        return s * cos

class Model(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()

        self.backbone = Backbone()
        self.num_classes = num_classes

        feat_dim = self.backbone.in_feats

        self.head = CosineSoftmaxHead(feat_dim, num_classes)

        self.projector = nn.Sequential(
            nn.Linear(feat_dim, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 256)
        )

    def forward(
        self,
        x,
        return_feats=False,
        normalize_feats=False,
        y=None,
        training=False
    ):
        feats = self.backbone(x)   

        logits = self.head(
            feats,
            y=y if training else None,
            training=training
        )

        if not return_feats:
            return logits

        proj = self.projector(feats)   

        if normalize_feats:
            proj = F.normalize(proj, dim=1)

        return logits, feats, proj

def set_trainable_layers(model, epoch):
    for name, p in model.backbone.body.named_parameters():
        if epoch < 10:
            p.requires_grad = ("layer4" in name)
        elif epoch < 25:
            p.requires_grad = ("layer3" in name or "layer4" in name)
        else:
            p.requires_grad = True

def extract_backbone_features(model: Model, loader: DataLoader, device):
    model.eval()
    feats, labels = [], []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            f = model.backbone(x)   
            feats.append(f.cpu())
            labels.append(y)

    feats = torch.cat(feats).numpy()
    labels = torch.cat(labels).numpy()
    return feats, labels

def pure_state_hs2_from_feats_unit(feats_unit: torch.Tensor):
    cos = feats_unit @ feats_unit.t()
    return 2.0 * (1.0 - cos.pow(2)), cos

@torch.inference_mode()
def evaluate_topk(model: Model, loader: DataLoader, device, topk=(1,5)):
    ce = nn.CrossEntropyLoss()
    model.eval()
    tot_loss, tot = 0.0, 0
    correct1, correct5 = 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = ce(logits, y)
        tot_loss += loss.item() * y.size(0)
        tot += y.size(0)

        maxk = min(max(topk), logits.size(1))
        _, pred = logits.topk(maxk, dim=1, largest=True, sorted=True)
        pred = pred.t()  
        correct = pred.eq(y.view(1, -1))

        correct1 += int(correct[:1].reshape(-1).float().sum().item())

        k5 = min(5, logits.size(1))
        correct5 += int(correct[:k5].reshape(-1).float().sum().item())

    return tot_loss / max(tot, 1), correct1 / max(tot, 1), correct5 / max(tot, 1)

@torch.inference_mode()
def extract_features_raw(model: Model, loader: DataLoader, device) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    feats, labels = [], []
    for x, y in tqdm(loader, desc="Collecting features", leave=False):
        x = x.to(device)
        _, _, f = model(x, return_feats=True, normalize_feats=False)
        feats.append(f.cpu())
        labels.append(y)
    feats  = torch.cat(feats).numpy()
    labels = torch.cat(labels).numpy()
    return feats, labels

def compute_pure_state_hs_stats(feats_unit: np.ndarray, labels: np.ndarray) -> Tuple[float,float,float]:
    cos = feats_unit @ feats_unit.T
    hs2 = 2.0 * (1.0 - cos**2)
    y = labels[:, None]
    same = (y == y.T)
    diff = ~same
    np.fill_diagonal(same, False)
    np.fill_diagonal(diff, False)
    hs_intra = float(hs2[same].mean()) if same.any() else 0.0
    hs_inter = float(hs2[diff].mean()) if diff.any() else 0.0
    hs_ratio = hs_inter / (hs_intra + 1e-6)
    return hs_intra, hs_inter, hs_ratio

def compute_hs_stats(feats_unit: np.ndarray, labels: np.ndarray) -> Tuple[float,float,float]:
    return compute_pure_state_hs_stats(feats_unit, labels)

def fisher_multiclass_global(feats: np.ndarray, labels: np.ndarray) -> float:
    eps = 1e-6
    D = feats.shape[1]
    mu = feats.mean(axis=0, keepdims=True)
    Sw = 0.0
    Sb = 0.0
    for c in np.unique(labels):
        Xc = feats[labels == c]
        if len(Xc) < 2:
            continue
        muc = Xc.mean(axis=0, keepdims=True)
        Sw += float(((Xc - muc)**2).sum() / (len(Xc) * D))
        Sb += float(((muc - mu)**2).sum())
    return float(Sb / (Sw + eps))

def fisher_pairwise_stats(feats: np.ndarray, labels: np.ndarray) -> Tuple[float,float,float]:
    eps = 1e-6
    classes = np.unique(labels)
    mus = {}
    vars_ = {}
    for c in classes:
        Xc = feats[labels == c]
        mus[c] = Xc.mean(axis=0)
        vars_[c] = float(Xc.var(axis=0).mean())
    vals = []
    for i in range(len(classes)):
        for j in range(i+1, len(classes)):
            ci, cj = classes[i], classes[j]
            num = float(np.linalg.norm(mus[ci] - mus[cj])**2)
            den = vars_[ci] + vars_[cj] + eps
            vals.append(num / den)
    if len(vals) == 0:
        return 0.0, 0.0, 0.0
    vals = np.asarray(vals)
    return float(vals.mean()), float(vals.min()), float(vals.max())

def intra_inter_metrics(feats: np.ndarray, labels: np.ndarray) -> Tuple[float,float,float,float]:
    classes = np.unique(labels)
    centroids = []
    intra_vals = []
    for c in classes:
        Xc = feats[labels == c]
        if len(Xc) == 0:
            continue
        muc = Xc.mean(axis=0, keepdims=True)
        centroids.append(muc.squeeze(0))
        intra_vals.append(float(((Xc - muc)**2).sum(axis=1).mean()))
    intra = float(np.mean(intra_vals)) if len(intra_vals) else 0.0

    centroids = np.asarray(centroids)
    if len(centroids) < 2:
        return intra, 0.0, 0.0, 0.0

    dists = []
    for i in range(len(centroids)):
        for j in range(i+1, len(centroids)):
            dists.append(float(np.linalg.norm(centroids[i] - centroids[j])))
    dists = np.asarray(dists)
    return intra, float(dists.mean()), float(dists.min()), float(dists.max())

def cluster_overlap_percent(feats: np.ndarray, labels: np.ndarray) -> float:
    classes = np.unique(labels)
    centroids = {}
    for c in classes:
        Xc = feats[labels == c]
        centroids[c] = Xc.mean(axis=0)

    preds = []
    for x in feats:
        best_c = None
        best_d = None
        for c in classes:
            d = float(np.linalg.norm(x - centroids[c]))
            if best_d is None or d < best_d:
                best_d = d
                best_c = c
        preds.append(int(best_c))   # type: ignore
    preds = np.asarray(preds)
    acc = float((preds == labels).mean()) if len(labels) else 0.0
    return 100.0 * (1.0 - acc)

def angular_gap_metrics(model: Model, feats_raw: np.ndarray, labels: np.ndarray) -> Tuple[float,float,float]:
    W = model.head.W.detach().cpu().numpy()  
    W = W / (np.linalg.norm(W, axis=1, keepdims=True) + 1e-9)

    X = feats_raw / (np.linalg.norm(feats_raw, axis=1, keepdims=True) + 1e-9)
    cos = X @ W.T  
    cos = np.clip(cos, -1.0, 1.0)
    angles = np.arccos(cos)  

    gaps = []
    for i in range(len(labels)):
        y = int(labels[i])
        a_true = float(angles[i, y])
        
        a_other = float(np.min(np.delete(angles[i], y)))
        gaps.append(a_other - a_true)
    gaps = np.asarray(gaps) if len(gaps) else np.asarray([0.0])
    return float(gaps.mean()), float(gaps.min()), float(gaps.max())

def safe_sample(feats: np.ndarray, labels: np.ndarray, max_n: int, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    if len(labels) <= max_n:
        return feats, labels
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(labels), size=max_n, replace=False)
    return feats[idx], labels[idx]

def sample_few_shot_episode_indices(
    labels: np.ndarray,
    k_shot: int,
    query_per_class: int,
    n_way: Optional[int] = None,
    seed: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(RANDOM_SEED if seed is None else seed)

    need = k_shot + query_per_class
    feasible_classes = [
        int(c) for c in np.unique(labels)
        if int((labels == c).sum()) >= need
    ]

    if n_way is None:
        episode_classes = feasible_classes
    else:
        if n_way < 2:
            raise ValueError(f"n_way must be at least 2, got {n_way}")
        if len(feasible_classes) < n_way:
            raise ValueError(
                f"Not enough feasible classes for {n_way}-way {k_shot}-shot: "
                f"need {n_way}, got {len(feasible_classes)} classes with at least {need} samples"
            )
        episode_classes = rng.choice(feasible_classes, size=n_way, replace=False).tolist()

    support_idx = []
    query_idx = []
    for c in episode_classes:
        idx_c = np.where(labels == c)[0]
        chosen = rng.choice(idx_c, size=need, replace=False)
        support_idx.extend(chosen[:k_shot].tolist())
        query_idx.extend(chosen[k_shot:].tolist())

    return np.asarray(support_idx, dtype=int), np.asarray(query_idx, dtype=int)

def plot_pca_tsne(
    feats,
    labels,
    class_names,
    out_path,
    support_mask: Optional[np.ndarray] = None,
    title_prefix: str = ""
):
    set_pretty_rc()

    mask = np.isfinite(feats).all(axis=1)
    feats = feats[mask]
    labels = labels[mask]
    if support_mask is not None:
        support_mask = support_mask[mask]

    if feats.shape[0] < 10:
        print("Warning: too few valid points for PCA/t-SNE")
        return

    pca_2d = PCA(n_components=2, random_state=RANDOM_SEED).fit_transform(feats)
    perplexity = min(30, max(5, len(feats) - 1))
    tsne_2d = TSNE(
        n_components=2,
        perplexity=perplexity,
        init="pca",
        random_state=RANDOM_SEED,
        learning_rate="auto"
    ).fit_transform(feats)

    fig, ax = plt.subplots(1, 2, figsize=(16, 6))
    unique = np.unique(labels)

    def scatter_panel(ax_obj, coords, title):
        for c in unique:
            idx_c = labels == c
            color = palette_color(int(c), len(unique))

            if support_mask is None:
                idx_plot = idx_c

                ax_obj.scatter(
                    coords[idx_plot, 0],
                    coords[idx_plot, 1],
                    s=40,
                    alpha=0.75,
                    color=color,
                    edgecolors="white",
                    linewidths=0.5,
                    zorder=2
                )

            else:
                idx_sup = idx_c & support_mask
                idx_q = idx_c & (~support_mask)

                if idx_q.any():
                    ax_obj.scatter(
                        coords[idx_q, 0],
                        coords[idx_q, 1],
                        s=35,
                        alpha=0.65,
                        color=color,
                        edgecolors="white",
                        linewidths=0.4,
                        marker="o",
                        zorder=2
                    )

                if idx_sup.any():
                    ax_obj.scatter(
                        coords[idx_sup, 0],
                        coords[idx_sup, 1],
                        s=120,
                        alpha=1.0,
                        color=color,
                        edgecolors="black",
                        linewidths=0.8,
                        marker="*",
                        zorder=3
                    )

            if idx_c.sum() > 0:
                centroid = coords[idx_c].mean(axis=0)

                ax_obj.scatter(
                    centroid[0],
                    centroid[1],
                    s=220,
                    color=color,
                    edgecolors="black",
                    linewidths=1.5,
                    marker="X",
                    zorder=5
                )

            if idx_c.sum() > 20:
                try:
                    import scipy.stats as st

                    x = coords[idx_c, 0]
                    y = coords[idx_c, 1]

                    kde = st.gaussian_kde(np.vstack([x, y]))
                    xx, yy = np.mgrid[
                        x.min():x.max():100j,
                        y.min():y.max():100j
                    ]
                    zz = kde(np.vstack([xx.ravel(), yy.ravel()]))

                    ax_obj.contour(
                        xx, yy, zz.reshape(xx.shape),
                        levels=3,
                        colors=[color],
                        alpha=0.25,
                        linewidths=1.2,
                        zorder=1
                    )
                except:
                    pass

        full_title = f"{title_prefix}{title}" if title_prefix else title

        ax_obj.set_title(full_title, pad=18, fontweight="bold")
        ax_obj.set_xlabel("Dimension 1")
        ax_obj.set_ylabel("Dimension 2")

        ax_obj.spines["top"].set_visible(False)
        ax_obj.spines["right"].set_visible(False)

    scatter_panel(ax[0], pca_2d, "PCA Projection")
    scatter_panel(ax[1], tsne_2d, "t-SNE Projection")

    if SHOW_LEGENDS:
        handles = [
            plt.Line2D(  # type: ignore
                [0], [0],
                marker='o',
                linestyle='',
                markersize=10,
                markerfacecolor=palette_color(i, len(unique)),
                markeredgecolor="black",
                label=class_names[i]
            )
            for i in range(int(unique.max()) + 1)
            if i < len(class_names)
        ]
        if support_mask is not None:
            handles.append(
                plt.Line2D([0],[0], marker='*', linestyle='', markersize=12,  # type: ignore
                           markerfacecolor='gray', markeredgecolor='black',
                           label='support')
            )
            handles.append(
                plt.Line2D([0],[0], marker='o', linestyle='', markersize=8,  # type: ignore
                           markerfacecolor='gray', markeredgecolor='black',
                           label='query')
            )

        fig.legend(
            handles=handles,
            loc="upper center",
            bbox_to_anchor=(0.5, 1.01),
            ncol=min(10, len(handles)),
            frameon=True
        )
        plt.tight_layout(rect=[0, 0, 1, 0.93])  # type: ignore
    else:
        plt.tight_layout()

    plt.savefig(out_path, dpi=PLOT_DPI)
    plt.close()

def plot_pca_3d(feats, labels, class_names, out_path):
    set_pretty_rc()

    mask = np.isfinite(feats).all(axis=1)
    feats = feats[mask]
    labels = labels[mask]

    if feats.shape[0] < 10:
        print("Warning: too few valid points for PCA 3D")
        return

    pca_3d = PCA(n_components=3, random_state=RANDOM_SEED).fit_transform(feats)

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    unique = np.unique(labels)

    for c in unique:
        idx = labels == c
        color = palette_color(int(c), len(unique))

        ax.scatter(
            pca_3d[idx, 0],
            pca_3d[idx, 1],
            pca_3d[idx, 2],
            s=MARKER_SIZE,  # type: ignore
            alpha=MARKER_ALPHA,
            color=color,
            edgecolors="white",
            linewidths=MARKER_EDGE_WIDTH,
            marker="o",
            depthshade=True
        )

        centroid = pca_3d[idx].mean(axis=0)
        ax.scatter(
            centroid[0],
            centroid[1],
            centroid[2],
            s=220,  # type: ignore
            color=color,
            edgecolors="black",
            linewidths=1.5,
            marker="X",
            depthshade=False
        )

    format_3d_axis(
        ax,
        pca_3d,
        "3D PCA Embedding",
        xlabel="PC1",
        ylabel="PC2",
        zlabel="PC3"
    )

    plt.tight_layout()
    plt.savefig(out_path, dpi=PLOT_DPI)
    plt.close()

def plot_tsne_3d(feats, labels, class_names, out_path):
    set_pretty_rc()

    mask = np.isfinite(feats).all(axis=1)
    feats = feats[mask]
    labels = labels[mask]

    if feats.shape[0] < 10:
        print("Warning: too few valid points for t-SNE 3D")
        return

    perplexity = min(30, max(5, len(feats) - 1))

    tsne_3d = TSNE(
        n_components=3,
        perplexity=perplexity,
        init="pca",
        random_state=RANDOM_SEED,
        learning_rate="auto"
    ).fit_transform(feats)

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    unique = np.unique(labels)

    for c in unique:
        idx = labels == c
        color = palette_color(int(c), len(unique))

        ax.scatter(
            tsne_3d[idx, 0],
            tsne_3d[idx, 1],
            tsne_3d[idx, 2],
            s=MARKER_SIZE,  # type: ignore
            alpha=MARKER_ALPHA,
            color=color,
            edgecolors="white",
            linewidths=MARKER_EDGE_WIDTH,
            marker="o",
            depthshade=True
        )

        centroid = tsne_3d[idx].mean(axis=0)
        ax.scatter(
            centroid[0],
            centroid[1],
            centroid[2],
            s=220,  # type: ignore
            color=color,
            edgecolors="black",
            linewidths=1.5,
            marker="X",
            depthshade=False
        )

    format_3d_axis(
        ax,
        tsne_3d,
        "3D t-SNE Embedding",
        xlabel="Dimension 1",
        ylabel="Dimension 2",
        zlabel="Dimension 3"
    )

    plt.tight_layout()
    plt.savefig(out_path, dpi=PLOT_DPI)
    plt.close()

def plot_pca_tsne_3d(
    feats,
    labels,
    class_names,
    out_path,
    support_mask=None,
    title_prefix=""
):
    set_pretty_rc()

    mask = np.isfinite(feats).all(axis=1)
    feats = feats[mask]
    labels = labels[mask]

    if support_mask is not None:
        support_mask = support_mask[mask]

    if feats.shape[0] < 10:
        print("Warning: too few valid points for 3D PCA/t-SNE")
        return

    pca_3d = PCA(n_components=3, random_state=RANDOM_SEED).fit_transform(feats)

    perplexity = min(30, max(5, len(feats) - 1))
    tsne_3d = TSNE(
        n_components=3,
        perplexity=perplexity,
        init="pca",
        random_state=RANDOM_SEED,
        learning_rate="auto"
    ).fit_transform(feats)

    fig = plt.figure(figsize=(14, 6))
    ax1 = fig.add_subplot(121, projection="3d")
    ax2 = fig.add_subplot(122, projection="3d")

    unique = np.unique(labels)

    def scatter_3d(ax, coords, title):
        for c in unique:
            idx_c = labels == c
            color = palette_color(int(c), len(unique))

            if support_mask is None:
                ax.scatter(
                    coords[idx_c, 0],
                    coords[idx_c, 1],
                    coords[idx_c, 2],
                    s=MARKER_SIZE,
                    alpha=MARKER_ALPHA,
                    color=color,
                    edgecolors="white",
                    linewidths=MARKER_EDGE_WIDTH,
                    marker="o",
                    depthshade=True,
                    zorder=2
                )
            else:
                idx_sup = idx_c & support_mask
                idx_q = idx_c & (~support_mask)

                if idx_q.any():
                    ax.scatter(
                        coords[idx_q, 0],
                        coords[idx_q, 1],
                        coords[idx_q, 2],
                        s=MARKER_SIZE,
                        alpha=0.65,
                        color=color,
                        edgecolors="white",
                        linewidths=MARKER_EDGE_WIDTH,
                        marker="o",
                        depthshade=True,
                        zorder=2
                    )

                if idx_sup.any():
                    ax.scatter(
                        coords[idx_sup, 0],
                        coords[idx_sup, 1],
                        coords[idx_sup, 2],
                        s=120,
                        alpha=1.0,
                        color=color,
                        edgecolors="black",
                        linewidths=0.8,
                        marker="*",
                        depthshade=True,
                        zorder=3
                    )

            if idx_c.sum() > 0:
                centroid = coords[idx_c].mean(axis=0)
                ax.scatter(
                    centroid[0],
                    centroid[1],
                    centroid[2],
                    s=220,
                    color=color,
                    edgecolors="black",
                    linewidths=1.5,
                    marker="X",
                    depthshade=False,
                    zorder=5
                )

        format_3d_axis(
            ax,
            coords,
            f"{title_prefix}{title}",
            xlabel="Dimension 1",
            ylabel="Dimension 2",
            zlabel="Dimension 3"
        )

    scatter_3d(ax1, pca_3d, "PCA Projection 3D")
    scatter_3d(ax2, tsne_3d, "t-SNE Projection 3D")

    if SHOW_LEGENDS:
        handles = [
            plt.Line2D(  # type: ignore
                [0], [0],
                marker="o",
                linestyle="",
                markersize=9,
                markerfacecolor=palette_color(i, len(unique)),
                markeredgecolor="black",
                label=class_names[i]
            )
            for i in range(int(unique.max()) + 1)
            if i < len(class_names)
        ]

        handles.append(
            plt.Line2D(  # type: ignore
                [0], [0],
                marker="X",
                linestyle="",
                markersize=10,
                markerfacecolor="gray",
                markeredgecolor="black",
                label="centroid"
            )
        )

        if support_mask is not None:
            handles.append(
                plt.Line2D(  # type: ignore
                    [0], [0],
                    marker="*",
                    linestyle="",
                    markersize=12,
                    markerfacecolor="gray",
                    markeredgecolor="black",
                    label="support"
                )
            )
            handles.append(
                plt.Line2D(  # type: ignore
                    [0], [0],
                    marker="o",
                    linestyle="",
                    markersize=8,
                    markerfacecolor="gray",
                    markeredgecolor="black",
                    label="query"
                )
            )

        fig.legend(
            handles=handles,
            loc="upper center",
            bbox_to_anchor=(0.5, 1.02),
            ncol=min(10, len(handles)),
            frameon=True
        )

        plt.tight_layout(rect=[0, 0, 1, 0.93])  # type: ignore
    else:
        plt.tight_layout()

    plt.savefig(out_path, dpi=PLOT_DPI)
    plt.close()

def plot_few_shot_episode(
    feats: np.ndarray,
    labels: np.ndarray,
    class_names: List[str],
    k_shot: int,
    out_path: str,
    n_way: Optional[int] = FEW_SHOT_N_WAY,
    seed: int = RANDOM_SEED
):
    support_idx, query_idx = sample_few_shot_episode_indices(
        labels, k_shot, FEW_SHOT_QUERY_PER_CLASS, n_way=n_way, seed=seed
    )
    episode_idx = np.concatenate([support_idx, query_idx])
    episode_feats = feats[episode_idx]
    episode_labels = labels[episode_idx]
    support_mask = np.zeros(len(episode_idx), dtype=bool)
    support_mask[:len(support_idx)] = True

    plot_pca_tsne(
        episode_feats,
        episode_labels,
        class_names,
        out_path,
        support_mask=support_mask,
        title_prefix=f"{n_way if n_way is not None else len(np.unique(labels))}-way {k_shot}-shot episode: "
    )

def build_batch_class_density_operators(
    feats_unit: torch.Tensor,
    labels: torch.Tensor,
    num_classes: int,
    min_samples_per_class: int = QIT_MIN_CLASS_SAMPLES,
    trace_normalize: bool = QIT_TRACE_NORMALIZE,
    eps: float = 1e-9,
):
    rhos = []
    class_ids = []
    counts = []

    for c in range(num_classes):
        mask = (labels == c)
        n_c = int(mask.sum().item())
        if n_c < min_samples_per_class:
            continue

        z = feats_unit[mask]                         
        rho = (z.t() @ z) / float(n_c)             
        rho = 0.5 * (rho + rho.t())                 

        if trace_normalize:
            rho = rho / (torch.trace(rho) + eps)

        rhos.append(rho)
        class_ids.append(c)
        counts.append(n_c)

    if len(rhos) == 0:
        return None, class_ids, counts

    return torch.stack(rhos, dim=0), class_ids, counts

def qit_density_operator_loss(
    feats_unit: torch.Tensor,
    labels: torch.Tensor,
    num_classes: int,
):
    device = feats_unit.device
    rhos, class_ids, counts = build_batch_class_density_operators(
        feats_unit, labels, num_classes
    )

    zero = torch.zeros((), device=device, dtype=feats_unit.dtype)
    if rhos is None or rhos.size(0) < 2:
        stats = {
            "purity_mean": zero,
            "overlap_mean": zero,
            "class_hs2_mean": zero,
            "equiv_error": zero,
            "num_class_states": 0 if rhos is None else int(rhos.size(0)),
        }
        return zero, stats

    purities = torch.einsum("cij,cji->c", rhos, rhos)
    purity_mean = purities.mean()

    overlaps = []
    hs2_values = []
    for i in range(rhos.size(0)):
        for j in range(i + 1, rhos.size(0)):
            overlap_ij = torch.trace(rhos[i] @ rhos[j])
            overlaps.append(overlap_ij)
            hs2_values.append(purities[i] + purities[j] - 2.0 * overlap_ij)

    overlap_mean = torch.stack(overlaps).mean()
    class_hs2_mean = torch.stack(hs2_values).mean()

    loss_qit = (1.0 - purity_mean) + overlap_mean

    loss_hs_equiv = 1.0 - 0.5 * class_hs2_mean
    equiv_error = torch.abs(loss_qit - loss_hs_equiv)

    stats = {
        "purity_mean": purity_mean,
        "overlap_mean": overlap_mean,
        "class_hs2_mean": class_hs2_mean,
        "equiv_error": equiv_error,
        "num_class_states": int(rhos.size(0)),
    }
    return loss_qit, stats

def train(model: Model, tr_loader: DataLoader, val_loader: DataLoader, device):
    ce = nn.CrossEntropyLoss()
    
    model.to(device)
    
    opt_model = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LR,
        weight_decay=WEIGHT_DECAY
    )

    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt_model, T_max=EPOCHS)

    metrics_log = []

    start_time = time.time()
    epoch_times = []

    for ep in range(1, EPOCHS + 1):
        set_trainable_layers(model, ep)
        ep_start = time.time()
        model.train()

        ce_sum = qit_sum = qit_purity_sum = qit_overlap_sum = qit_hs2_sum = qit_equiv_sum = 0.0
        loss_sum = 0.0
        seen = 0
        correct1 = 0

        for x, y in tr_loader:
            x, y = x.to(device), y.to(device)

            opt_model.zero_grad()

            logits, feats_backbone, feats_proj = model(
                x,
                return_feats=True,
                normalize_feats=False,
                y=y,
                training=True
            )

            loss_ce = ce(logits, y)
            loss = loss_ce

            if USE_QIT_LOSS:
                feats_unit = F.normalize(feats_proj, dim=1)
                L_qit, qit_stats = qit_density_operator_loss(
                    feats_unit, y, model.num_classes
                )
                loss = loss + LAMBDA_QIT * L_qit

                batch_n = y.size(0)
                qit_sum += float(L_qit.detach().cpu()) * batch_n
                qit_purity_sum += float(qit_stats["purity_mean"].detach().cpu()) * batch_n
                qit_overlap_sum += float(qit_stats["overlap_mean"].detach().cpu()) * batch_n
                qit_hs2_sum += float(qit_stats["class_hs2_mean"].detach().cpu()) * batch_n
                qit_equiv_sum += float(qit_stats["equiv_error"].detach().cpu()) * batch_n

            loss.backward()

            opt_model.step()

            ce_sum   += float(loss_ce.detach().cpu()) * y.size(0)
            loss_sum += float(loss.detach().cpu())    * y.size(0)
            seen     += y.size(0)
            correct1 += int((logits.argmax(1) == y).sum().item())

        sch.step()

        tr_ce   = ce_sum / max(seen, 1)
        tr_loss = loss_sum / max(seen, 1)
        tr_top1 = correct1 / max(seen, 1)

        vl_loss, vl_top1, vl_top5 = evaluate_topk(
            model, val_loader, device, topk=(1, 5)
        )

        feats_list, labs_list = [], []
        with torch.no_grad():
            for x_val, y_val in val_loader:
                _, _, f_val = model(
                    x_val.to(device),
                    return_feats=True,
                    normalize_feats=False
                )
                feats_list.append(F.normalize(f_val, dim=1).cpu())
                labs_list.append(y_val)

        feats_unit = torch.cat(feats_list).numpy()
        labs = torch.cat(labs_list).numpy()
        hs_intra, hs_inter, hs_ratio = compute_pure_state_hs_stats(feats_unit, labs)

        feats_raw, labs_raw = extract_features_raw(model, val_loader, device)
        
        density_metrics = compute_density_spectrum_metrics(
            feats_raw,
            labs_raw
        )

        mean_entropy = np.mean(
            [v["entropy"] for v in density_metrics.values()]
        )

        std_entropy = np.std(
            [v["entropy"] for v in density_metrics.values()]
        )

        mean_r = np.mean(
            [v["dominant_ratio"] for v in density_metrics.values()]
        )

        std_r = np.std(
            [v["dominant_ratio"] for v in density_metrics.values()]
        )

        mean_erank = np.mean(
            [v["effective_rank"] for v in density_metrics.values()]
        )

        mean_purity = np.mean(
            [v["purity"] for v in density_metrics.values()]
        )

        mean_top5 = np.mean(
            [v["top5_energy"] for v in density_metrics.values()]
        )

        operator_overlap_metrics = compute_operator_overlap_metrics(
            feats_raw,
            labs_raw
        )

        qit_val_metrics = compute_qit_class_state_metrics(
            feats_raw,
            labs_raw
        )

        fisher_global_raw = fisher_multiclass_global(feats_raw, labs_raw)
        fisher_mean_raw, fisher_min_raw, fisher_max_raw = fisher_pairwise_stats(
            feats_raw, labs_raw
        )

        feats_norm = feats_raw / (np.linalg.norm(feats_raw, axis=1, keepdims=True) + 1e-9)
        fisher_global_norm = fisher_multiclass_global(feats_norm, labs_raw)
        fisher_mean_norm, fisher_min_norm, fisher_max_norm = fisher_pairwise_stats(
            feats_norm, labs_raw
        )

        ep_time = time.time() - ep_start
        epoch_times.append(ep_time)

        metrics_log.append({
            "epoch": ep,
            "CE": tr_ce,
            "QIT_LOSS": (qit_sum / max(seen, 1)) if USE_QIT_LOSS else None,
            "QIT_PURITY_TRAIN": (qit_purity_sum / max(seen, 1)) if USE_QIT_LOSS else None,
            "QIT_OVERLAP_TRAIN": (qit_overlap_sum / max(seen, 1)) if USE_QIT_LOSS else None,
            "QIT_CLASS_HS2_TRAIN": (qit_hs2_sum / max(seen, 1)) if USE_QIT_LOSS else None,
            "QIT_EQUIV_ERROR": (qit_equiv_sum / max(seen, 1)) if USE_QIT_LOSS else None,

            "STATE_HS2_INTRA": hs_intra,
            "STATE_HS2_INTER": hs_inter,
            "STATE_HS2_RATIO": hs_ratio,

            "HS_INTRA": hs_intra,
            "HS_INTER": hs_inter,
            "HS_RATIO": hs_ratio,

            "FISHER_GLOBAL_RAW": fisher_global_raw,
            "FISHER_PAIR_MEAN_RAW": fisher_mean_raw,
            "FISHER_PAIR_MIN_RAW": fisher_min_raw,
            "FISHER_PAIR_MAX_RAW": fisher_max_raw,

            "FISHER_GLOBAL_NORM": fisher_global_norm,
            "FISHER_PAIR_MEAN_NORM": fisher_mean_norm,
            "FISHER_PAIR_MIN_NORM": fisher_min_norm,
            "FISHER_PAIR_MAX_NORM": fisher_max_norm,

            "QIT_PURITY_VAL": qit_val_metrics["QIT_PURITY"],
            "QIT_OVERLAP_VAL": qit_val_metrics["QIT_OVERLAP"],
            "QIT_CLASS_HS2_VAL": qit_val_metrics["QIT_CLASS_HS2"],
            "QIT_OBJECTIVE_VAL": qit_val_metrics["QIT_OBJECTIVE"],
            "QIT_EQUIV_ERROR_VAL": qit_val_metrics["QIT_EQUIV_ERROR"],

            "MEAN_VN_ENTROPY": mean_entropy,
            "STD_VN_ENTROPY": std_entropy,

            "MEAN_DOMINANT_RATIO": mean_r,
            "STD_DOMINANT_RATIO": std_r,

            "MEAN_EFFECTIVE_RANK": mean_erank,

            "MEAN_PURITY": mean_purity,

            "MEAN_TOP5_ENERGY": mean_top5,

            "OP_OVERLAP_MEAN": operator_overlap_metrics["OP_OVERLAP_MEAN"],
            "OP_OVERLAP_MIN": operator_overlap_metrics["OP_OVERLAP_MIN"],
            "OP_OVERLAP_MAX": operator_overlap_metrics["OP_OVERLAP_MAX"],

            "Train_Top1": tr_top1,
            "Train_Loss": tr_loss,
            "Val_Top1": vl_top1,
            "Val_Top5": vl_top5,
            "Val_Loss": vl_loss,
            "Epoch_Time_sec": ep_time,
        })

    avg_epoch_time = sum(epoch_times) / len(epoch_times)
    total_time = time.time() - start_time

    return model, metrics_log, avg_epoch_time

def extract_features_from_dataset(model: Model, dataset: Dataset, device, batch_size=BATCH_SIZE) -> Tuple[np.ndarray, np.ndarray]:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    return extract_features_raw(model, loader, device)

def compute_class_prototypes(feats: np.ndarray, labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    classes = np.unique(labels)
    prototypes = []
    for c in classes:
        prototypes.append(feats[labels == c].mean(axis=0))
    prototypes = np.asarray(prototypes)
    return classes, prototypes

def prototype_predict(
    support_feats: np.ndarray,
    support_labels: np.ndarray,
    query_feats: np.ndarray,
    normalize: bool = True,
) -> np.ndarray:
    if normalize:
        support_feats = support_feats / (np.linalg.norm(support_feats, axis=1, keepdims=True) + 1e-9)
        query_feats = query_feats / (np.linalg.norm(query_feats, axis=1, keepdims=True) + 1e-9)

    classes, prototypes = compute_class_prototypes(support_feats, support_labels)

    if normalize:
        prototypes = prototypes / (np.linalg.norm(prototypes, axis=1, keepdims=True) + 1e-9)
        sims = query_feats @ prototypes.T
        pred_idx = np.argmax(sims, axis=1)
    else:
        dists = ((query_feats[:, None, :] - prototypes[None, :, :]) ** 2).sum(axis=2)
        pred_idx = np.argmin(dists, axis=1)

    return classes[pred_idx]

def run_few_shot_evaluation(
    model: Model,
    eval_dataset: Dataset,
    device,
    shots: List[int] = FEW_SHOT_KS,
    episodes: int = FEW_SHOT_EPISODES,
    n_way: Optional[int] = FEW_SHOT_N_WAY,
    query_per_class: int = FEW_SHOT_QUERY_PER_CLASS,
    normalize: bool = FEW_SHOT_USE_NORMALIZED,
) -> Dict[str, Dict[str, float]]:
    feats_raw, labels = extract_features_from_dataset(model, eval_dataset, device)
    rng = np.random.default_rng(RANDOM_SEED)

    class_to_indices: Dict[int, np.ndarray] = {}
    for c in np.unique(labels):
        class_to_indices[int(c)] = np.where(labels == c)[0]

    results: Dict[str, Dict[str, float]] = {}

    for k in shots:

        need = k + query_per_class
        feasible_classes = [
            c for c, idx in class_to_indices.items()
            if len(idx) >= need
        ]

        if n_way is None:
            episode_n_way = len(feasible_classes)
            if episode_n_way < 2:
                result_key = f"all-way {k}-shot"
                results[result_key] = {
                    "episodes": 0,
                    "n_way": int(episode_n_way),
                    "support_per_class": int(k),
                    "support_total": 0,
                    "query_per_class": int(query_per_class),
                    "query_total": 0,
                    "mean_acc": float("nan"),
                    "std_acc": float("nan"),
                }
                continue
        else:
            episode_n_way = int(n_way)
            if episode_n_way < 2 or len(feasible_classes) < episode_n_way:
                result_key = f"{episode_n_way}-way {k}-shot"
                results[result_key] = {
                    "episodes": 0,
                    "n_way": int(episode_n_way),
                    "support_per_class": int(k),
                    "support_total": 0,
                    "query_per_class": int(query_per_class),
                    "query_total": 0,
                    "mean_acc": float("nan"),
                    "std_acc": float("nan"),
                }
                continue

        accs = []
        for _ in range(episodes):
            if n_way is None:
                episode_classes = feasible_classes
            else:
                episode_classes = rng.choice(
                    feasible_classes,
                    size=episode_n_way,
                    replace=False
                ).tolist()

            support_idx = []
            query_idx = []
            for c in episode_classes:
                chosen = rng.choice(
                    class_to_indices[int(c)],
                    size=k + query_per_class,
                    replace=False
                )
                support_idx.extend(chosen[:k].tolist())
                query_idx.extend(chosen[k:].tolist())

            support_idx = np.asarray(support_idx, dtype=int)
            query_idx = np.asarray(query_idx, dtype=int)

            pred = prototype_predict(
                feats_raw[support_idx],
                labels[support_idx],
                feats_raw[query_idx],
                normalize=normalize,
            )
            acc = float((pred == labels[query_idx]).mean())
            accs.append(acc)

        accs_np = np.asarray(accs, dtype=np.float64)
        result_key = f"{episode_n_way}-way {k}-shot" if n_way is not None else f"all-way {k}-shot"
        results[result_key] = {
            "episodes": int(len(accs_np)),
            "n_way": int(episode_n_way),
            "support_per_class": int(k),
            "support_total": int(episode_n_way * k),
            "query_per_class": int(query_per_class),
            "query_total": int(episode_n_way * query_per_class),
            "mean_acc": float(accs_np.mean()),
            "std_acc": float(accs_np.std(ddof=1) if len(accs_np) > 1 else 0.0),
        }

    return results

def print_few_shot_report(results: Dict[str, Dict[str, float]]):
    print("\n==================== Few-Shot Evaluation ====================\n")
    for shot_name, stats in results.items():
        if int(stats["episodes"]) == 0:
            print(f"{shot_name:>12}: skipped (not enough samples per class)")
            continue
        print(
            f"{shot_name:>12}: "
            f"mean_acc={100.0 * stats['mean_acc']:.4f}% | "
            f"std={100.0 * stats['std_acc']:.4f}% | "
            f"episodes={int(stats['episodes'])} | "
            f"way={int(stats.get('n_way', 0))} | "
            f"support={int(stats.get('support_total', 0))} | "
            f"query/class={int(stats['query_per_class'])}"
        )
    print("\n=============================================================\n")

def save_few_shot_csv(results: Dict[str, Dict[str, float]], out_path: str):
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "setting", "episodes", "n_way", "support_per_class",
            "support_total", "query_per_class", "query_total",
            "mean_acc", "std_acc"
        ])
        for shot_name, stats in results.items():
            writer.writerow([
                shot_name,
                int(stats["episodes"]),
                int(stats.get("n_way", 0)),
                int(stats.get("support_per_class", 0)),
                int(stats.get("support_total", 0)),
                int(stats["query_per_class"]),
                int(stats.get("query_total", 0)),
                float(stats["mean_acc"]),
                float(stats["std_acc"]),
            ])

def set_pretty_rc():
    if not PRETTY_PLOTS:
        return

    sns.set_theme(style="whitegrid", context="talk")

    plt.rcParams.update({
        "figure.figsize": (8, 6),
        "figure.dpi": PLOT_DPI,

        "axes.facecolor": "#FAFAFA",
        "axes.edgecolor": "#333333",
        "axes.linewidth": 1.0,

        "grid.alpha": 0.3,
        "grid.linestyle": "--",

        "font.family": "DejaVu Sans",
        "font.size": 13,

        "axes.titlesize": 18,
        "axes.titleweight": "bold",

        "axes.labelsize": 14,

        "legend.frameon": True,
        "legend.framealpha": 0.9,

        "lines.linewidth": 2.5,
        "lines.markersize": 6,
    })
    
def palette_color(i, n_classes=10):
    cmap = sns.color_palette("husl", n_classes)
    return cmap[i % n_classes]

def save_metrics_csv(metrics_log, path):
    fieldnames = list(metrics_log[0].keys()) if len(metrics_log) else []
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in metrics_log:
            w.writerow(row)

def smooth(x, k=5):
    x = np.asarray(x, dtype=float)
    k = min(k, len(x))
    if k <= 1:
        return x
    return np.convolve(x, np.ones(k) / k, mode="same")

def plot_loss_acc(metrics_log, out_path):
    set_pretty_rc()
    epochs = [m["epoch"] for m in metrics_log]
    tr_loss = [m["Train_Loss"] for m in metrics_log]
    vl_loss = [m["Val_Loss"] for m in metrics_log]
    tr_top1 = [m["Train_Top1"] for m in metrics_log]
    vl_top1 = [m["Val_Top1"] for m in metrics_log]
    vl_top5 = [m["Val_Top5"] for m in metrics_log]

    fig, ax = plt.subplots(1, 2, figsize=(12, 4))

    ax[0].plot(epochs, smooth(tr_loss), label="Train", color="#1f77b4")
    ax[0].plot(epochs, smooth(vl_loss), label="Val", color="#d62728")
    ax[0].set_title("Loss", pad=18)
    ax[0].set_xlabel("Epoch")
    ax[0].set_ylabel("CE loss")
    ax[0].legend(["Train", "Val"], frameon=True)

    ax[1].plot(epochs, smooth(tr_top1), label="Train Top-1", color="#1f77b4")
    ax[1].plot(epochs, smooth(vl_top1), label="Val Top-1", color="#d62728")
    ax[1].plot(epochs, smooth(vl_top5), label="Val Top-5", color="#2ca02c")
    ax[1].set_title("Accuracy (Top-1 / Top-5)", pad=18)
    ax[1].set_xlabel("Epoch")
    ax[1].set_ylabel("Accuracy")
    ax[1].legend(["Train Top-1", "Val Top-1", "Val Top-5"], frameon=True)

    plt.tight_layout()
    plt.savefig(out_path, dpi=PLOT_DPI)
    plt.close()

def plot_confusion(cm, class_names, out_path):
    set_pretty_rc()

    plt.figure(figsize=(8, 6))

    cm_norm = cm / (cm.sum(axis=1, keepdims=True) + 1e-9)

    sns.heatmap(
        cm_norm,
        annot=True,
        fmt=".2f",
        cmap="mako",
        xticklabels=class_names,
        yticklabels=class_names,
        linewidths=0.5,
        linecolor="white"
    )

    plt.title("Normalized Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")

    plt.tight_layout()
    plt.savefig(out_path, dpi=PLOT_DPI)
    plt.close()

def state_hs_separation_bar(feats_unit: np.ndarray, labels: np.ndarray, out_txt: str, out_png: str):
    set_pretty_rc()
    hs_intra, hs_inter, _ = compute_pure_state_hs_stats(feats_unit, labels)
    sep_index = hs_inter - hs_intra

    with open(out_txt, "w") as f:
        f.write(f"Pure-state same-class mean D_HS^2: {hs_intra:.6f}\n")
        f.write(f"Pure-state diff-class mean D_HS^2: {hs_inter:.6f}\n")
        f.write(f"Separation index (diff - same): {sep_index:.6f}\n")

    plt.figure(figsize=(5, 4))
    plt.bar(["same", "diff"], [hs_intra, hs_inter])
    plt.ylabel("Mean pure-state $D_{HS}^2$")
    plt.title("Pure-State Hilbert-Schmidt Separation", pad=18)
    plt.tight_layout()
    plt.savefig(out_png, dpi=PLOT_DPI)
    plt.close()

def final_eval_report(model: Model, val_loader: DataLoader, device) -> Dict[str, float]:
    
    val_loss, top1, top5 = evaluate_topk(model, val_loader, device, topk=(1,5))

    feats_backbone, labels = extract_backbone_features(model, val_loader, device)

    feats_raw, _ = extract_features_raw(model, val_loader, device)

    feats_m, labels_m = safe_sample(feats_raw, labels, METRICS_MAX_POINTS, RANDOM_SEED)

    fisher_global_raw = fisher_multiclass_global(feats_raw, labels)
    fisher_mean_raw, fisher_min_raw, fisher_max_raw = fisher_pairwise_stats(
        feats_raw, labels
    )

    feats_norm = feats_raw / (np.linalg.norm(feats_raw, axis=1, keepdims=True) + 1e-9)

    fisher_global_norm = fisher_multiclass_global(feats_norm, labels)
    fisher_mean_norm, fisher_min_norm, fisher_max_norm = fisher_pairwise_stats(
        feats_norm, labels
    )

    intra_var, inter_mean, inter_min, inter_max = intra_inter_metrics(feats_raw, labels)

    overlap_pct = cluster_overlap_percent(feats_raw, labels)

    sil = 0.0
    if len(np.unique(labels_m)) >= 2:
        sil = float(silhouette_score(feats_m, labels_m, metric="cosine"))

    db = 0.0
    if len(np.unique(labels_m)) >= 2:
        db = float(davies_bouldin_score(feats_m, labels_m))

    if hasattr(model.head, "W"):
        ang_gap_mean, ang_gap_min, ang_gap_max = angular_gap_metrics(model, feats_backbone, labels)
    else:
        ang_gap_mean, ang_gap_min, ang_gap_max = (float("nan"), float("nan"), float("nan"))

    feats_unit = feats_raw / (np.linalg.norm(feats_raw, axis=1, keepdims=True) + 1e-9)
    hs_intra, hs_inter, hs_ratio = compute_pure_state_hs_stats(feats_unit, labels)
    
    overlap_pct = cluster_overlap_percent(feats_raw, labels)
    centroid_metrics_raw = centroid_evaluation_metrics(
        feats_raw,
        labels,
        normalize=False
    )

    centroid_metrics_norm = centroid_evaluation_metrics(
        feats_raw,
        labels,
        normalize=True
    )

    report = {
        "Top1_Acc_%": 100.0 * float(top1),
        "Top5_Acc_%": 100.0 * float(top5),
        "Val_Loss": float(val_loss),

        "FISHER_GLOBAL_NORM": float(fisher_global_norm),
        "FISHER_PAIR_MEAN_NORM": float(fisher_mean_norm),
        "FISHER_PAIR_MIN_NORM": float(fisher_min_norm),
        "FISHER_PAIR_MAX_NORM": float(fisher_max_norm),

        "FISHER_GLOBAL_RAW": float(fisher_global_raw),
        "FISHER_PAIR_MEAN_RAW": float(fisher_mean_raw),
        "FISHER_PAIR_MIN_RAW": float(fisher_min_raw),
        "FISHER_PAIR_MAX_RAW": float(fisher_max_raw),

        "Silhouette_Cosine": float(sil),
        "DaviesBouldin_Euclid": float(db),

        "IntraClass_Var": float(intra_var),
        "InterClassDist_Mean": float(inter_mean),
        "InterClassDist_Min": float(inter_min),
        "InterClassDist_Max": float(inter_max),

        "ClusterOverlap_%": float(overlap_pct),

        "COS_M": float(COS_M),
        "AngularGap_MeanRad": float(ang_gap_mean),
        "AngularGap_MinRad": float(ang_gap_min),
        "AngularGap_MaxRad": float(ang_gap_max),

        "STATE_HS2_INTRA": float(hs_intra),
        "STATE_HS2_INTER": float(hs_inter),
        "STATE_HS2_RATIO": float(hs_ratio),

        "HS_INTRA": float(hs_intra),
        "HS_INTER": float(hs_inter),
        "HS_RATIO": float(hs_ratio),

        "Centroid_NCA_Acc_RAW_%": centroid_metrics_raw["Centroid_NCA_Acc_%"],
        "Centroid_Overlap_RAW_%": centroid_metrics_raw["Centroid_Overlap_%"],
        "Centroid_IntraDist_RAW_Mean": centroid_metrics_raw["Centroid_IntraDist_Mean"],
        "Centroid_InterDist_RAW_Mean": centroid_metrics_raw["Centroid_InterDist_Mean"],
        "Centroid_InterDist_RAW_Min": centroid_metrics_raw["Centroid_InterDist_Min"],
        "Centroid_SeparationRatio_RAW": centroid_metrics_raw["Centroid_SeparationRatio"],
        "Centroid_Margin_RAW_Mean": centroid_metrics_raw["Centroid_Margin_Mean"],
        "Centroid_Margin_RAW_Min": centroid_metrics_raw["Centroid_Margin_Min"],

        "Centroid_NCA_Acc_NORM_%": centroid_metrics_norm["Centroid_NCA_Acc_%"],
        "Centroid_Overlap_NORM_%": centroid_metrics_norm["Centroid_Overlap_%"],
        "Centroid_IntraDist_NORM_Mean": centroid_metrics_norm["Centroid_IntraDist_Mean"],
        "Centroid_InterDist_NORM_Mean": centroid_metrics_norm["Centroid_InterDist_Mean"],
        "Centroid_InterDist_NORM_Min": centroid_metrics_norm["Centroid_InterDist_Min"],
        "Centroid_SeparationRatio_NORM": centroid_metrics_norm["Centroid_SeparationRatio"],
        "Centroid_Margin_NORM_Mean": centroid_metrics_norm["Centroid_Margin_Mean"],
        "Centroid_Margin_NORM_Min": centroid_metrics_norm["Centroid_Margin_Min"],
        "Centroid_Cosine_NORM_Mean": centroid_metrics_norm["Centroid_Cosine_Mean"],
        "Centroid_Cosine_NORM_Max": centroid_metrics_norm["Centroid_Cosine_Max"],
    }
    return report

def print_final_report(report: Dict[str, float], train_top1: float, train_loss: float):
    
    print("\n==================== Final Evaluation Metrics ====================\n")
    print(f"{'Train_Top1_%':>22}: {100.0*train_top1:.4f}")
    print(f"{'Train_Loss':>22}: {train_loss:.6f}")
    print(f"{'Top1_Acc_%':>22}: {report['Top1_Acc_%']:.4f}")
    print(f"{'Top5_Acc_%':>22}: {report['Top5_Acc_%']:.4f}")
    print(f"{'Val_Loss':>22}: {report['Val_Loss']:.6f}")
    
    print("")
    for k in [
        "FISHER_GLOBAL_NORM",
        "FISHER_PAIR_MEAN_NORM",
        "FISHER_PAIR_MIN_NORM",
        "FISHER_PAIR_MAX_NORM",

        "FISHER_GLOBAL_RAW",

        "Silhouette_Cosine",
        "DaviesBouldin_Euclid",

        "IntraClass_Var",
        "InterClassDist_Mean",
        "InterClassDist_Min",
        "InterClassDist_Max",
        "ClusterOverlap_%",

        "Centroid_NCA_Acc_NORM_%",
        "Centroid_Overlap_NORM_%",
        "Centroid_IntraDist_NORM_Mean",
        "Centroid_InterDist_NORM_Mean",
        "Centroid_InterDist_NORM_Min",
        "Centroid_SeparationRatio_NORM",
        "Centroid_Margin_NORM_Mean",
        "Centroid_Margin_NORM_Min",
        "Centroid_Cosine_NORM_Mean",
        "Centroid_Cosine_NORM_Max",

        "Centroid_NCA_Acc_RAW_%",
        "Centroid_Overlap_RAW_%",
        "Centroid_SeparationRatio_RAW",
        "Centroid_Margin_RAW_Mean",

        "COS_M",
        "AngularGap_MeanRad",
        "AngularGap_MinRad",
        "AngularGap_MaxRad",

        "STATE_HS2_INTRA",
        "STATE_HS2_INTER",
        "STATE_HS2_RATIO",
    ]:
        print(f"{k:>30}: {report[k]:.6f}")
    print("\n===================================================================\n")

def format_3d_axis(ax, coords, title, xlabel="Dim 1", ylabel="Dim 2", zlabel="Dim 3"):
    ax.set_title(title, pad=18, fontweight="bold")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_zlabel(zlabel)

    ax.set_box_aspect([1, 1, 0.8])
    expand_limits(ax, coords)

    ax.view_init(elev=30, azim=135)

    try:
        ax.dist = 12
    except Exception:
        pass

    for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
        axis.pane.fill = True
        axis.pane.set_facecolor((0.98, 0.98, 0.98, 1.0))
        axis.pane.set_edgecolor((0.85, 0.85, 0.85, 1.0))

    ax.grid(True, linestyle="--", alpha=0.30, linewidth=0.6)

def compute_density_spectrum_metrics(
    feats: np.ndarray,
    labels: np.ndarray,
):
    feats = feats / (
        np.linalg.norm(feats, axis=1, keepdims=True) + 1e-9
    )

    metrics = {}

    for c in np.unique(labels):

        Z = feats[labels == c]

        if len(Z) < 2:
            continue

        rho = (Z.T @ Z) / len(Z)

        rho = 0.5 * (rho + rho.T)

        eigvals = np.linalg.eigvalsh(rho)

        eigvals = np.clip(eigvals, 1e-12, None)

        eigvals = eigvals / eigvals.sum()

        purity = np.sum(eigvals ** 2)

        entropy = -np.sum(
            eigvals * np.log(eigvals)
        )

        effective_rank = np.exp(entropy)

        dominant_ratio = eigvals.max()

        k = min(5, len(eigvals))
        top5_energy = eigvals[::-1][:k].sum()

        metrics[int(c)] = {
            "purity": float(purity),
            "entropy": float(entropy),
            "effective_rank": float(effective_rank),
            "dominant_ratio": float(dominant_ratio),
            "top5_energy": float(top5_energy),
        }

    return metrics

def compute_operator_overlap_metrics(
    feats: np.ndarray,
    labels: np.ndarray,
) -> Dict[str, float]:
    eps = 1e-9

    feats = np.asarray(feats, dtype=np.float64)
    labels = np.asarray(labels)

    feats = feats / (np.linalg.norm(feats, axis=1, keepdims=True) + eps)

    rhos = []
    classes = []

    for c in np.unique(labels):
        Z = feats[labels == c]

        if len(Z) < 2:
            continue

        rho = (Z.T @ Z) / len(Z)

        rho = 0.5 * (rho + rho.T)

        rho = rho / (np.trace(rho) + eps)

        rhos.append(rho)
        classes.append(int(c))

    if len(rhos) < 2:
        return {
            "OP_OVERLAP_MEAN": 0.0,
            "OP_OVERLAP_MIN": 0.0,
            "OP_OVERLAP_MAX": 0.0,
        }

    overlaps = []

    for i in range(len(rhos)):
        for j in range(i + 1, len(rhos)):
            ov = np.trace(rhos[i] @ rhos[j])
            overlaps.append(float(ov))

    overlaps = np.asarray(overlaps, dtype=np.float64)

    return {
        "OP_OVERLAP_MEAN": float(overlaps.mean()),
        "OP_OVERLAP_MIN": float(overlaps.min()),
        "OP_OVERLAP_MAX": float(overlaps.max()),
    }

def build_class_density_operators(
    feats: np.ndarray,
    labels: np.ndarray,
    normalize: bool = True,
    trace_normalize: bool = True,
    eps: float = 1e-9,
) -> Tuple[np.ndarray, np.ndarray]:
    feats = np.asarray(feats, dtype=np.float64)
    labels = np.asarray(labels)

    if normalize:
        feats = feats / (np.linalg.norm(feats, axis=1, keepdims=True) + eps)

    classes = np.unique(labels)
    rhos = []

    for c in classes:
        Z = feats[labels == c]

        if len(Z) == 0:
            continue

        rho = (Z.T @ Z) / max(len(Z), 1)
        rho = 0.5 * (rho + rho.T)

        if trace_normalize:
            rho = rho / (np.trace(rho) + eps)

        rhos.append(rho)

    return classes.astype(int), np.stack(rhos, axis=0)

def compute_qit_class_state_metrics(
    feats: np.ndarray,
    labels: np.ndarray,
) -> Dict[str, float]:
    _, rhos = build_class_density_operators(
        feats, labels, normalize=True, trace_normalize=True
    )

    if len(rhos) < 2:
        return {
            "QIT_PURITY": 0.0,
            "QIT_OVERLAP": 0.0,
            "QIT_CLASS_HS2": 0.0,
            "QIT_OBJECTIVE": 0.0,
            "QIT_EQUIV_ERROR": 0.0,
        }

    purities = np.einsum("cij,cji->c", rhos, rhos)
    purity_mean = float(purities.mean())

    overlaps = []
    hs2_values = []
    for i in range(len(rhos)):
        for j in range(i + 1, len(rhos)):
            overlap_ij = float(np.trace(rhos[i] @ rhos[j]))
            overlaps.append(overlap_ij)
            hs2_values.append(
                float(purities[i] + purities[j] - 2.0 * overlap_ij)
            )

    overlap_mean = float(np.mean(overlaps))
    hs2_mean = float(np.mean(hs2_values))
    qit_objective = float((1.0 - purity_mean) + overlap_mean)
    qit_from_hs = float(1.0 - 0.5 * hs2_mean)

    return {
        "QIT_PURITY": purity_mean,
        "QIT_OVERLAP": overlap_mean,
        "QIT_CLASS_HS2": hs2_mean,
        "QIT_OBJECTIVE": qit_objective,
        "QIT_EQUIV_ERROR": abs(qit_objective - qit_from_hs),
    }

def operator_overlap_scores(
    feats: np.ndarray,
    rhos: np.ndarray,
    normalize: bool = True,
    eps: float = 1e-9,
) -> np.ndarray:
    feats = np.asarray(feats, dtype=np.float64)

    if normalize:
        feats = feats / (np.linalg.norm(feats, axis=1, keepdims=True) + eps)

    scores = np.einsum("nd,cde,ne->nc", feats, rhos, feats)
    return scores.astype(np.float64)

def operator_distance_scores(
    feats: np.ndarray,
    rhos: np.ndarray,
    normalize: bool = True,
    eps: float = 1e-9,
) -> np.ndarray:
    feats = np.asarray(feats, dtype=np.float64)

    if normalize:
        feats = feats / (np.linalg.norm(feats, axis=1, keepdims=True) + eps)

    overlap = np.einsum("nd,cde,ne->nc", feats, rhos, feats)
    rho_purity = np.einsum("cde,ced->c", rhos, rhos)

    distances = 1.0 + rho_purity[None, :] - 2.0 * overlap
    return distances.astype(np.float64)

def softmax_np(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = x - np.max(x, axis=axis, keepdims=True)
    ex = np.exp(x)
    return ex / (np.sum(ex, axis=axis, keepdims=True) + 1e-12)

def expected_calibration_error(
    confidences: np.ndarray,
    predictions: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 15,
) -> float:
    confidences = np.asarray(confidences, dtype=np.float64)
    predictions = np.asarray(predictions)
    labels = np.asarray(labels)

    correct = (predictions == labels).astype(np.float64)
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)

    ece = 0.0
    n = len(labels)

    for b in range(n_bins):
        lo, hi = bin_edges[b], bin_edges[b + 1]

        if b == n_bins - 1:
            mask = (confidences >= lo) & (confidences <= hi)
        else:
            mask = (confidences >= lo) & (confidences < hi)

        if not np.any(mask):
            continue

        bin_acc = correct[mask].mean()
        bin_conf = confidences[mask].mean()
        ece += (mask.mean()) * abs(bin_acc - bin_conf)

    return float(ece)

def tune_temperature_for_ece(
    logits_like: np.ndarray,
    predictions: np.ndarray,
    labels: np.ndarray,
    temperatures: Optional[List[float]] = None,
    n_bins: int = 15,
) -> Tuple[float, float]:
    if temperatures is None:
        temperatures = [
            0.05, 0.075, 0.10, 0.15, 0.20, 0.30, 0.50,
            0.75, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0
        ]

    best_tau = temperatures[0]
    best_ece = float("inf")

    pred_idx = np.argmax(logits_like, axis=1)

    for tau in temperatures:
        probs = softmax_np(logits_like / tau, axis=1)
        conf = probs[np.arange(len(probs)), pred_idx]

        ece = expected_calibration_error(
            conf,
            predictions,
            labels,
            n_bins=n_bins,
        )

        if ece < best_ece:
            best_ece = ece
            best_tau = tau

    return float(best_tau), float(best_ece)

def selective_classification_curve(
    confidences: np.ndarray,
    predictions: np.ndarray,
    labels: np.ndarray,
    coverages: Optional[List[float]] = None,
) -> Dict[str, float]:
    if coverages is None:
        coverages = [1.0, 0.95, 0.90, 0.80, 0.70, 0.50]

    confidences = np.asarray(confidences, dtype=np.float64)
    predictions = np.asarray(predictions)
    labels = np.asarray(labels)

    order = np.argsort(-confidences)
    n = len(labels)

    out = {}
    for cov in coverages:
        k = max(1, int(round(cov * n)))
        idx = order[:k]
        acc = float((predictions[idx] == labels[idx]).mean())
        out[f"Selective_Acc@{int(round(cov * 100))}%"] = 100.0 * acc

    return out

def operator_uncertainty_evaluation(
    reference_feats: np.ndarray,
    reference_labels: np.ndarray,
    eval_feats: np.ndarray,
    eval_labels: np.ndarray,
    beta: float = 20.0,
    n_bins: int = 15,
    tune_temperature: bool = True,
) -> Dict[str, float]:
    classes, rhos = build_class_density_operators(
        reference_feats,
        reference_labels
    )

    overlap_scores = operator_overlap_scores(eval_feats, rhos)

    overlap_pred_idx = np.argmax(overlap_scores, axis=1)
    overlap_predictions = classes[overlap_pred_idx]

    sorted_overlap = np.sort(overlap_scores, axis=1)
    overlap_top1 = sorted_overlap[:, -1]
    overlap_top2 = sorted_overlap[:, -2] if overlap_scores.shape[1] > 1 else np.zeros_like(overlap_top1)
    overlap_margins = overlap_top1 - overlap_top2

    overlap_probs_beta = softmax_np(beta * overlap_scores, axis=1)
    overlap_conf_beta = overlap_probs_beta[np.arange(len(overlap_probs_beta)), overlap_pred_idx]

    overlap_acc = float((overlap_predictions == eval_labels).mean())

    overlap_ece_beta = expected_calibration_error(
        overlap_conf_beta,
        overlap_predictions,
        eval_labels,
        n_bins=n_bins,
    )

    if tune_temperature:
        overlap_tau, overlap_ece_tau = tune_temperature_for_ece(
            logits_like=overlap_scores,
            predictions=overlap_predictions,
            labels=eval_labels,
            n_bins=n_bins,
        )
    else:
        overlap_tau, overlap_ece_tau = 1.0 / beta, overlap_ece_beta

    overlap_probs_tau = softmax_np(overlap_scores / overlap_tau, axis=1)
    overlap_conf_tau = overlap_probs_tau[np.arange(len(overlap_probs_tau)), overlap_pred_idx]

    distances = operator_distance_scores(eval_feats, rhos)

    dist_pred_idx = np.argmin(distances, axis=1)
    dist_predictions = classes[dist_pred_idx]

    sorted_dist = np.sort(distances, axis=1)
    dist_top1 = sorted_dist[:, 0]
    dist_top2 = sorted_dist[:, 1] if distances.shape[1] > 1 else np.zeros_like(dist_top1)
    dist_margin = dist_top2 - dist_top1

    distance_logits = -distances

    if tune_temperature:
        dist_tau, dist_ece_tau = tune_temperature_for_ece(
            logits_like=distance_logits,
            predictions=dist_predictions,
            labels=eval_labels,
            n_bins=n_bins,
        )
    else:
        dist_tau, dist_ece_tau = 1.0, float("nan")

    dist_probs_tau = softmax_np(distance_logits / dist_tau, axis=1)
    dist_conf_tau = dist_probs_tau[np.arange(len(dist_probs_tau)), dist_pred_idx]

    dist_acc = float((dist_predictions == eval_labels).mean())

    selective_overlap_prob = selective_classification_curve(
        overlap_conf_tau,
        overlap_predictions,
        eval_labels,
    )

    selective_overlap_margin = selective_classification_curve(
        overlap_margins,
        overlap_predictions,
        eval_labels,
    )
    selective_overlap_margin = {
        k.replace("Selective_", "Selective_OverlapMargin_"): v
        for k, v in selective_overlap_margin.items()
    }

    selective_dist_prob = selective_classification_curve(
        dist_conf_tau,
        dist_predictions,
        eval_labels,
    )
    selective_dist_prob = {
        k.replace("Selective_", "Selective_DistanceProb_"): v
        for k, v in selective_dist_prob.items()
    }

    selective_dist_margin = selective_classification_curve(
        dist_margin,
        dist_predictions,
        eval_labels,
    )
    selective_dist_margin = {
        k.replace("Selective_", "Selective_DistanceMargin_"): v
        for k, v in selective_dist_margin.items()
    }

    out = {
        
        "Operator_Overlap_Acc_%": 100.0 * overlap_acc,
        "Operator_Overlap_ECE_Beta": overlap_ece_beta,
        "Operator_Overlap_Beta": float(beta),
        "Operator_Overlap_Tau": float(overlap_tau),
        "Operator_Overlap_ECE_Tuned": float(overlap_ece_tau),
        "Operator_Overlap_Conf_Mean": float(overlap_conf_tau.mean()),
        "Operator_Overlap_Conf_Std": float(overlap_conf_tau.std(ddof=1)) if len(overlap_conf_tau) > 1 else 0.0,
        "Operator_Overlap_Top1Score_Mean": float(overlap_top1.mean()),
        "Operator_Overlap_Top2Score_Mean": float(overlap_top2.mean()),
        "Operator_Overlap_Margin_Mean": float(overlap_margins.mean()),
        "Operator_Overlap_Margin_Std": float(overlap_margins.std(ddof=1)) if len(overlap_margins) > 1 else 0.0,

        "Operator_Distance_Acc_%": 100.0 * dist_acc,
        "Operator_Distance_Tau": float(dist_tau),
        "Operator_Distance_ECE_Tuned": float(dist_ece_tau),
        "Operator_Distance_Conf_Mean": float(dist_conf_tau.mean()),
        "Operator_Distance_Conf_Std": float(dist_conf_tau.std(ddof=1)) if len(dist_conf_tau) > 1 else 0.0,
        "Operator_Distance_MinDist_Mean": float(dist_top1.mean()),
        "Operator_Distance_SecondDist_Mean": float(dist_top2.mean()),
        "Operator_Distance_Margin_Mean": float(dist_margin.mean()),
        "Operator_Distance_Margin_Std": float(dist_margin.std(ddof=1)) if len(dist_margin) > 1 else 0.0,
    }

    out["Operator_Acc_%"] = out["Operator_Overlap_Acc_%"]
    out["Operator_ECE"] = out["Operator_Overlap_ECE_Beta"]
    out["Operator_Margin_Mean"] = out["Operator_Overlap_Margin_Mean"]
    out["Operator_ProbConf_Mean"] = float(overlap_conf_beta.mean())

    out.update(selective_overlap_prob)
    out.update(selective_overlap_margin)
    out.update(selective_dist_prob)
    out.update(selective_dist_margin)

    return out

def save_operator_uncertainty_csv(metrics: Dict[str, float], out_path: str):
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(metrics.keys()))
        writer.writeheader()
        writer.writerow(metrics)

def centroid_evaluation_metrics(
    feats: np.ndarray,
    labels: np.ndarray,
    normalize: bool = True
) -> Dict[str, float]:
    eps = 1e-9

    feats = np.asarray(feats, dtype=np.float64)
    labels = np.asarray(labels)

    if normalize:
        feats = feats / (np.linalg.norm(feats, axis=1, keepdims=True) + eps)

    classes = np.unique(labels)

    centroids = []
    class_intra_mean = []
    class_intra_max = []

    for c in classes:
        Xc = feats[labels == c]
        muc = Xc.mean(axis=0)

        if normalize:
            muc = muc / (np.linalg.norm(muc) + eps)

        centroids.append(muc)

        dists_to_own = np.linalg.norm(Xc - muc[None, :], axis=1)
        class_intra_mean.append(float(dists_to_own.mean()))
        class_intra_max.append(float(dists_to_own.max()))

    centroids = np.asarray(centroids)

    inter_dists = []
    for i in range(len(centroids)):
        for j in range(i + 1, len(centroids)):
            inter_dists.append(float(np.linalg.norm(centroids[i] - centroids[j])))

    inter_dists = np.asarray(inter_dists, dtype=np.float64)

    intra_mean = float(np.mean(class_intra_mean)) if len(class_intra_mean) else 0.0
    intra_max_mean = float(np.mean(class_intra_max)) if len(class_intra_max) else 0.0

    inter_mean = float(inter_dists.mean()) if len(inter_dists) else 0.0
    inter_min = float(inter_dists.min()) if len(inter_dists) else 0.0
    inter_max = float(inter_dists.max()) if len(inter_dists) else 0.0

    dmat = np.linalg.norm(feats[:, None, :] - centroids[None, :, :], axis=2)
    pred_idx = np.argmin(dmat, axis=1)
    preds = classes[pred_idx]

    nca_acc = float((preds == labels).mean()) if len(labels) else 0.0
    overlap = 100.0 * (1.0 - nca_acc)

    margins = []
    for i, y in enumerate(labels):
        true_pos = int(np.where(classes == y)[0][0])
        d_true = dmat[i, true_pos]

        d_wrong = np.delete(dmat[i], true_pos)
        nearest_wrong = float(d_wrong.min())

        margins.append(nearest_wrong - d_true)

    margins = np.asarray(margins, dtype=np.float64)

    separation_ratio = inter_min / (intra_mean + eps)

    cent_norm = centroids / (np.linalg.norm(centroids, axis=1, keepdims=True) + eps)
    cos_cent = cent_norm @ cent_norm.T

    if len(classes) > 1:
        off_diag = cos_cent[~np.eye(len(classes), dtype=bool)]
        centroid_cosine_mean = float(off_diag.mean())
        centroid_cosine_max = float(off_diag.max())
    else:
        centroid_cosine_mean = 0.0
        centroid_cosine_max = 0.0

    return {
        "Centroid_NCA_Acc_%": 100.0 * nca_acc,
        "Centroid_Overlap_%": overlap,

        "Centroid_IntraDist_Mean": intra_mean,
        "Centroid_IntraDist_MaxMean": intra_max_mean,

        "Centroid_InterDist_Mean": inter_mean,
        "Centroid_InterDist_Min": inter_min,
        "Centroid_InterDist_Max": inter_max,

        "Centroid_SeparationRatio": float(separation_ratio),

        "Centroid_Margin_Mean": float(margins.mean()) if len(margins) else 0.0,
        "Centroid_Margin_Min": float(margins.min()) if len(margins) else 0.0,
        "Centroid_Margin_Std": float(margins.std(ddof=1)) if len(margins) > 1 else 0.0,

        "Centroid_Cosine_Mean": centroid_cosine_mean,
        "Centroid_Cosine_Max": centroid_cosine_max,
    }

def plot_distance_distribution(feats, labels, out_path):
    set_pretty_rc()

    feats = feats / np.linalg.norm(feats, axis=1, keepdims=True)
    cos = feats @ feats.T

    same = labels[:, None] == labels[None, :]
    diff = ~same

    eye = np.eye(len(labels), dtype=bool)
    same = same & (~eye)
    diff = diff & (~eye)

    intra = cos[same]
    inter = cos[diff]

    plt.figure(figsize=(6,4))
    sns.kdeplot(intra, label="Intra-class", fill=True)
    sns.kdeplot(inter, label="Inter-class", fill=True)

    plt.title("Embedding Separation")
    plt.xlabel("Cosine Similarity")
    plt.legend()

    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

def expand_limits(ax, coords, pad=0.2):
    mins = coords.min(axis=0)
    maxs = coords.max(axis=0)
    ranges = maxs - mins

    ax.set_xlim(mins[0] - pad*ranges[0], maxs[0] + pad*ranges[0])
    ax.set_ylim(mins[1] - pad*ranges[1], maxs[1] + pad*ranges[1])
    ax.set_zlim(mins[2] - pad*ranges[2], maxs[2] + pad*ranges[2])

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tr_loader, val_loader, class_names, num_classes, run_suffix = get_dataset_and_loaders(
        batch_size=BATCH_SIZE
    )

    RUN_NAME = f"qit_on_cosine_ls_{run_suffix}"

    RUN_DIR = os.path.join("runs", RUN_NAME)
    os.makedirs(RUN_DIR, exist_ok=True)

    model = Model(num_classes=num_classes)

    model, metrics_log, avg_epoch_time = train(model, tr_loader, val_loader, device)

    feats_raw, labs_raw = extract_features_raw(
        model,
        val_loader,
        device
    )

    density_metrics = compute_density_spectrum_metrics(
        feats_raw,
        labs_raw
    )

    operator_overlap_metrics = compute_operator_overlap_metrics(
        feats_raw,
        labs_raw
    )

    qit_class_state_metrics = compute_qit_class_state_metrics(
        feats_raw,
        labs_raw
    )

    print("\n===== Density Operator Spectrum Metrics =====")

    for cls, vals in density_metrics.items():

        print(
            f"Class {cls}: "
            f"H={vals['entropy']:.4f} "
            f"r={vals['dominant_ratio']:.4f}"
        )

    print("\n===== Operator Overlap Metrics =====")
    print(f"Mean Operator Overlap: {operator_overlap_metrics['OP_OVERLAP_MEAN']:.6f}")
    print(f"Min  Operator Overlap: {operator_overlap_metrics['OP_OVERLAP_MIN']:.6f}")
    print(f"Max  Operator Overlap: {operator_overlap_metrics['OP_OVERLAP_MAX']:.6f}")

    print("\n===== QIT Class-State Geometry =====")
    print(f"Mean class purity: {qit_class_state_metrics['QIT_PURITY']:.6f}")
    print(f"Mean class overlap: {qit_class_state_metrics['QIT_OVERLAP']:.6f}")
    print(f"Mean class D_HS^2: {qit_class_state_metrics['QIT_CLASS_HS2']:.6f}")
    print(f"QIT objective: {qit_class_state_metrics['QIT_OBJECTIVE']:.6f}")
    print(f"Theory equivalence error: {qit_class_state_metrics['QIT_EQUIV_ERROR']:.3e}")

    operator_uncertainty_metrics = operator_uncertainty_evaluation(
        reference_feats=feats_raw,
        reference_labels=labs_raw,
        eval_feats=feats_raw,
        eval_labels=labs_raw,
        beta=20.0,
        n_bins=15,
    )

    print("\n===== Operator-Based Uncertainty Metrics =====")
    print(f"Operator Acc: {operator_uncertainty_metrics['Operator_Acc_%']:.4f}%")
    print(f"Operator ECE: {operator_uncertainty_metrics['Operator_ECE']:.6f}")
    print(f"Operator Margin Mean: {operator_uncertainty_metrics['Operator_Margin_Mean']:.6f}")
    print(f"Selective Acc@90%: {operator_uncertainty_metrics['Selective_Acc@90%']:.4f}%")
    print(f"Selective Overlap-Margin Acc@90%: {operator_uncertainty_metrics['Selective_OverlapMargin_Acc@90%']:.4f}%")
    print(f"Selective Distance-Margin Acc@90%: {operator_uncertainty_metrics['Selective_DistanceMargin_Acc@90%']:.4f}%")

    save_operator_uncertainty_csv(
        operator_uncertainty_metrics,
        os.path.join(RUN_DIR, "operator_uncertainty_metrics.csv")
    )

    print(f"Average epoch time: {avg_epoch_time:.2f} seconds")

    torch.save(model.state_dict(), os.path.join(RUN_DIR, "model_final.pth"))
    csv_path = os.path.join(RUN_DIR, f"{RUN_NAME}_metrics.csv")
    save_metrics_csv(metrics_log, csv_path)

    plot_loss_acc(metrics_log, os.path.join(RUN_DIR, "loss_acc_curves.png"))

    model.eval()
    ys, preds = [], []
    with torch.no_grad():
        for x, y in val_loader:
            x = x.to(device)
            logits = model(x)
            p = logits.argmax(1).cpu().numpy()
            ys.extend(y.numpy())
            preds.extend(p)
    cm = confusion_matrix(ys, preds, labels=list(range(num_classes)))
    plot_confusion(cm, class_names, os.path.join(RUN_DIR, "val_confusion.png"))

    feats_raw, labels = extract_features_raw(model, val_loader, device)
    plot_pca_tsne(feats_raw, labels, class_names, os.path.join(RUN_DIR, "pca_tsne.png"))

    plot_pca_3d(
        feats_raw,
        labels,
        class_names, 
        os.path.join(RUN_DIR, "pca_3d.png")
    )
    
    plot_tsne_3d(
        feats_raw,
        labels,
        class_names, 
        os.path.join(RUN_DIR, "tsne_3d.png")
    )

    feats_norm = feats_raw / (np.linalg.norm(feats_raw, axis=1, keepdims=True) + 1e-9)

    plot_pca_3d(
        feats_norm,
        labels,
        class_names, 
        os.path.join(RUN_DIR, "pca_3d_normalized.png")
    )
    
    plot_tsne_3d(
        feats_norm,
        labels,
        class_names, 
        os.path.join(RUN_DIR, "tsne_3d_normalized.png")
    )

    if USE_FEW_SHOT_EVAL:
        feats_fs_vis = (
            feats_raw / (np.linalg.norm(feats_raw, axis=1, keepdims=True) + 1e-9)
            if FEW_SHOT_USE_NORMALIZED else feats_raw
        )

        for k in FEW_SHOT_KS:
            try:

                plot_few_shot_episode(
                    feats_fs_vis,
                    labels,
                    class_names, 
                    k_shot=k,
                    out_path=os.path.join(RUN_DIR, f"fewshot_{FEW_SHOT_N_WAY if FEW_SHOT_N_WAY is not None else 'all'}way_{k}shot_pca_tsne.png"),
                    n_way=FEW_SHOT_N_WAY,
                    seed=RANDOM_SEED
                )

                support_idx, query_idx = sample_few_shot_episode_indices(
                    labels,
                    k,
                    FEW_SHOT_QUERY_PER_CLASS,
                    n_way=FEW_SHOT_N_WAY,
                    seed=RANDOM_SEED
                )

                episode_idx = np.concatenate([support_idx, query_idx])
                episode_feats = feats_fs_vis[episode_idx]
                episode_labels = labels[episode_idx]

                support_mask = np.zeros(len(episode_idx), dtype=bool)
                support_mask[:len(support_idx)] = True

                plot_pca_tsne_3d(
                    episode_feats,
                    episode_labels,
                    class_names, 
                    os.path.join(RUN_DIR, f"fewshot_{FEW_SHOT_N_WAY if FEW_SHOT_N_WAY is not None else 'all'}way_{k}shot_3d.png"),
                    support_mask=support_mask,
                    title_prefix=f"{FEW_SHOT_N_WAY if FEW_SHOT_N_WAY is not None else 'all'}-way {k}-shot: "
                )

            except ValueError as e:
                print(f"Skipping {k}-shot visualization: {e}")

    feats_unit = feats_raw / (np.linalg.norm(feats_raw, axis=1, keepdims=True) + 1e-9)
    state_hs_separation_bar(feats_unit, labels, os.path.join(RUN_DIR, "state_hs_separation.txt"), os.path.join(RUN_DIR, "state_hs_separation.png"))

    report = final_eval_report(model, val_loader, device)

    last = metrics_log[-1]
    train_top1 = float(last["Train_Top1"])
    train_loss = float(last["Train_Loss"])
    print_final_report(report, train_top1, train_loss)

    few_shot_report = None
    if USE_FEW_SHOT_EVAL:
        few_shot_report = run_few_shot_evaluation(model, val_loader.dataset, device)
        print_few_shot_report(few_shot_report)
        with open(os.path.join(RUN_DIR, "few_shot_report.json"), "w") as f:
            json.dump(few_shot_report, f, indent=2)
        save_few_shot_csv(few_shot_report, os.path.join(RUN_DIR, "few_shot_report.csv"))

    final_payload = dict(report)

    final_payload.update({
        "OP_OVERLAP_MEAN": operator_overlap_metrics["OP_OVERLAP_MEAN"],
        "OP_OVERLAP_MIN": operator_overlap_metrics["OP_OVERLAP_MIN"],
        "OP_OVERLAP_MAX": operator_overlap_metrics["OP_OVERLAP_MAX"],
    })

    final_payload.update(qit_class_state_metrics)

    final_payload.update(operator_uncertainty_metrics)

    if few_shot_report is not None:
        for shot_name, stats in few_shot_report.items():
            key = shot_name.replace("-", "_")
            final_payload[f"FewShot_{key}_MeanAcc_%"] = 100.0 * float(stats["mean_acc"]) if int(stats["episodes"]) > 0 else float("nan")
            final_payload[f"FewShot_{key}_Std_%"] = 100.0 * float(stats["std_acc"]) if int(stats["episodes"]) > 0 else float("nan")
    with open(os.path.join(RUN_DIR, "final_report.json"), "w") as f:
        json.dump(final_payload, f, indent=2)
