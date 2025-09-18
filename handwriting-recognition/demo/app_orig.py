import os
import sys
import re
import importlib
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
import streamlit as st
from torchvision import transforms
from torchvision.datasets import ImageFolder

# ============================================================
# Config
# ============================================================
DATA_ROOT      = '../../IAM+RIMES'            # root dataset ImageFolder
SPLIT_PATH     = '../model/splits/IAM+RIMES.pth'       # usa lo STESSO split ovunque
CENTROIDS_PATH = 'author_centroids_names.npy'       # file centroidi (chiavi = nomi o id)
MODEL_PATH     = 'arcface_full_model.pth'
IMG_SIZE       = 128                          # come in training/val
TOPK           = 5                            # top-k vicini da mostrare
SHOW_GT_SIM    = True                         # mostra sim col centroide del GT per sanity

# ============================================================
# Setup
# ============================================================
st.set_page_config(page_title="Writer Identification – Open-set Demo", layout="wide")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Import del modello custom
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "model")))
import resnet_arcface
importlib.reload(resnet_arcface)
from resnet_arcface import ArcFaceNet  # se serve la classe

# ============================================================
# Utility
# ============================================================
def natkey(s: str):
    """Ordinamento naturale per stringhe numeriche con zeri iniziali."""
    m = re.match(r'^0*(\d+)$', s)
    return (0, int(m.group(1))) if m else (1, s)

def l2_normalize(v, eps=1e-8):
    v = np.asarray(v, dtype=np.float32)
    n = np.linalg.norm(v) + eps
    return v / n

@st.cache_resource
def load_model():
    model = torch.load(MODEL_PATH, map_location=device)
    if not isinstance(model, nn.Module):
        raise TypeError("MODEL_PATH non contiene un torch.nn.Module (hai salvato solo lo state_dict?).")
    model.eval().to(device)
    return model

@st.cache_resource
def load_split_and_dataset():
    dataset = ImageFolder(root=DATA_ROOT)               # senza transform
    split = torch.load(SPLIT_PATH, map_location='cpu')  # {'train_indices','test_indices','label_map'}
    return dataset, split

@st.cache_resource
def load_centroids(path):
    if not os.path.isfile(path):
        st.error(f"File centroidi non trovato: {path}")
        st.stop()
    d = np.load(path, allow_pickle=True).item()
    return d

def build_preprocess(model):
    """
    Preprocess allineato al training:
    - grayscale 128×128, Normalize((0.5,), (0.5,))
    - se conv1=3, duplica il canale in 3 ma con mean/std = 0.5 (NON ImageNet)
    """
    in_ch = 3
    try:
        in_ch = model.backbone.conv1.in_channels
    except Exception:
        pass

    if in_ch == 1:
        tf = transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ])
        mode = "L"
    else:
        tf = transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.Grayscale(num_output_channels=3),  # duplica il canale
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                 std =[0.5, 0.5, 0.5]),
        ])
        mode = "RGB"
    return tf, mode, in_ch

def _pick_matching_output(model_out, target_dim):
    """
    Se il modello ritorna una tupla (es. (logits, embeddings)), scegli il tensore con dim feature == target_dim.
    """
    if isinstance(model_out, tuple):
        candidates = [o for o in model_out if hasattr(o, "shape") and o.ndim == 2]
        for o in candidates:
            if o.shape[1] == target_dim:
                return o
        if candidates:
            return min(candidates, key=lambda o: abs(o.shape[1] - target_dim))
        for o in model_out:
            if hasattr(o, "shape"):
                return o
        return model_out[0]
    return model_out

def get_vector_for_matching(img_pil, preprocess_tf, color_mode, model, target_dim):
    img = img_pil.convert(color_mode)
    x = preprocess_tf(img).unsqueeze(0).to(device)  # [1,C,H,W]
    with torch.no_grad():
        out = model(x)
        vec = _pick_matching_output(out, target_dim)
        vec = vec.detach().cpu().numpy().reshape(-1)
    return l2_normalize(vec)

# ============================================================
# Load risorse
# ============================================================
model = load_model()
dataset, split = load_split_and_dataset()
idx_to_name = dataset.classes                  # original_label_idx -> nome cartella
label_map = split['label_map']                 # {original_label_idx -> new_id} solo autorizzati
authorized_ids = set(label_map.keys())
authorized_names = sorted([idx_to_name[i] for i in authorized_ids], key=natkey)

author_centroids_raw = load_centroids(CENTROIDS_PATH)

raw_keys = list(author_centroids_raw.keys())
st.sidebar.write(f"Centroidi nel file (grezzi): {len(author_centroids_raw)}")
st.sidebar.write(f"Tipo chiavi: {type(raw_keys[0]).__name__ if raw_keys else 'n/a'}")
st.sidebar.write(f"Prime chiavi: {raw_keys[:10]}")
st.sidebar.write(f"Autorizzati nello split (expected): {len(authorized_names)}")

# Porta le chiavi dei centroidi ai NOME classe e normalizza
first_key = next(iter(author_centroids_raw.keys()))
if isinstance(first_key, (int, np.integer)):
    author_centroids = {
        idx_to_name[k]: l2_normalize(v) for k, v in author_centroids_raw.items()
        if k in authorized_ids
    }
else:
    author_centroids = {
        k: l2_normalize(v) for k, v in author_centroids_raw.items()
        if k in authorized_names
    }

if not author_centroids:
    st.error("Nessun centroide dopo il filtro: controlla che centroidi e split corrispondano e che le chiavi siano nomi/ID coerenti.")
    st.stop()

CENTROID_DIM = len(next(iter(author_centroids.values())))
preprocess_tf, color_mode, in_ch = build_preprocess(model)

# ============================================================
# Sidebar diagnostica
# ============================================================
with st.sidebar.expander("🔎 Diagnostica", expanded=True):
    st.write(f"Device: **{device}**")
    st.write(f"Backbone in_channels: **{in_ch}** • Preprocess: **{color_mode} {IMG_SIZE}×{IMG_SIZE}**")
    st.write(f"Centroidi caricati: **{len(author_centroids)}** • Dim features centroidi: **{CENTROID_DIM}**")
    # Coerenza con test_set (se esiste)
    test_auth_dir = "test_set/autorizzati"
    if os.path.isdir(test_auth_dir):
        folders_auth = set(os.listdir(test_auth_dir))
        labels_app   = set(author_centroids.keys())
        miss_in_test = sorted(labels_app - folders_auth)[:10]
        miss_in_app  = sorted(folders_auth - labels_app)[:10]
        if miss_in_test:
            st.warning(f"Presenti in app/centroidi ma NON in test_set: {miss_in_test} …")
        if miss_in_app:
            st.warning(f"In test_set ma NON in app/centroidi: {miss_in_app} …")
        if not miss_in_test and not miss_in_app:
            st.success("Etichette coerenti con test_set/autorizzati ✅")

# ============================================================
# UI
# ============================================================
st.title("🖊️ Writer Identification – Open-Set Demo")
st.write("Carica un'immagine di manoscritto per verificare se l'autore è riconosciuto (genuino) o respinto (impostore).")

gt_choice = st.selectbox(
    "Etichetta attesa (ground truth)",
    options=["— Seleziona —", "Impostore/Sconosciuto"] + authorized_names,
    index=0
)

uploaded = st.file_uploader("Carica immagine", type=["png", "jpg", "jpeg"])

threshold = st.slider("Soglia (cosine similarity)", 0.0, 1.0, 0.65, 0.01)

# ============================================================
# Predizione
# ============================================================
def predict_author(vector_l2, threshold=0.65):
    sims = {a: float(np.dot(vector_l2, c)) for a, c in author_centroids.items()}  # cosine
    best_author, best_sim = max(sims.items(), key=lambda x: x[1])
    return (best_author, best_sim, sims) if best_sim >= threshold else ("Impostore", best_sim, sims)

if uploaded:
    img = Image.open(uploaded)
    st.image(img, caption="Input", use_column_width=True)

    vector = get_vector_for_matching(img, preprocess_tf, color_mode, model, CENTROID_DIM)
    st.caption(f"Dim vettore demo: {vector.shape[0]} • Dim centroidi: {CENTROID_DIM}")

    pred_author, best_sim, sims = predict_author(vector, threshold)

    st.subheader("Risultato modello")
    st.write(f"**Decisione**: {pred_author}")
    st.write(f"**Similarità massima (cosine)**: {best_sim:.4f}")
    st.caption(f"(threshold = {threshold:.2f})")

    # Top-k
    st.subheader(f"Top-{TOPK} autori più simili")
    sorted_sims = sorted(sims.items(), key=lambda x: x[1], reverse=True)
    lines = []
    for i, (a, s) in enumerate(sorted_sims[:TOPK], 1):
        mark = []
        if a == gt_choice and gt_choice != "Impostore/Sconosciuto":
            mark.append("GT")
        if a == pred_author:
            mark.append("Pred")
        tag = f" ← {', '.join(mark)}" if mark else ""
        lines.append(f"{i}. {a}: {s:.4f}{tag}")
    st.code("\n".join(lines))

    # Sanity: sim col centroide del GT
    if SHOW_GT_SIM and gt_choice not in ("— Seleziona —", "Impostore/Sconosciuto") and gt_choice in author_centroids:
        sim_gt = float(np.dot(vector, author_centroids[gt_choice]))
        st.caption(f"sim(input, centroide[{gt_choice}]) = {sim_gt:.4f}")

    # Valutazione rispetto al ground truth (senza metriche live)
    if gt_choice != "— Seleziona —":
        is_gt_impostor = (gt_choice == "Impostore/Sconosciuto")
        is_pred_impostor = (pred_author == "Impostore")

        if is_gt_impostor and is_pred_impostor:
            st.success("✅ CORRETTO: Impostore respinto.")
        elif (not is_gt_impostor) and (not is_pred_impostor) and (pred_author == gt_choice):
            st.success(f"✅ CORRETTO: Genuino accettato come **{gt_choice}**.")
        elif is_gt_impostor and (not is_pred_impostor):
            st.error(f"❌ ERRORE: Impostore accettato come **{pred_author}** (False Accept).")
        elif (not is_gt_impostor) and is_pred_impostor:
            st.error(f"❌ ERRORE: Genuino **{gt_choice}** respinto come impostore (False Reject).")
        else:
            st.error(f"❌ ERRORE: Genuino **{gt_choice}** identificato come **{pred_author}**.")
