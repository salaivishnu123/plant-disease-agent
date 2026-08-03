# 🌿 Plant Disease Detection Agent — Areca Nut / Coconut / Mango

A lightweight, multi-agent plant disease detection system designed to run and
train on a **low-spec laptop**. Scope is deliberately narrowed to three crops
(Areca Nut, Coconut, Mango) so a single small transfer-learning model can reach
good accuracy without a GPU or huge dataset.

---

## 🚀 Live Demo

**[🌐 Try the live app on Streamlit Community Cloud](https://plantdiseasedetector-app-mcqyx2cexyhr2yosfgjsw4.streamlit.app/)**

Upload a plant leaf image and get instant disease detection, treatment recommendations, and AI model explainability (Grad-CAM visualization).

---

## Deployment

**Platform:** Streamlit Community Cloud  
**Python Runtime:** 3.11  
**Key Dependencies:** TensorFlow 2.15+, Streamlit 1.28+, MobileNetV2 backbone

### Run locally:
```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```
Then open `http://localhost:8501` in your browser.

---

## 1. Why the scope was narrowed

Training one model across dozens of species/diseases (like the full
PlantVillage set) needs a lot of images per class and a bigger backbone to
separate that many categories — expensive to train on a laptop CPU. Restricting
to 3 crops means:

- Fewer total classes → the softmax has an easier job → higher accuracy for the
  same model size and epoch budget.
- A MobileNet-family backbone (2–6M params) is enough.
- Full training runs finish in a reasonable time on CPU (or a laptop GPU if you
  have one).

## 2. Datasets (per crop)

There is **no single combined dataset** for these three crops together, so we
combine three separate public datasets into one folder layout. Class counts
below are what's published as of writing — always re-check the dataset page,
since versions get updated.

| Crop | Dataset | Classes |
|---|---|---|
| Mango | ["MangoLeafBD" — Mango Leaf Disease Dataset (Kaggle, aryashah2k)](https://www.kaggle.com/datasets/aryashah2k/mango-leaf-disease-dataset) | Anthracnose, Bacterial Canker, Cutting Weevil, Die Back, Gall Midge, Powdery Mildew, Sooty Mould, Healthy (8 classes) |
| Coconut | ["Coconut (Cocos nucifera) Tree Disease Dataset"](https://www.kaggle.com/datasets/devavratpatil/coconut-tree-disease-dataset) (also on Mendeley Data / ScienceDirect) | Bud Root Dropping, Bud Rot, Gray Leaf Spot, Leaf Rot, Stem Bleeding (5 classes — **no "Healthy" class published**, see note below) |
| Areca Nut | Scattered across a few Kaggle uploads (search "areca nut disease dataset kaggle" / "arecanut disease kaggle") — no single canonical source. A published GNN study used a curated 1,000-image subset of a ~8,847-image Kaggle set spanning 9 disease categories. | Varies by source — you'll need to pick one dataset and verify its class list before training |

**Important gaps to handle yourself before training:**
1. The Coconut dataset as published has no "Healthy" class — you should add one
   (photograph/collect ~100–150 healthy coconut leaf images, or find a
   healthy-leaf subset elsewhere) so the model can say "no disease detected."
2. Areca Nut doesn't have one authoritative dataset — download 1–2 candidates,
   inspect class balance yourself, and prune duplicates/mislabeled images
   before training. Budget extra time for this crop specifically.
3. License/attribution: check each dataset's license on Kaggle/Mendeley before
   using it in a report or redistributing it.

### Target folder layout after download

```
dataset/
├── train/
│   ├── Mango___Anthracnose/
│   ├── Mango___Healthy/
│   ├── ...
│   ├── Coconut___Bud_Rot/
│   ├── Coconut___Healthy/
│   ├── ...
│   ├── Areca_Nut___<Disease>/
│   └── Areca_Nut___Healthy/
├── val/        (same class subfolders, ~15% of data)
└── test/       (same class subfolders, ~15% of data)
```

Each class folder name follows `Plant___Disease` (matches the PlantVillage
convention) — this is what lets one model serve both the "which plant"
and "which disease" jobs at once (see architecture note below).

## 3. Model

- **Backbone:** MobileNetV2 (ImageNet weights, frozen for the first training
  phase, then partially unfrozen for fine-tuning). MobileNetV3-Small is a
  drop-in alternative if you want it even smaller.
- **Head:** GlobalAveragePooling2D → Dropout(0.3) → Dense(128, relu) →
  BatchNormalization → Dropout(0.3) → Dense(num_classes, softmax)
- **Input size:** 224×224 (160×160 if your laptop really struggles)
- **One combined classifier, not three separate ones.** Classes are labeled
  `Plant___Disease`. This is a deliberate simplification for low-spec
  hardware: one training run instead of managing 3+ models. The **agents**
  still cleanly separate "which plant" from "which disease" logically (see
  below) — they just share one set of learned features underneath.

### Training features included in `train.py`
- Stratified train/val/test split awareness (expects pre-split folders)
- `ImageDataGenerator`/`tf.data` augmentation: rotation, zoom, shear, flips,
  brightness jitter
- `class_weight` computed automatically for class imbalance
- Callbacks: `EarlyStopping`, `ReduceLROnPlateau`, `ModelCheckpoint`
- Evaluation: confusion matrix, precision/recall/F1 per class, top-k accuracy

## 4. Multi-Agent Architecture

```
User Upload
   │
   ▼
Image Processing Agent      (validate, resize, normalize → tensor)
   │
   ▼
Plant Identification Agent  (groups model output by plant prefix,
   │                          reports plant + confidence, flags uncertainty)
   ▼
Disease Diagnosis Agent     (restricts to the identified plant's classes,
   │                          renormalizes, returns disease + confidence +
   │                          symptoms + causes)
   ▼
Treatment Recommendation Agent (looks up organic/chemical treatment,
   │                             fertilizer, watering, prevention)
   ▼
Explainability Agent        (Grad-CAM heatmap overlay + plain-language
   │                          rationale of what the model focused on)
   ▼
Response Agent               (assembles everything into one JSON payload)
   │
   ▼
Flask UI dashboard
```

Each agent is a small, independent Python class with one job and one
public method — easy to explain in a review or interview, and easy to
swap out later (e.g. replace the single-model approach with real
per-plant models without touching the other agents).

## 5. Folder structure

```
plant-disease-agent/
├── app.py                     # Flask server, wires agents together, serves UI
├── train.py                   # Trains the MobileNetV2 classifier
├── predict.py                 # CLI + shared prediction pipeline (used by app.py)
├── agents/
│   ├── image_agent.py          # Image Processing Agent
│   ├── plant_agent.py          # Plant Identification Agent
│   ├── disease_agent.py        # Disease Diagnosis Agent
│   ├── treatment_agent.py      # Treatment Recommendation Agent
│   ├── explainability_agent.py # Grad-CAM Agent
│   └── response_agent.py       # Final response assembler
├── utils/
│   └── gradcam_utils.py        # Low-level Grad-CAM math (used by explainability_agent)
├── data/
│   └── treatment_db.json       # Symptoms / causes / treatment knowledge base
├── models/                     # class_indices.json + training history/plots land here
├── weights/                    # saved .keras model weights land here
├── dataset/                    # train/ val/ test/ image folders (you populate this)
├── templates/
│   └── index.html              # Dashboard UI
├── static/
│   ├── style.css
│   └── script.js
└── requirements.txt
```

## 6. Setup

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt

# 1. Populate dataset/train, dataset/val, dataset/test as described above
# 2. Train:
python train.py

# 3. Run the app:
python app.py
# open templates/index.html is served automatically at http://127.0.0.1:5000
```

## 7. Status / next steps

This scaffold gives you working, runnable code for every agent plus training
and serving scripts. Before it's fully accurate for your three crops you still
need to:
1. Download and organize the datasets above into `dataset/train|val|test`.
2. Run `python train.py` to produce `weights/best_model.keras` and
   `models/class_indices.json`.
3. Expand `data/treatment_db.json` with real entries for every class your
   model ends up with (a handful of starter entries are included so you can
   see the expected format).
