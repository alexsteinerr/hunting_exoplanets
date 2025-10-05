#  HUNTING EXOPLANETS

**NASA Space Apps Hackathon 2025 Project**

> **Goal:** Build a data-driven pipeline to detect and characterize exoplanet transits using light curve data, combining astrophysical modeling and machine learning.

---

## Overview

**HUNTING-EXOPLANETS** is a modular system for analyzing light curves from missions such as **Kepler** and **TESS**.
It performs:

* light-curve preprocessing and normalization,
* theoretical transit modeling and polynomial derivation,
* machine-learning–based classification of potential exoplanet candidates, and
* visualization of detected transit events via a web dashboard.

All processing steps are implemented in clean, reusable modules that reflect astrophysical principles (e.g., boundary detection, limb-darkening transit shape, flux variation models).

---

## Folder Architecture

```
HUNTING-EXOPLANETS/
│
├── analysis/               # Core astrophysical + mathematical models
│   ├── boundaries.py       # Detects transit boundaries and flux dips
│   ├── derivation.py       # Symbolic / analytic derivations of flux models
│   ├── polynomial.py       # Polynomial and curve-fitting utilities
│   └── transit.py          # Transit shape modeling and area integration
│
├── config/
│   └── settings.py         # Global configuration: paths, constants, API keys
│
├── data/
│   ├── loader.py           # Loads and caches light curve data
│   ├── processor.py        # Cleans, normalizes, and detrends raw flux data
│   └── lightcurve_cache/   # Local cached datasets (auto-generated)
│
├── models/
│   ├── mlp.py              # MLP (Neural Network) model for classification
│   └── trainer.py          # Training loop, evaluation, and model saving
│
├── targets/
│   └── target_list.py      # Defines NASA/TESS target catalogues and IDs
│
├── utils/
│   ├── api_client.py       # Fetches light curves from NASA/MAST APIs
│   ├── features.py         # Feature extraction (flux ratios, depths, slopes)
│   └── plotting.py         # Matplotlib plots for light curves & detections
│
├── templates/
│   └── index.html          # Web dashboard (Flask front end)
│
├── static/                 # CSS, JS, and images for the web UI
│
├── app.py                  # Flask web application (interactive dashboard)
├── train.py                # CLI entry point for model training
├── README.md               # You are here
└── hunting_exoplanents.pdf # Project report / presentation for NASA Space Apps
```

---

## Key Features

* **Automated light-curve pipeline** — from NASA archive ingestion → detrending → transit detection.
* **Machine Learning detection** — `models/mlp.py` provides a baseline neural classifier.
* **Astrophysical modeling** — `analysis/transit.py` and `derivation.py` implement theoretical transit equations for flux overlap.
* **Visualization** — Flask dashboard (`app.py`) + `templates/index.html` shows interactive light-curve plots.
* **Offline caching** — local cache in `data/lightcurve_cache/` for fast re-runs.

---

## Installation

```bash
git clone https://github.com/alexsteinerr/hunting_exoplanets.git
cd hunting_exoplanets
pip install -r requirements.txt
```

Requirements (example):

```
flask
numpy
pandas
scipy
matplotlib
torch         
astropy
lightkurve
```

---

## Usage

### Web Visualization Interface

Run the Flask dashboard:

```bash
python app.py
```

Then open [http://localhost:5000](http://localhost:5000)
Upload a `.csv` or `.fits` light curve, visualize the flux, and detect possible transits.

---

### Training the Neural Model

Train your model on labeled data (planet vs non-planet):

```bash
python train.py
```

* Loads configuration from `config/settings.py`
* Uses data pipeline (`data/loader.py`, `data/processor.py`)
* Saves trained weights to `models/`
* Prints evaluation metrics

---

### Data Processing Pipeline Example

```python
from data.loader import load_lightcurve
from data.processor import preprocess
from analysis.transit import analyze_transit_dip
from models.mlp import MLP
from utils.plotting import plot_lightcurve

lc = load_lightcurve("data/lightcurve_cache/example.csv")
clean = preprocess(lc)
dip_info = analyze_transit_dip(clean)
plot_lightcurve(clean, dips=dip_info)
```

---

## Data Sources

This project supports **NASA Exoplanet Archive** and **MAST (TESS/Kepler)** datasets.

* `utils/api_client.py` manages API calls and caching.
* Light curves can also be loaded manually into `data/lightcurve_cache/`.

**Example dataset:** TESS mission (flux vs. time) in `.csv` or `.fits` format.

---

## Configuration

All settings (e.g., cache paths, model hyperparameters, debug flags) are defined in `config/settings.py`.

Example:

```python
DATA_DIR = "data/lightcurve_cache/"
MODEL_DIR = "models/"
LEARNING_RATE = 0.001
EPOCHS = 20
THRESHOLD = 0.05
```

---

## For NASA Hackathon Judges

| Stage              | Description                                  | Script                |
| ------------------ | -------------------------------------------- | --------------------- |
| **Data Ingestion** | Download or load existing TESS/Kepler curves | `data/loader.py`      |
| **Preprocessing**  | Clean, normalize, remove systematics         | `data/processor.py`   |
| **Analysis**       | Fit transit model, derive dip metrics        | `analysis/transit.py` |
| **Training**       | MLP classifier training                      | `train.py`            |
| **Visualization**  | Flask dashboard visualization                | `app.py`              |

Typical run-through (≈5 min demo):

1. Launch the dashboard: `python app.py`
2. Upload a sample curve from `data/lightcurve_cache/`
3. Click **Detect** → displays detected flux dips with timestamps
4. (Optional) Run `python train.py` to show the ML detection process

---

## License

Released under the **MIT License**.

---

## Citation

If you use this project or parts of it for research or hackathon extensions, please cite as:

```
Steiner, A. (2025). HUNTING_EXOPLANETS: A Modular Pipeline for Exoplanet Detection.
NASA Space Apps Challenge Repository. GitHub.
```

---
