# PlantCare AI — backend

FastAPI service that diagnoses leaf photos with the project's WL graph-kernel models and keeps
per-user history in SQLite.

## What happens to an uploaded photo

1. **Leaf found and measured.** The leaf is separated with ExG + Otsu, and the 38 measurements are
   taken with the exact code from `full_pipeline/08_OLID_Full_Feature_Extraction.ipynb`.
   Photos where no leaf is found are rejected with a message.
2. **Crop identified.** A Random Forest routes the leaf to one of 8 crops (as in NB23 Stage 1). The
   user can also choose the crop.
3. **Healthy vs stressed.** The crop's WL kernel SVM decides, using that crop's best graph type and
   WL depth from the Level 2 experiments (NB22).
4. **Top 3 likely conditions.** A second WL kernel SVM ranks the specific conditions.
5. **Explanation.** The 5 measurements that differ most from healthy leaves of the same crop are shown.
6. **Spot check** (`app/lesions.py`). The 38 measurements are leaf averages, so a few small spots
   hardly change them. A separate rule looks for spots whose colour has turned from green towards
   brown/purple (Lab a*) relative to the surrounding leaf, and warns when there are at least 3
   covering 0.2% of the leaf. It does not change the model output. On 40 OLID-I leaves per class
   it warned on 1 of 320 healthy leaves; it is conservative and misses many small dark spots.
7. **Reliability check.** The photo is compared with what new leaves of the same crop normally look
   like to the model (5-fold CV limits in `metadata.json` → `familiarity`): crop confidence,
   measurements outside the training range, leaf size in pixels, and similarity to the closest
   training leaf. One failed check → "use with care", two or more → "unreliable" (the app then shows
   "Unsure" with the model's best guess). The result also lists the conditions the crop's model
   knows, so users can see when a problem (e.g. tomato leaf spot) is outside what it was trained on.
   Refresh the limits without retraining: `python train.py --calibrate-only`.

Every prediction is shown with the model's cross-validated reliability (`models/metadata.json`).

Each result also has a `trace` that records what every step produced for that photo: the Otsu
threshold and leaf share, all 38 values against the healthy average, the votes of each tree in the
crop forest, the feature graph with its WL labels and how common each label is in training, and the
SVM's pull from each support vector. Three step pictures (`<id>_exg.jpg`, `<id>_mask.png`,
`<id>_regions.jpg`) are saved next to the photo. The **Analyze Leaf** page replays these steps.

## Setup

```bash
cd Application/backend
pip install -r requirements.txt
python train.py                         # ~20 s, writes models/ from full_pipeline/olid_full_features.csv
uvicorn app.main:app --port 8000        # API on http://localhost:8000, docs on /docs
```

Then start the frontend (`cd Application && npm run dev`) and open http://localhost:5173.
Vite forwards `/api` and `/files` to port 8000.

## Reliability of the trained models (5-fold CV, `python train.py`)

| Crop | Graph (WL depth) | Stress Macro-F1 | Condition top-1 | Condition top-3 |
|---|---|---|---|---|
| Ash gourd | Feature-Family (2) | 0.607 | 0.594 | 0.895 |
| Bitter gourd | kNN (3) | 0.670 | 0.476 | 0.834 |
| Bottle gourd | Hybrid (3) | 0.652 | 0.530 | 0.855 |
| Cucumber | Feature-Family (2) | 0.834 | 0.571 | 0.979 |
| Eggplant | Hybrid (3) | 0.711 | 0.494 | 0.843 |
| Ridge gourd | Hybrid (1) | 0.518 | 0.554 | 0.911 |
| Snake gourd | Hybrid (3) | 0.744 | 0.637 | 0.943 |
| Tomato | Correlation (2) | 0.762 | 0.686 | 0.959 |

Crop router accuracy: 0.914. These numbers are for OLID-I photos (one leaf on a dark background).

## API

| Method | Path | Auth | Purpose |
|---|---|---|---|
| GET | `/api/health` | – | Server and model status |
| GET | `/api/model-info` | – | Model configuration and reliability |
| POST | `/api/auth/register` | – | `{name, email, password}` → `{token, user}` |
| POST | `/api/auth/login` | – | `{email, password}` → `{token, user}` |
| GET | `/api/auth/me` | ✓ | Current user |
| POST | `/api/predict?crop=` | ✓ | Raw image body (`Content-Type: image/*`, optional `X-Filename`) → saved diagnosis |
| GET | `/api/predictions` | ✓ | The user's diagnoses, newest first |
| GET | `/api/predictions/stats` | ✓ | Totals and per-crop counts |
| GET / PATCH / DELETE | `/api/predictions/{id}` | ✓ | Read, add a note, or delete |
| GET | `/files/{name}` | – | Stored photo / leaf outline (random file names) |

Auth is a Bearer JWT (7 days). Passwords are hashed with PBKDF2-SHA256 (240,000 iterations).

## Storage

`data/` (git-ignored) holds `plantcare.db` (SQLite: `users`, `predictions`), the uploaded photos in
`uploads/`, and `secret.key` (JWT signing key, created on first run; set `PLANTCARE_SECRET` to
override). Delete `data/` to reset all accounts and history.

## Tests

```bash
# from the repository root
python -m pytest Application/backend/tests -q
```

* `test_wl_equivalence.py`: the backend WL engine gives the same kernels as the research implementation.
* `test_api.py`: register/login, diagnosis of real OLID-I photos, history, notes, delete, and
  per-user isolation, run against a temporary database.
