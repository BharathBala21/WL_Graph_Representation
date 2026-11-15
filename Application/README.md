# WL-LeafKernel web app

Research portal for the WL graph-kernel plant-stress project, with the **PlantCare AI** tools
(dashboard, leaf analysis, diagnostics, stress library, crops, prediction log, ML pipeline, protocol)
built in. React + Vite + Tailwind.

## Run locally

Two terminals. The backend diagnoses uploaded photos and stores accounts and history; see
[backend/README.md](backend/README.md).

```bash
# 1. backend (first time: pip install -r requirements.txt && python train.py)
cd Application/backend
uvicorn app.main:app --port 8000

# 2. frontend
cd Application
npm install
npm run dev        # http://localhost:5173  (proxies /api and /files to :8000)
npm run build      # production build in dist/
```

Open http://localhost:5173, create an account, then go to **Analyze Leaf → Upload image**.
Diagnoses are saved under **My Diagnoses**. The research pages (dashboard, diagnostics, history)
work without the backend.

## Deploy (Vercel)

Set the Vercel project's **Root Directory** to `Application`. `vercel.json` sets the install/build
commands, the `dist` output folder, and the rewrite that lets routes such as `/plantcare/dashboard`
load directly.

## Layout

| Path | What |
|---|---|
| `src/pages/Home.jsx`, `src/lib/content.js` | Research portal page and its text content |
| `src/components/` | Shared layout (sidebar, top bar), portal widgets |
| `src/plantcare/` | PlantCare AI pages (`components/`), shared helpers (`lib.tsx`), generated data (`data/research.ts`) |
| `public/team/` | Team photos |
| `public/plantcare/` | Sample leaf thumbnails and the full out-of-fold prediction log |
| `scripts/build_plantcare_data.py` | Regenerates `src/plantcare/data/research.ts` and `public/plantcare/` |

## Updating the PlantCare AI data

The PlantCare pages read the experiment outputs under `full_pipeline/` (git-ignored) and the raw
OLID-I images in `../archive (3)`. After re-running experiments:

```bash
cd Application
python scripts/build_plantcare_data.py
```

Commit the regenerated `src/plantcare/data/research.ts` and `public/plantcare/` — Vercel only
builds the web app and does not run the Python pipeline.
