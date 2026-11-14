# GreenML Studio

Full-stack empirical machine learning platform evaluating the accuracy and energy trade-offs in machine learning classifiers, implementing the IEEE GI research methodology unified with cross-paradigm classifier benchmarks. Built with an Apple Human Interface Guidelines (HIG) frontend, FastAPI REST backend, and dual-layer CPU energy telemetry.

---

## Architecture Overview
1. **Apple HIG Frontend (`frontend/`)**: Modern macOS / iOS inspired responsive interface (`index.html`, `styles.css`, `app.js` + `Plotly.js`) featuring frosted glass navigation, segmented pill controls, rounded squircles (18px radius), Apple color palette, interactive Pareto explorers, empirical attainment curves, dynamic carbon payback simulators, and live inference hot-swapping.
2. **FastAPI REST Backend (`backend/api.py`)**: High-performance asynchronous API server serving frontend assets, typed data endpoints (`/api/overview`, `/api/exp1/{dataset}`, `/api/exp2/{dataset}`, `/api/exp3/{dataset}`, `/api/cross-model/{dataset}`, `/api/hotswap/predict`, `/api/live-experiment`, `/api/figures`), and real-time classifier profiling.
3. **Data & ML Energy Engine (`src/` + `data/` + `outputs/`)**:
   - `src/config.py`: Hyperparameter grids, NSGA-II optimization bounds, dataset definitions.
   - `src/data_loader.py`: Authentic loaders for UCI and benchmark datasets with strict 75:25 splits and leakage-free normalization.
   - `src/energy_profiler.py`: Dual-layer `CodeCarbon` + nanosecond `process_time_ns` CPU energy & CO₂ telemetry.
   - `src/pareto.py`: 2D Pareto front extraction, knee-point detection, hypervolume, and empirical attainment surfaces.
   - `src/paper_experiments.py`: Rigorous empirical reproduction of Grid Search, Pareto analysis, and NSGA-II multi-objective optimization.
   - `src/extended_benchmark.py`: Cross-model classifier benchmarking, carbon payback calculator, and runtime Pareto model router.
   - `src/visualizations.py`: Publication-ready static figures generator.

---

## How to Run

### Step 1: Clone or Navigate to Project Directory
```bash
cd "c:\Users\muham\Downloads\ds Project"
```

### Step 2: Install Dependencies
Create and activate a virtual environment (optional but recommended), then install requirements:
```bash
pip install -r requirements.txt
```

### Step 3: Run Tests (Verification)
Verify backend logic, Pareto extraction, and API routes:
```bash
python -m pytest -v tests/
```

### Step 4: Start the Web Application
Launch the FastAPI server (which automatically serves the frontend at `/`):
```bash
python -m uvicorn backend.api:app --host 127.0.0.1 --port 8000
```
Or with auto-reload for development:
```bash
python -m uvicorn backend.api:app --host 127.0.0.1 --port 8000 --reload
```

### Step 5: Open in Your Browser
Open your web browser and navigate to:
```
http://127.0.0.1:8000
```

---

## (Optional) Re-running Offline Pipeline & Training
All empirical results and pre-trained Pareto bundles are already precomputed in `outputs/`. If you want to re-execute the entire pipeline from scratch:
```bash
python run_pipeline.py --mode standard --resume
```

---

## Deployment Options

### Option 1: Render (Single Service - Recommended)
1. Push code to GitHub.
2. In Render, select **New Web Service** and connect your repository.
3. Use the following settings:
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `uvicorn backend.api:app --host 0.0.0.0 --port $PORT`
   *(Render will also automatically detect the provided `render.yaml` blueprint).*

### Option 2: Vercel (Frontend) + Render (Backend)
- Frontend is configured for static hosting via `vercel.json`.
- Set `window.GREENML_API_BASE` in `frontend/app.js` to point to your live Render backend URL. CORS is already enabled on all backend routes.
