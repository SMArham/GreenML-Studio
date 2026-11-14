# 🌿 GreenML Studio

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![CodeCarbon](https://img.shields.io/badge/CodeCarbon-Hardware%20Telemetry-2ECC71?logo=leaf&logoColor=white)](https://codecarbon.io/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.5+-F7931E?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Tests](https://img.shields.io/badge/Tests-16%20Passed-brightgreen?logo=pytest&logoColor=white)](tests/)
[![Render](https://img.shields.io/badge/Deploy-Render-46E3B7?logo=render&logoColor=white)](https://render.com)

> **Empirical Machine Learning Accuracy & Hardware Energy Profiling Engine**  
> Finding the Pareto-optimal balance between high accuracy, low electricity consumption (Joules), and sustainable carbon emissions ($\text{CO}_2$). Calibrated with `CodeCarbon` and hardware CPU telemetry.

---

## 🔴 1. The Problem (Why We Built This)

In modern Machine Learning and Data Science, practitioners and researchers almost exclusively optimize for **a single metric: Accuracy**. Almost nobody measures **how much hardware energy or electricity the model burns**.

* **Massive Energy Waste:** A deep Multi-Layer Perceptron (MLP) or massive Random Forest often consumes **80× more electricity** and **100× more time** than a lightweight model, just to gain a tiny **1% improvement in accuracy**.
* **Battery Drain on Edge Devices:** Deploying unoptimized models onto smartphones, wearable devices, and IoT sensors causes batteries to overheat and drain in hours.
* **Costly Cloud Bills & Carbon Footprint:** Running millions of inferences daily in cloud data centers consumes megawatts of power, generating substantial carbon dioxide ($\text{CO}_2$) emissions and high infrastructure bills.
* **The Blind Spot:** Until now, developers lacked an interactive, empirical workbench to measure CPU Joules and carbon emissions alongside accuracy in real time.

---

## 🟢 2. The Solution (What GreenML Studio Does)

**GreenML Studio** provides an end-to-end empirical platform that profiles both **Accuracy AND Energy (Joules)** using real hardware telemetry:

1. **Multi-Objective Pareto Optimization (NSGA-II):**
   * Automatically plots the 2D trade-off surface (**Test Accuracy vs. Energy**).
   * Identifies the **"Knee Point"** — the sweet-spot model that delivers up to **96.3% energy savings** with less than a 3% drop in accuracy.
2. **Best Models Curation:**
   * Directly filters out dangerous "energy traps" (such as SVM models consuming 3,333 Joules) and highlights lightweight, highly accurate candidates (under 0.5s training time and minimal inference energy).
3. **Adaptive Pareto Model Hot-Swapping:**
   * Dynamically switches models at runtime based on the device's live battery status and user-defined accuracy threshold.
   * **Low Battery?** Shifts to an **Ultra-Green model** (e.g., Logistic Regression).
   * **Plugged in / High Accuracy needed?** Shifts to a **Max-Accuracy model** (e.g., Decision Tree).
4. **Carbon & Financial Payback Simulator:**
   * Visualizes exactly how many days of production inference it takes for an energy-efficient model to pay back its initial training carbon and energy costs.

---

## 🖥️ 3. Dashboard Tour (Tabs Overview)

GreenML Studio features an Apple Human Interface Guidelines (HIG) responsive web interface:

| Tab | Feature | Description |
| :--- | :--- | :--- |
| **1. Project Overview** | Master Research Synthesis | High-resolution publication plates, hardware specs, and methodology overview. |
| **2. Grid & Pareto** | Multi-Objective Trade-Off | 2D Pareto frontiers comparing Training vs. Inference stages across UCI datasets. |
| **3. Drivers & Stages** | Statistical Drivers | Spearman rank correlation and Kruskal-Wallis tests identifying which hyperparameters drive energy. |
| **4. NSGA-II & Payback** | Multi-Objective Optimization | NSGA-II empirical attainment curves and real-time 60fps carbon payback slider. |
| **5. Cross-Model Benchmark** | Multi-Family Comparison | Sortable **Best Models Comparison Table** spanning Logistic Regression, Decision Trees, Random Forests, MLPs, SVMs, and KNN. |
| **6. Live Simulator & Hot-Swap** | Dynamic Router & Live Lab | Battery-aware dynamic inference routing and real-time on-demand model training with CPU telemetry. |

---

## 🚀 4. How to Use It (Quickstart)

### Option A: Run Locally (3 Simple Steps)

#### 1. Clone the Repository
```bash
git clone https://github.com/SMArham/GreenML-Studio.git
cd GreenML-Studio
```

#### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

#### 3. Start the Server
```bash
python -m uvicorn backend.api:app --host 127.0.0.1 --port 8000
```
Open your web browser and go to:
👉 **`http://127.0.0.1:8000`**

---

### Option B: Deploy to Render (All-in-One Cloud Host)

The repository includes a ready-to-deploy `render.yaml` and `Procfile`.

1. Go to [Render.com](https://dashboard.render.com) and log in.
2. Click **`New +`** → **`Web Service`**.
3. Connect your **`GreenML-Studio`** GitHub repository.
4. Verify the build and start commands:
   * **Runtime:** `Python 3`
   * **Build Command:** `pip install -r requirements.txt`
   * **Start Command:** `uvicorn backend.api:app --host 0.0.0.0 --port $PORT`
   * **Plan:** `Free`
5. Click **`Deploy Web Service`**. Your app will be live with a dedicated URL!

---

## 🧪 5. Testing & Verification

Run the full automated test suite covering API contracts, Pareto dominance logic, and leakage-free data preprocessing:

```bash
python -m pytest -v tests/
```

**Result:**
```text
============================= 16 passed in 16.19s =============================
```

---

## 🏛️ 6. Architecture & Tech Stack

```text
GreenML-Studio/
├── backend/
│   └── api.py              # FastAPI REST server & static file dispatcher
├── frontend/
│   ├── index.html          # Apple HIG dark/light responsive layout
│   ├── styles.css          # Frosted glass, squircle cards & typography
│   ├── app.js              # Real-time state manager & Plotly.js charts
│   ├── favicon.svg / .ico  # Eco-leaf favicon assets
├── src/
│   ├── energy_profiler.py  # CodeCarbon & process_time_ns telemetry engine
│   ├── pareto.py           # 2D Pareto dominance, knee detection, hypervolume
│   ├── data_loader.py      # Leakage-free preprocessing & stratified sampling
│   ├── extended_benchmark.py # Cross-model benchmarking & hot-swap policies
│   └── generate_master_figures.py # Publication figure generation
├── data/                   # Benchmark datasets (Diabetes, Iris, Glass, etc.)
├── outputs/                # Precomputed empirical results & trained models
├── tests/                  # Pytest integration & pipeline test suite
├── render.yaml             # Render infrastructure blueprint
└── requirements.txt        # Production Python dependencies
```

---

## 📄 License & Credits

Built for academic and empirical research in Green Software Engineering and Sustainable Machine Learning.  
**Author:** Syed Muhammad Arham ([@SMArham](https://github.com/SMArham))
