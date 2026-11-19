# Marine engine anomaly detection

Unsupervised anomaly detection on ship-engine telemetry, completed for Cambridge PACE Course 101 (week 5).

The practical problem is to flag unusual operating states in six continuously monitored signals so engineers can inspect the riskiest cases first. Anomalies are expected to be rare (about 1–5% of records). There are no ground-truth labels, so the work compares a statistical rule with two unsupervised models and looks at where they agree.

## Data

19,535 records and six features, with no missing values and no duplicate rows:

- Engine rpm
- Lubrication oil pressure
- Fuel pressure
- Coolant pressure
- Lubrication oil temperature
- Coolant temperature

The course provided the file and cited Devabrat (2022). A copy is in `engine.csv`. It can also be pulled from the course’s public repository:

https://raw.githubusercontent.com/fourthrevlxd/cam_dsb/main/engine.csv

## Approach

1. Exploratory checks and descriptive statistics, including a correlation heatmap.
2. Statistical IQR rule: a record is flagged if at least two features fall outside the 1.5×IQR fences at the same time. This is stricter than flagging a single extreme reading.
3. One-Class SVM with an RBF kernel (`nu=0.03`, `gamma=0.1`), so the expected outlier fraction sits at the 3% target.
4. Isolation Forest (`contamination=0.03`; the searched parameters selected `n_estimators=50`).
5. A 2D PCA “health map” of each model’s flags. The first two components explain 36.69% of variance (18.99% and 17.69%), so the scatter is only a view, not a complete picture.
6. Overlap analysis, to separate high-agreement cases from model-only flags.

## Results

Figures are in `figures/`. Counts below are taken from the executed notebook.

| Method | Setting | Anomalies | Rate |
| --- | --- | ---: | ---: |
| IQR, ≥2 features at once | 1.5×IQR | 422 | 2.16% |
| One-Class SVM | `nu=0.03`, `gamma=0.1` | 586 | 3.00% |
| Isolation Forest | `contamination=0.03` | 587 | 3.00% |

Overlap:

- All three methods: 133 records
- One-Class SVM and Isolation Forest only: 186
- IQR and One-Class SVM only: 37
- IQR and Isolation Forest only: 57
- IQR only: 195; One-Class SVM only: 230; Isolation Forest only: 211

The 133 records flagged by every method are the natural first-inspection list. The two machine-learning models agree on 319 records (133 + 186). In the PCA plots the flags do not form a single tight cluster at the edge of the cloud; they are scattered, which is consistent with “few and different” operating states rather than one failure mode.

## How to run

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
jupyter notebook notebooks/engine_anomaly_detection.ipynb
```

The notebook loads `engine.csv` from the project folder if the file is present, and otherwise downloads the public course copy.

## Tech stack

Python, pandas, NumPy, scikit-learn, matplotlib, seaborn.

## Files

- `notebooks/engine_anomaly_detection.ipynb` — full analysis with executed outputs
- `engine.csv` — telemetry used in the notebook
- `figures/` — correlation heatmap, PCA views, and the method comparison chart
