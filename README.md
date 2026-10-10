# Chronic Kidney Disease (CKD) Predictor

A Flask web app that predicts whether **Chronic Kidney Disease is present** using a
**Logistic Regression** model trained on clinical data.

- `0` = not present / normal / good
- `1` = present / abnormal / poor

The user picks options such as *Present* / *Not Present* in the form; the app converts them to `0` / `1`
before sending them to the model.

## Project structure

```
ckd-predictor/
├── app.py                # Flask application
├── train.py              # Trains the model and creates model.pkl
├── model.pkl             # Pickled StandardScaler + LogisticRegression pipeline
├── data/
│   └── kidney_disease_clean_data.csv
├── templates/
│   └── index.html
├── static/
│   └── style.css
├── requirements.txt
├── vercel.json           # Vercel config
├── Dockerfile            # For Docker-based hosting
└── README.md
```

## Model

- The `sg` (Specific Gravity) column is **dropped** before training and is not part of the form
- X = all remaining columns except the last one, y = `classification` (the last column)
- 80/20 stratified train/test split, then refit on all data for the final model
- Pipeline: `StandardScaler` -> `LogisticRegression`
- Saved with `pickle` as `model.pkl` together with the feature order
- Run `python train.py` to see test accuracy

## Run locally

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt pandas

python train.py                  # optional: re-creates model.pkl
python app.py                    # open http://127.0.0.1:5000
```

> `pandas` is only needed for `train.py`, so it is not in `requirements.txt` (keeps the deployment small).

## Run with Docker

```bash
docker build -t ckd-predictor .
docker run -p 8000:8000 ckd-predictor
# open http://localhost:8000
```

## Deploy to Vercel

1. Push this folder to a GitHub repository (make sure `model.pkl` is committed).
2. Go to [vercel.com](https://vercel.com) -> **Add New Project** -> import the repo.
3. Leave the framework preset as **Other** and click **Deploy**.

Or with the CLI:

```bash
npm i -g vercel
vercel
vercel --prod
```

Notes:
- Vercel reads `vercel.json` and `requirements.txt`; it does **not** use the `Dockerfile`
  (that file is for Render, Railway, Fly.io, a VPS, etc.).
- `scikit-learn` must stay pinned to the version used to create `model.pkl`.
  If you retrain with a different version, update `requirements.txt`.
- If the deployment exceeds Vercel's size limit, consider Render or Railway with the Dockerfile.

## Disclaimer

For educational purposes only. Not a medical diagnosis.
