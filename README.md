# Chest X-ray Classifier (Flask + Vercel)

A small Flask web app that loads the CNN you trained with `train_xray_model.py`
and predicts NORMAL vs PNEUMONIA for an uploaded chest X-ray.

> Educational demo only. Not a medical device; do not use it for diagnosis.

```
xray-flask-app/
├── app.py                 Flask app (Vercel entrypoint: the `app` object)
├── convert_to_tflite.py   .keras -> .tflite converter (run once on your Mac)
├── templates/index.html   upload page
├── model/                 xray_model.tflite + classes.json go here
├── requirements.txt       dependencies for the deployed app
├── vercel.json            Vercel function settings
└── .python-version        Python version used on Vercel
```

## 1. Convert your trained model

Why: TensorFlow is about 1.5 GB installed, far over Vercel's 500 MB limit for
Python functions. A `.tflite` copy of the model (about 14 MB) runs with a runtime of
roughly 60 MB installed and gives the same predictions.

Run this on your Mac, in the environment where you trained the model:

```bash
python convert_to_tflite.py --model /path/to/xray_model.keras
```

It writes `model/xray_model.tflite` and `model/classes.json`, and prints how closely
the converted model matches the original (the difference should be tiny).

## 2. Run locally

Use the same virtual environment you trained in (TensorFlow is already there), then:

```bash
pip install flask pillow
python app.py
```

Open http://127.0.0.1:8000 (port 5000 is skipped because macOS AirPlay uses it).
Change the port with `PORT=9000 python app.py`.

## 3. Deploy to Vercel

```bash
npm install -g vercel      # needs CLI 48.2.10 or newer
vercel login
vercel                     # preview deployment
vercel --prod              # production deployment
```

Vercel detects Flask from `requirements.txt` and uses `app.py` as the entrypoint,
and `vercel.json` sets the function timeout and leaves out files the app doesn't need.
`model/xray_model.tflite` must exist before you deploy, because it is bundled with the app.

If you use Git instead, commit the `.tflite` file (`*.keras` is ignored on purpose) and
import the repository in the Vercel dashboard.

## API

`POST /predict` with a multipart form field named `file`:

```bash
curl -F "file=@some_xray.jpeg" http://127.0.0.1:8000/predict
```

```json
{"label": "PNEUMONIA", "confidence": 0.97,
 "probabilities": {"NORMAL": 0.03, "PNEUMONIA": 0.97}}
```

`GET /health` reports whether the model file was found.

## Limits to know about

- Vercel rejects request bodies above 4.5 MB, so the app accepts images up to 4 MB.
- The first request after a period of inactivity is slower (cold start while the model loads).
