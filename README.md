# NYC 311 MLOps course and shadow API

This repository contains the course notebooks, reusable prediction code, and automated code checks. The trained model and raw NYC 311 snapshots are stored separately. The current model's release decision is **HOLD_FOR_OPERATIONAL_USE**: it missed all 552 late HEAT/HOT WATER cases in its final test. The API is for local demonstration and shadow evaluation only.

## Project question

When a new NYC 311 service request arrives, will it remain open more than 72 actual hours after creation? The model accepts `complaint_type`, `borough`, and NYC local `created_date`. It derives month, hour, and weekday from that date. It never uses closure or resolution fields as inputs.

## Learn in Colab

Open the numbered files in `notebooks/` in order. The [course roadmap](NYC311_MLOps_Course_Roadmap.md) explains the stages. Raw snapshots, manifests, and fitted experiments from the lessons remain in your Google Drive; they are not committed here.

## Run code checks locally

Use Python **3.13.15** to match the saved model environment:

```bash
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
```

GitHub Actions runs the same code checks on each push and pull request. These checks validate request handling and API behavior with a fake model. The real model must be verified separately with its SHA-256 and a smoke prediction before any release.

## Run the local shadow API

Download `model.joblib` and `release.json` from the corresponding Hugging Face model repository into a local directory. Verify the model hash in `release.json`; `load_release` performs this check before loading. Do not commit these files here.

Set `NYC311_RELEASE_DIR` to that directory, then run:

```bash
python -m uvicorn nyc311_shadow.api:app --host 127.0.0.1 --port 8000
```

On Windows PowerShell, first run `$env:NYC311_RELEASE_DIR = "C:\path\to\model-folder"`. Visit `http://127.0.0.1:8000/docs` for the local API documentation. `GET /health` reports the model ID and HOLD status. `POST /predict` accepts a request such as:

```json
{"complaint_type":"HEAT/HOT WATER","borough":"BRONX","created_date":"2026-09-26T10:00:00"}
```

The response includes a probability, threshold decision, model ID, shadow mode, and a HEAT/HOT WATER warning. It does not take action on any NYC 311 case.

## Lineage and limits

- Source: [NYC Open Data 311 Service Requests](https://data.cityofnewyork.us/d/erm2-nwe9)
- Final model ID: `20260926T143549799163Z`
- Decision ID: `20260926T143947829974Z`
- Final test: 64,751 March–April 2026 requests; late-case recall 0.871, precision 0.672
- Important failure: HEAT/HOT WATER recall 0.0 on 552 late requests

March–April 2026 has already served as a final test and must not be used to select a replacement model. A future candidate needs a later untouched test period and a new release decision.

