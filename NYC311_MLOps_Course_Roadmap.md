# NYC 311: MLOps from zero to a working system

## The project

**Question:** When a new NYC 311 service request arrives, will it remain open more than 72 hours after it was created?

**Prediction moment:** Immediately after a request is created.

**Allowed starting information:** Request type, borough, and the time the request was created. We may add other fields only after checking that they are known at creation time.

**Target:** `1` means the request was not closed within 72 hours; `0` means it was closed within 72 hours. A request younger than 72 hours without a closure is *unknown* and must not be used as a training label.

**Intended use:** A learning and operational demonstration. Predictions are for review and planning, not automatic decisions about who receives city services.

**Data source:** [NYC Open Data: 311 Service Requests from 2020 to Present](https://data.cityofnewyork.us/Social-Services/311-Service-Requests-from-2020-to-Present/erm2-nwe9). The city says the dataset is updated daily and existing rows can change.

## What we will finish with

1. Colab notebooks that explain and run every learning stage.
2. A repeatable data collection and validation pipeline.
3. A trained model with reproducible experiments and a saved version.
4. A prediction function and local API that accept a new request and return a probability, plus a free scheduled batch prediction workflow.
5. Automated tests and checks before a model is released.
6. Scheduled data collection, performance monitoring, and a retraining rule.
7. A README explaining how another person can run and maintain the system.

Colab is our classroom and development environment. Because its runtime is temporary, scheduled jobs and an always-available prediction service will eventually run outside Colab.

## Stage map

| Stage | What we build | What you learn | Checkpoint |
|---|---|---|---|
| 0. Orientation | Project question, success measure, folder plan | What MLOps is; data, model, and operations | Explain the prediction in one sentence |
| 1. Meet the data | Small, live API request in Colab | Rows, columns, APIs, missing values, data dictionary | Identify the request ID, input fields, and outcome field |
| 2. Collect data | First immutable historical snapshot and manifest | Data ingestion, changing source data, checksums | Reopen the saved data and verify its checksum |
| 3. Define the label | 72-hour outcome and eligible records | Prediction time, delayed labels, leakage | Explain why an open 2-hour-old request has no label yet |
| 4. Explore and validate | Charts and automatic data checks | Data quality, missingness, class balance, bias | Reject deliberately broken input data |
| 5. Broaden data and build a baseline | Fixed historical dates, simple rule, first machine learning model | Sampling limits, chronological split, precision, recall, calibration | Compare the model with a baseline on later requests |
| 6. Reproducible training | Preprocessing pipeline, fixed settings, experiment log | Repeatability, model comparison, artifacts | Re-run training and identify the chosen model |
| 7. Serve predictions | Input contract and prediction API | Deployment, input validation, model loading, version IDs | Send a new request and receive a probability |
| 8. Monitor | Prediction log joined with later outcomes | Data drift, performance drift, alert thresholds | Produce a report for a later time window |
| 9. Automate and release | Hardened collector, tests, scheduled workflow, release rules | CI/CD, retraining, rollback, operational ownership | Run the project from fresh data to monitoring report |

## Free production path

The required course uses free Colab and Google Drive for learning, then a public GitHub repository with [GitHub Actions](https://docs.github.com/en/actions/concepts/billing-and-usage) for scheduled jobs and [GitHub Pages](https://docs.github.com/en/pages/getting-started-with-github-pages/what-is-github-pages) for a published results page. We will build and test an API locally; a continuously available hosted API is not required for the free path. Free services have limits, so the result is a production-style learning system rather than a guaranteed-uptime city service.

## How I will teach each cell

For each cell, we will cover: **what it does → why we need it → what each important line means → expected output → one small check or exercise**. You can paste your output or error here, and we will resolve it before moving on. We will start with simple Python and add tools only when their purpose is clear.

## Important design rules

- Split training and testing by **creation time** so the test simulates future requests.
- Never give the model `closed_date`, `status`, `resolution_description`, or any other information learned after creation.
- Save an immutable raw data snapshot for each collection run. The source can change after we fetch it, so re-fetching the same query is not the same as reproducing a run.
- Record the exact query, collection time, row count, schema, and SHA-256 checksum in a manifest next to every snapshot.
- Pin package versions, set random seeds where relevant, and record the code commit and model settings for every training run.
- Make data collection safe to repeat: use request IDs for deduplication and fail visibly when a page, schema check, or checksum fails.
- Keep a model version and a release decision for every deployed model; preserve the previous version for rollback.
- Measure prediction quality only after enough time has passed for the 72-hour outcome to be known.
- Monitor quality by time and by request type and borough, not only as one overall score.
- Keep a simple baseline so more complex models must earn their place.

## Learning pace

Start with **Stage 1**. When you can explain its checkpoint, move to Stage 2. Each stage will have its own Colab lesson and a small finished piece of the operational project. No GPU or paid Colab plan is required for the planned starter models.
