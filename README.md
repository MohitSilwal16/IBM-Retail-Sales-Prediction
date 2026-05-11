# IBM Retail Sales Prediction

A full-stack web application for domain-agnostic retail sales forecasting, built as part of an IBM project under the mentorship of **Anoj Arun Dixit (IBM Mentor)**. The system allows businesses to upload historical sales data, train machine learning models, and query predicted sales counts filtered by any combination of product attributes.

---

## Overview

This application is **domain-agnostic** — it works for any retail category (cars, electronics, apparel, etc.) as long as the data follows the required structure. It enables:

- Uploading historical sales datasets
- Training forecasting models segmented by product attribute combinations
- Querying predicted sales counts for any attribute combination and date range
- Storing and retrieving trained model artifacts via object storage

---

## Data Format

The application enforces a specific CSV structure:

- **`date`** — mandatory column (date of sale)
- **All other columns** — treated as categorical product attributes (e.g., `color`, `brand`, `category`, `size`)
- **No ID columns required** — product or record IDs are not needed
- **No quantity column** — each row represents a single product sold; counts are derived by grouping

Example:

| date       | color | brand   | category    |
|------------|-------|---------|-------------|
| 2023-01-05 | red   | Sony    | electronics |
| 2023-01-05 | black | Samsung | electronics |
| 2023-01-06 | red   | Sony    | electronics |

Demo data files are available in the `/demo` directory of this repository.

---

## Architecture

```
Client (Browser)
      │
      ▼
   Nginx (Reverse Proxy, port 80/443)   ← SSL manually configured, not in repo
      │
      ▼
   Gunicorn (WSGI Server)
      │
      ▼
   Flask App (Python)
      ├── SQLAlchemy ──► AWS RDS (PostgreSQL)
      └── Boto3 Client ──► AWS S3 (model artifacts, uploaded files)
```

In development, Flask's built-in server is used with hot-reload via Docker Compose watch mode. In production, Gunicorn serves the app behind Nginx. The public domain and SSL certificate were manually provisioned and are not part of this repository.

---

## Tech Stack

| Layer            | Technology                           |
|------------------|--------------------------------------|
| Web Framework    | Flask + Jinja2                       |
| Frontend         | Tailwind CSS                         |
| ORM              | SQLAlchemy                           |
| Object Storage   | AWS S3                               |
| Database         | AWS RDS (PostgreSQL)                 |
| WSGI Server      | Gunicorn                             |
| Reverse Proxy    | Nginx                                |
| ML Model         | XGBoost                              |
| Containerization | Docker + Docker Compose              |
| ML Training      | AWS SageMaker (Local Mode)           |
| Deployment       | AWS EC2                              |

### Development vs Production Infrastructure

To keep **development costs low**, self-hosted or lightweight alternatives are used locally in place of managed AWS services. The production stack runs entirely on AWS:

| Concern        | Development (cost-saving local alternative) | Production  |
|----------------|---------------------------------------------|-------------|
| Object Storage | MinIO (self-hosted, S3-compatible API)      | AWS S3      |
| Database       | SQLite                                      | AWS RDS     |
| ML Training    | SageMaker Local Mode                        | SageMaker (pending instance approval) |

> **Note:** MinIO and SQLite are **development-only** substitutes chosen purely to avoid AWS costs during the build phase. The application code targets AWS S3 and RDS — MinIO's S3-compatible API and SQLite are drop-in replacements that require no code changes to swap out. SageMaker remote training instances were not approved during development, so Local Mode is used throughout; the training code is structured for straightforward migration to remote SageMaker once access is available.

---

## Setup

Clone the repo and configure environment variables:

```bash
git clone https://github.com/MohitSilwal16/IBM-Retail-Sales-Prediction.git
cd IBM-Retail-Sales-Prediction
```

A demo `dev.env` file is provided under `env/` — copy and edit it for production use as well, adjusting values as needed.

**Makefile targets:**

| Target      | Description                                           |
|-------------|-------------------------------------------------------|
| `make dev`  | Tear down, rebuild, and start dev stack with watch    |
| `make prod` | Rebuild and start production stack (detached)         |

---

## ML Pipeline

### Why XGBoost?

The initial approach evaluated three classical time-series models — **Prophet, ARIMA, and SARIMA** — all of which proved unsuitable for this problem's requirements.

**ARIMA and SARIMA** are autoregressive models built specifically for stationary univariate time series. ARIMA models a single numeric series using past values (autoregressive terms), differencing (to enforce stationarity), and moving average terms. SARIMA extends this to handle one seasonal component, but it still operates on a single series with no mechanism to accept additional input features. Categorical attributes like `color` or `brand` cannot be passed as inputs — ARIMA's model formulation simply has no place for them.

**Prophet** (Meta's forecasting library) is also fundamentally a univariate model, decomposing a single time series into trend, seasonality, and holiday components. While Prophet does support `add_regressor()` to include exogenous variables, this comes with a critical constraint: **the regressor's future values must be known at inference time**. Prophet was designed for continuous numeric regressors (e.g., temperature, promotional spend) where you have or can separately forecast future values. Passing arbitrary categorical product attributes — whose future "values" are simply the filter conditions being queried — does not fit this paradigm. You cannot tell Prophet "predict sales for `color=red, brand=Sony` over the next 30 days" without either pre-computing a separate series for that combination or engineering a workaround that defeats the purpose of using Prophet in the first place.

This exposes the core problem: **the requirement is to support arbitrary attribute filter combinations at inference time** (e.g., `color=red, brand=Sony` or `category=electronics, brand=Samsung`). Forcing any of these three models into that use case would mean training a completely separate model for every possible attribute combination — a combinatorial explosion that does not scale.

The solution was to reframe the problem as a **supervised regression task**: group records by `date + attribute combination` to derive daily sales counts, then train a single XGBoost model that takes date-derived features (day of week, month, quarter, year, etc.) and encoded categorical attributes together as input features. This allows any attribute combination to be queried within a single model at inference time, with no per-combination retraining required.

XGBoost is well suited to this formulation. It handles mixed numeric and encoded categorical features natively, captures non-linear interactions between date and product attributes, and has demonstrated strong performance on retail demand forecasting in practice. The approach follows a **simple-to-complex philosophy to avoid overfitting** — if XGBoost proves insufficient, the next step would be a sequence model such as GRU or LSTM.

### Training Flow

1. User uploads a historical sales CSV
2. File is stored in **AWS S3** (MinIO locally)
3. Rows are grouped by date + attribute combination to derive counts
4. Categorical attributes are encoded; date features are extracted
5. XGBoost model is trained via **SageMaker Local Mode**
6. Trained model artifact is serialized and stored back in S3
7. Job metadata is recorded in **AWS RDS** via SQLAlchemy

### Inference Flow

1. User selects an attribute combination and date range
2. App loads the model artifact from S3
3. Predicted sales count is returned and displayed

---

## API & Features

- Upload sales data files
- Trigger model training jobs
- Query predictions by attribute combination and date range
- View available attribute options per trained model
- Models loaded at server init for low-latency inference

---

### Wildcard Attribute Queries

Any attribute field can be left **blank** during inference to query across all values of that dimension.

For example, leaving `color` blank while setting `type=SUV` aggregates predictions over every
color variant — `color=red, type=SUV`, `color=blue, type=SUV`, `color=black, type=SUV`, and so on
— returning a combined sales count without needing to know or specify the color.

Multiple fields can be left blank simultaneously. The model runs a separate inference pass for
each combination that needs to be resolved and sums the results, so **queries with more blank
fields will take longer** depending on the cardinality of the omitted attributes.

This is useful when you care about a subset of dimensions (e.g. "how many SUVs total,
regardless of color or brand?") without having to manually enumerate and sum every combination yourself.

---
