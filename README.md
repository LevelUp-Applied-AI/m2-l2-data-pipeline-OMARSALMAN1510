[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/7UAbFuim)
# Lab 2 — Data Pipeline: Retail Sales Analysis

**Module 2 — Programming for AI & Data Science**

---

## Overview

You will build a complete, modular data pipeline that processes a messy retail sales dataset, computes summary statistics, and generates publication-ready charts.

The pipeline is structured as a set of focused functions — each responsible for one stage of processing. You will also write pytest tests to validate your own code before submitting.

---

## Dataset

`data/sales_records.csv` — ~500 rows of synthetic retail sales data from a fictional Jordanian electronics store chain.

| Column | Type | Notes |
|---|---|---|
| `date` | string | YYYY-MM-DD; ~3% in non-standard format |
| `store_id` | int | Store IDs 1–5 |
| `product_category` | string | Electronics, Home Appliances, Mobile Accessories, Computing, Audio |
| `quantity` | int | ~5% missing values |
| `unit_price` | float | ~5% missing values |
| `payment_method` | string | Cash, Credit Card, Mobile Payment, Bank Transfer |

---

## Pipeline Architecture

```
load_data()
    └── clean_data()
            └── add_features()
                    ├── generate_summary()
                    └── create_visualizations()
```

Each function takes the output of the previous stage. `main()` calls them in order.

---

## Function Signatures

```python
def load_data(filepath: str) -> pd.DataFrame
def clean_data(df: pd.DataFrame) -> pd.DataFrame
def add_features(df: pd.DataFrame) -> pd.DataFrame
def generate_summary(df: pd.DataFrame) -> dict
def create_visualizations(df: pd.DataFrame, output_dir: str = 'output') -> None
def main() -> None
```

**`generate_summary` must return a dict with exactly these keys:**
- `total_revenue` — sum of all revenue
- `avg_order_value` — mean order value
- `top_category` — product category with highest total revenue
- `record_count` — number of rows in the DataFrame

**`create_visualizations` must save exactly 3 PNG files to `output/`:**
- `revenue_by_category.png` — bar chart: total revenue by product category
- `daily_revenue_trend.png` — line chart: daily revenue aggregated by date
- `avg_order_by_payment.png` — horizontal bar chart: average order value by payment method

---

## Tasks

### 1. Complete `pipeline.py`

Open `pipeline.py` and implement each function. Follow the `TODO:` markers and docstrings. Do not change the function signatures.

Key requirements:
- `clean_data`: fill missing `quantity` and `unit_price` with their column medians; parse `date` to datetime using `errors='coerce'`
- `add_features`: add `revenue` column (`quantity * unit_price`) and `day_of_week` column
- `create_visualizations`: use `fig.savefig()` to save charts; do **not** use `plt.show()` (it blocks pipeline scripts); close each figure with `plt.close(fig)` after saving
- `main`: orchestrate the full pipeline with progress print statements

### 2. Complete `tests/test_pipeline.py`

The test file has 3 stubs. Implement each one:

- `test_load_data_returns_dataframe` — load the CSV, confirm it returns a DataFrame with the expected columns
- `test_clean_data_no_nulls` — confirm no NaN values remain in `quantity` and `unit_price` after cleaning
- `test_add_features_creates_revenue` — confirm `revenue` column exists and equals `quantity * unit_price`

Run your tests locally before submitting:

```bash
pytest tests/test_pipeline.py -v
```

All 3 tests must pass.

### 3. Run the full pipeline

```bash
python pipeline.py
```

This should:
- Print progress messages at each stage
- Print the summary statistics dict
- Create 3 PNG files in `output/`
- Print `Pipeline complete.`

Commit your `output/` PNG files — they are part of your submission.

---

## Submission

1. Create branch `lab-2/data-pipeline` from `main`
2. Complete `pipeline.py` and `tests/test_pipeline.py`
3. Run `python pipeline.py` — confirm 3 PNGs in `output/`
4. Run `pytest tests/ -v` — confirm all tests pass
5. Commit all files (including `output/*.png`) and push
6. Open a PR to `main`
7. In the PR description, include:
   - Your summary statistics output (paste from terminal)
   - Confirmation that all tests pass locally

Paste your PR URL into the TalentLMS Lab 2 assignment submission field.
