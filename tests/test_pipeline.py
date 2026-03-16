"""
Lab 2 — Learner Test File

Write your own pytest tests here. You must implement at least 3 test functions:
  - test_load_data_returns_dataframe
  - test_clean_data_no_nulls
  - test_add_features_creates_revenue

The autograder will run your tests as part of the CI check.
"""

import os
import sys
import pandas as pd
import numpy as np
import pytest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from pipeline import load_data, clean_data, add_features, generate_summary


# ─── Base Test 1 ──────────────────────────────────────────────────────────────

def test_load_data_returns_dataframe():
    """load_data should return a DataFrame with expected columns and rows."""
    df = load_data('data/sales_records.csv')

    assert isinstance(df, pd.DataFrame)
    assert len(df) > 0

    expected_columns = [
        'date',
        'store_id',
        'product_category',
        'quantity',
        'unit_price',
        'payment_method'
    ]

    for col in expected_columns:
        assert col in df.columns


# ─── Base Test 2 ──────────────────────────────────────────────────────────────

def test_clean_data_no_nulls():
    """After clean_data, quantity and unit_price should have no NaN values."""
    df = load_data('data/sales_records.csv')
    cleaned = clean_data(df)

    assert cleaned['quantity'].isna().sum() == 0
    assert cleaned['unit_price'].isna().sum() == 0


# ─── Base Test 3 ──────────────────────────────────────────────────────────────

def test_add_features_creates_revenue():
    """add_features should add a 'revenue' column equal to quantity * unit_price."""
    df = load_data('data/sales_records.csv')
    cleaned = clean_data(df)
    enriched = add_features(cleaned)

    assert 'revenue' in enriched.columns

    expected_revenue = cleaned['quantity'] * cleaned['unit_price']

    pd.testing.assert_series_equal(
        enriched['revenue'].reset_index(drop=True),
        expected_revenue.reset_index(drop=True),
        check_names=False
    )


# ─── Challenge 1 Test 1: Empty CSV ────────────────────────────────────────────

def test_empty_csv_handled_gracefully(tmp_path):
    """Pipeline should handle an empty CSV without crashing."""
    empty_file = tmp_path / "empty_sales.csv"

    empty_df = pd.DataFrame(columns=[
        'date', 'store_id', 'product_category',
        'quantity', 'unit_price', 'payment_method'
    ])
    empty_df.to_csv(empty_file, index=False)

    df = load_data(empty_file)
    cleaned = clean_data(df)
    enriched = add_features(cleaned)
    summary = generate_summary(enriched)

    assert isinstance(df, pd.DataFrame)
    assert cleaned.empty
    assert enriched.empty
    assert summary['record_count'] == 0
    assert summary['total_revenue'] == 0.0
    assert summary['avg_order_value'] == 0.0
    assert summary['top_category'] == 'No data'


# ─── Challenge 1 Test 2: All quantity null ───────────────────────────────────

def test_all_null_quantity_column_handled(tmp_path):
    """Pipeline should handle a dataset where the entire quantity column is NaN."""
    file_path = tmp_path / "all_null_quantity.csv"

    df = pd.DataFrame({
        'date': ['2026-01-01', '2026-01-02', '2026-01-03'],
        'store_id': [1, 2, 3],
        'product_category': ['Electronics', 'Audio', 'Computing'],
        'quantity': [np.nan, np.nan, np.nan],
        'unit_price': [100.0, 200.0, 300.0],
        'payment_method': ['Cash', 'Credit Card', 'Mobile Payment']
    })
    df.to_csv(file_path, index=False)

    loaded = load_data(file_path)
    cleaned = clean_data(loaded)
    enriched = add_features(cleaned)
    summary = generate_summary(enriched)

    assert cleaned['quantity'].isna().sum() == 0
    assert (cleaned['quantity'] == 0).all()
    assert 'revenue' in enriched.columns
    assert (enriched['revenue'] == 0).all()
    assert summary['record_count'] == 3


# ─── Challenge 1 Test 3: Single row ──────────────────────────────────────────

def test_single_row_csv_handled_correctly(tmp_path):
    """Pipeline should handle a single-row CSV correctly."""
    file_path = tmp_path / "single_row.csv"

    df = pd.DataFrame({
        'date': ['2026-02-10'],
        'store_id': [1],
        'product_category': ['Electronics'],
        'quantity': [2],
        'unit_price': [500.0],
        'payment_method': ['Cash']
    })
    df.to_csv(file_path, index=False)

    loaded = load_data(file_path)
    cleaned = clean_data(loaded)
    enriched = add_features(cleaned)
    summary = generate_summary(enriched)

    assert len(enriched) == 1
    assert enriched['revenue'].iloc[0] == 1000.0
    assert summary['record_count'] == 1
    assert summary['total_revenue'] == 1000.0
    assert summary['top_category'] == 'Electronics'