"""
Lab 2 — Data Pipeline: Retail Sales Analysis
Module 2 — Programming for AI & Data Science

Extended for Challenge 2:
- Config-driven pipeline using JSON
"""

import os
import json
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# ─── Default Configuration ────────────────────────────────────────────────────

DEFAULT_CONFIG_PATH = "config_sales.json"


# ─── Config Helpers ───────────────────────────────────────────────────────────

def load_config(config_path=DEFAULT_CONFIG_PATH):
    """Load pipeline configuration from JSON file."""
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def apply_fill_strategy(series, strategy):
    """Apply fill strategy to a pandas Series."""
    if strategy == "median":
        value = series.median()
        if pd.isna(value):
            value = 0
        return series.fillna(value)

    if strategy == "mean":
        value = series.mean()
        if pd.isna(value):
            value = 0
        return series.fillna(value)

    if strategy == "zero":
        return series.fillna(0)

    return series


# ─── Pipeline Functions ───────────────────────────────────────────────────────

def load_data(filepath):
    """Load records from a CSV file."""
    df = pd.read_csv(filepath)
    print(f"Loaded {len(df)} records from {filepath}")
    return df


def clean_data(df, fillna_config=None):
    """Handle missing values and fix data types."""
    df = df.copy()

    if fillna_config is None:
        fillna_config = {
            "quantity": "median",
            "unit_price": "median"
        }

    if df.empty:
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
        print("Cleaned data: 0 records")
        return df

    for column, strategy in fillna_config.items():
        if column in df.columns:
            df[column] = apply_fill_strategy(df[column], strategy)

    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")

    if "quantity" in df.columns and "unit_price" in df.columns:
        df = df.dropna(subset=["quantity", "unit_price"], how="all")

    print(f"Cleaned data: {len(df)} records")
    return df


def add_features(df):
    """Compute derived columns."""
    df = df.copy()

    if df.empty:
        df["revenue"] = pd.Series(dtype="float64")
        df["day_of_week"] = pd.Series(dtype="object")
        return df

    if "quantity" in df.columns and "unit_price" in df.columns:
        df["revenue"] = df["quantity"] * df["unit_price"]
    else:
        df["revenue"] = 0

    if "date" in df.columns:
        df["day_of_week"] = df["date"].dt.day_name()
    else:
        df["day_of_week"] = "Unknown"

    return df


def generate_summary(df, summary_groupby="product_category"):
    """Compute summary statistics."""
    if df.empty or "revenue" not in df.columns:
        return {
            "total_revenue": 0.0,
            "avg_order_value": 0.0,
            "top_category": "No data",
            "record_count": 0
        }

    if summary_groupby in df.columns:
        grouped = df.groupby(summary_groupby)["revenue"].sum()
        top_category = grouped.idxmax() if not grouped.empty else "No data"
    else:
        top_category = "No data"

    avg_order_value = df["revenue"].mean()
    if pd.isna(avg_order_value):
        avg_order_value = 0.0

    return {
        "total_revenue": float(df["revenue"].sum()),
        "avg_order_value": float(avg_order_value),
        "top_category": top_category,
        "record_count": int(len(df))
    }


def create_visualizations(df, output_dir="output", charts=None):
    """Create and save charts based on config."""
    os.makedirs(output_dir, exist_ok=True)

    if charts is None:
        charts = [
            "revenue_by_category",
            "daily_revenue_trend",
            "avg_order_by_payment"
        ]

    if "revenue_by_category" in charts:
        revenue_by_category = (
            df.groupby("product_category")["revenue"].sum()
            if not df.empty and "product_category" in df.columns
            else pd.Series(dtype="float64")
        )
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(revenue_by_category.index.astype(str), revenue_by_category.values)
        ax.set_title("Total Revenue by Product Category")
        ax.set_xlabel("Product Category")
        ax.set_ylabel("Revenue")
        ax.tick_params(axis="x", rotation=45)
        fig.savefig(f"{output_dir}/revenue_by_category.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    if "daily_revenue_trend" in charts:
        daily_revenue = (
            df.groupby("date")["revenue"].sum().sort_index()
            if not df.empty and "date" in df.columns
            else pd.Series(dtype="float64")
        )
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(daily_revenue.index, daily_revenue.values)
        ax.set_title("Daily Revenue Trend")
        ax.set_xlabel("Date")
        ax.set_ylabel("Revenue")
        fig.savefig(f"{output_dir}/daily_revenue_trend.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    if "avg_order_by_payment" in charts:
        avg_order_by_payment = (
            df.groupby("payment_method")["revenue"].mean()
            if not df.empty and "payment_method" in df.columns
            else pd.Series(dtype="float64")
        )
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.barh(avg_order_by_payment.index.astype(str), avg_order_by_payment.values)
        ax.set_title("Average Order Value by Payment Method")
        ax.set_xlabel("Average Revenue")
        ax.set_ylabel("Payment Method")
        fig.savefig(f"{output_dir}/avg_order_by_payment.png", dpi=150, bbox_inches="tight")
        plt.close(fig)


def run_pipeline(config_path=DEFAULT_CONFIG_PATH):
    """Run full pipeline from config."""
    config = load_config(config_path)

    input_file = config["input_file"]
    output_dir = config.get("output_dir", "output")
    fillna_config = config.get("fillna", {})
    summary_groupby = config.get("summary_groupby", "product_category")
    charts = config.get("charts", [])

    df = load_data(input_file)
    df = clean_data(df, fillna_config=fillna_config)
    df = add_features(df)
    summary = generate_summary(df, summary_groupby=summary_groupby)

    print("=== Summary ===")
    print(f"Total Revenue: {summary['total_revenue']}")
    print(f"Average Order Value: {summary['avg_order_value']}")
    print(f"Top Category: {summary['top_category']}")
    print(f"Record Count: {summary['record_count']}")

    create_visualizations(df, output_dir=output_dir, charts=charts)
    print("Pipeline complete.")


def main():
    """Run the full pipeline end-to-end."""
    config_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_CONFIG_PATH
    run_pipeline(config_path)


if __name__ == "__main__":
    main()