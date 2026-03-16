import json
import pandas as pd


def check_null_percentage(df, column, max_null_pct):
    """Check that a column has no more than max_null_pct null values."""
    if column not in df.columns:
        return {
            "rule": f"{column}_null_percentage",
            "status": "fail",
            "details": f"Column '{column}' not found"
        }

    if len(df) == 0:
        null_pct = 0.0
    else:
        null_pct = float(df[column].isna().mean() * 100)

    status = "pass" if null_pct <= max_null_pct else "fail"

    return {
        "rule": f"{column}_null_percentage",
        "status": status,
        "details": f"{null_pct:.2f}% null values (max allowed: {max_null_pct}%)"
    }


def check_value_range(df, column, min_value=None, max_value=None):
    """Check that all non-null values in a column fall within a valid range."""
    if column not in df.columns:
        return {
            "rule": f"{column}_value_range",
            "status": "fail",
            "details": f"Column '{column}' not found"
        }

    series = df[column].dropna()

    if series.empty:
        return {
            "rule": f"{column}_value_range",
            "status": "pass",
            "details": "No non-null values to validate"
        }

    invalid_mask = pd.Series(False, index=series.index)

    if min_value is not None:
        invalid_mask = invalid_mask | (series < min_value)

    if max_value is not None:
        invalid_mask = invalid_mask | (series > max_value)

    invalid_count = int(invalid_mask.sum())
    status = "pass" if invalid_count == 0 else "fail"

    return {
        "rule": f"{column}_value_range",
        "status": status,
        "details": f"{invalid_count} invalid values outside range [{min_value}, {max_value}]"
    }


def check_duplicate_date_store(df, date_col="date", store_col="store_id"):
    """
    Check that there are no duplicate dates within the same store_id.
    """
    if date_col not in df.columns or store_col not in df.columns:
        return {
            "rule": "duplicate_date_within_store",
            "status": "fail",
            "details": f"Missing required columns: {date_col} or {store_col}"
        }

    duplicates = df.duplicated(subset=[date_col, store_col]).sum()
    status = "pass" if duplicates == 0 else "fail"

    return {
        "rule": "duplicate_date_within_store",
        "status": status,
        "details": f"{int(duplicates)} duplicate rows found for ({date_col}, {store_col})"
    }


def run_validation(df, expectations):
    """
    Run a list of validation rules against a DataFrame.

    Each expectation is a dict like:
    {"type": "null_percentage", "column": "quantity", "max_null_pct": 10}
    """
    results = []

    for exp in expectations:
        exp_type = exp.get("type")

        if exp_type == "null_percentage":
            results.append(
                check_null_percentage(
                    df,
                    column=exp["column"],
                    max_null_pct=exp["max_null_pct"]
                )
            )

        elif exp_type == "value_range":
            results.append(
                check_value_range(
                    df,
                    column=exp["column"],
                    min_value=exp.get("min_value"),
                    max_value=exp.get("max_value")
                )
            )

        elif exp_type == "duplicate_date_store":
            results.append(
                check_duplicate_date_store(
                    df,
                    date_col=exp.get("date_col", "date"),
                    store_col=exp.get("store_col", "store_id")
                )
            )

        else:
            results.append({
                "rule": "unknown_rule",
                "status": "fail",
                "details": f"Unknown expectation type: {exp_type}"
            })

    return results


def summarize_validation(results):
    """Summarize validation results."""
    passed = sum(1 for r in results if r["status"] == "pass")
    failed = sum(1 for r in results if r["status"] == "fail")

    return {
        "total_rules": len(results),
        "passed": passed,
        "failed": failed,
        "results": results
    }


def save_validation_report(report, filepath):
    """Save validation report as JSON."""
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)


def print_validation_report(report, title="Validation Report"):
    """Print validation report in a readable format."""
    print(f"=== {title} ===")
    print(f"Total Rules: {report['total_rules']}")
    print(f"Passed: {report['passed']}")
    print(f"Failed: {report['failed']}")

    for item in report["results"]:
        print(f"- {item['rule']}: {item['status'].upper()} | {item['details']}")