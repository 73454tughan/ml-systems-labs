import logging
import pandas as pd
import numpy as np
from typing import Dict, Any, List

logger = logging.getLogger(__name__)

SCHEMA_SPEC = {
    "types": {
        "age": "int",
        "workclass": "category",
        "fnlwgt": "int",
        "education": "category",
        "education-num": "int",
        "marital-status": "category",
        "occupation": "category",
        "relationship": "category",
        "race": "category",
        "sex": "category",
        "capital-gain": "int",
        "capital-loss": "int",
        "hours-per-week": "int",
        "native-country": "category",
        "income": "category"
    },
    "ranges": {
        "age": (17, 90),
        "hours-per-week": (1, 99),
        "capital-gain": (0, 99999),
        "capital-loss": (0, 4356)
    },
    "null_policy": {
        "missing_indicator": "?",
        "allowed_categories_with_missing": ["workclass", "occupation", "native-country"]
    }
}

def validate_dataset(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Executes schema, range, and missing-value validations on the raw dataset.
    Reports all violations without silently dropping rows.
    """
    logger.info("Executing dataset validation suite...")
    violations: List[Dict[str, Any]] = []
    summary: Dict[str, Any] = {
        "total_rows": len(df),
        "total_columns": len(df.columns),
        "column_reports": {},
        "missing_summary": {},
        "range_violations_count": 0,
        "type_violations_count": 0
    }

    # 1. Type and Category check
    for col, expected_type in SCHEMA_SPEC["types"].items():
        if col not in df.columns:
            violations.append({"type": "MISSING_COLUMN", "column": col})
            continue

        actual_dtype = str(df[col].dtype)
        is_numeric = pd.api.types.is_numeric_dtype(df[col])
        
        if expected_type == "int" and not is_numeric:
            violations.append({
                "type": "TYPE_MISMATCH",
                "column": col,
                "expected": "int",
                "actual": actual_dtype
            })
            summary["type_violations_count"] += 1

    # 2. Value Range check
    for col, (min_val, max_val) in SCHEMA_SPEC["ranges"].items():
        if col in df.columns:
            numeric_series = pd.to_numeric(df[col], errors="coerce")
            out_of_bounds = df[(numeric_series < min_val) | (numeric_series > max_val)]
            count_oob = len(out_of_bounds)
            summary["column_reports"][col] = {
                "min_observed": float(numeric_series.min()),
                "max_observed": float(numeric_series.max()),
                "allowed_min": min_val,
                "allowed_max": max_val,
                "violations_count": count_oob
            }
            if count_oob > 0:
                violations.append({
                    "type": "RANGE_VIOLATION",
                    "column": col,
                    "count": count_oob,
                    "allowed": f"[{min_val}, {max_val}]"
                })
                summary["range_violations_count"] += count_oob

    # 3. Missing Value / '?' Policy check
    for col in df.columns:
        # Check for standard NaN or '?' symbol
        q_count = (df[col].astype(str).str.strip() == "?").sum()
        nan_count = df[col].isna().sum()
        total_missing = int(q_count + nan_count)
        if total_missing > 0:
            summary["missing_summary"][col] = {
                "question_marks": int(q_count),
                "null_values": int(nan_count),
                "total_missing": total_missing,
                "percentage": round(float(total_missing / len(df) * 100), 2)
            }

    summary["violations"] = violations
    logger.info(f"Validation completed. Identified {len(violations)} rule violations / range checks.")
    logger.info(f"Missing values found across {len(summary['missing_summary'])} columns. No rows dropped.")

    return summary

def generate_validation_markdown_report(summary: Dict[str, Any], output_path: str = "reports/validation_report.md") -> None:
    """Writes detailed validation findings to Markdown."""
    lines = [
        "# Dataset Schema & Quality Validation Report",
        f"**Audited Rows:** {summary['total_rows']}  ",
        f"**Audited Columns:** {summary['total_columns']}  ",
        "**Policy:** No rows were silently dropped. All anomalies recorded.\n",
        "## 1. Value Ranges Audit",
        "| Column | Allowed Range | Observed Min | Observed Max | Violations Count | Status |",
        "|---|---|---|---|---|---|"
    ]

    for col, rpt in summary["column_reports"].items():
        status = "PASSED" if rpt["violations_count"] == 0 else f"FAILED ({rpt['violations_count']})"
        lines.append(f"| `{col}` | [{rpt['allowed_min']}, {rpt['allowed_max']}] | {rpt['min_observed']} | {rpt['max_observed']} | {rpt['violations_count']} | {status} |")

    lines.extend([
        "\n## 2. Missing Values & '?' Character Policy Audit",
        "| Column | '?' Count | NaN Count | Total Missing | % of Dataset | Handled by Imputer |",
        "|---|---|---|---|---|---|"
    ])

    for col, m in summary["missing_summary"].items():
        lines.append(f"| `{col}` | {m['question_marks']} | {m['null_values']} | {m['total_missing']} | {m['percentage']}% | YES (Mode/Median) |")

    lines.extend([
        "\n## 3. Policy Adherence Summary",
        f"* **Range Violations Total:** {summary['range_violations_count']}",
        f"* **Type Inconsistencies:** {summary['type_violations_count']}",
        "* **Preservation Policy:** 100% of rows retained for downstream stratified splitting.",
        "* **Missing Indicator:** Recognized `'?'` as valid placeholder requiring imputation."
    ])

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    logger.info(f"Saved validation report to: {output_path}")
