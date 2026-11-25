"""
Data Quality and Validation for Learning Analytics Relational Tables.
"""

from pathlib import Path
from typing import Any

import pandas as pd


class DataValidator:
    def __init__(self, raw_data_dir: Path):
        self.raw_data_dir = Path(raw_data_dir)
        self.required_tables = [
            "courses.csv",
            "assessments.csv",
            "vle.csv",
            "studentInfo.csv",
            "studentRegistration.csv",
            "studentAssessment.csv",
            "studentVle.csv",
        ]

    def validate_presence(self) -> dict[str, bool]:
        report = {}
        for tbl in self.required_tables:
            path = self.raw_data_dir / tbl
            report[tbl] = path.exists() and path.stat().st_size > 0
        return report

    def audit_quality(self) -> dict[str, Any]:
        audit = {}
        for tbl in self.required_tables:
            path = self.raw_data_dir / tbl
            if not path.exists():
                audit[tbl] = {"status": "MISSING"}
                continue
            df = pd.read_csv(path)
            null_counts = df.isnull().sum().to_dict()
            duplicate_count = int(df.duplicated().sum())
            audit[tbl] = {
                "row_count": len(df),
                "col_count": len(df.columns),
                "columns": list(df.columns),
                "null_values": {k: int(v) for k, v in null_counts.items() if v > 0},
                "duplicates": duplicate_count,
                "memory_mb": round(df.memory_usage(deep=True).sum() / (1024 * 1024), 2),
            }
        return audit


if __name__ == "__main__":
    validator = DataValidator("data/raw")
    results = validator.audit_quality()
    for tbl, info in results.items():
        print(
            f"[{tbl}] Rows: {info['row_count']}, Dups: {info['duplicates']}, Nulls: {info['null_values']}"
        )
