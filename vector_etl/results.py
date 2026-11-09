"""Result types for a batched, partial-success-aware ETL run."""

import json
from dataclasses import dataclass, field, asdict
from typing import List, Dict


@dataclass
class SkippedBatch:
    start_row: int
    end_row: int
    error: str
    traceback: str

    def to_dict(self):
        return asdict(self)


@dataclass
class ETLRunResult:
    status: str  # "success" | "partial_success" | "failed" | "no_data"
    total_rows: int
    success_count: int
    failed_count: int
    skipped_batches: List[SkippedBatch] = field(default_factory=list)
    retries: Dict[str, int] = field(default_factory=dict)

    def to_dict(self):
        return {
            "status": self.status,
            "total_rows": self.total_rows,
            "success_count": self.success_count,
            "failed_count": self.failed_count,
            "skipped_batches": [b.to_dict() for b in self.skipped_batches],
            "retries": dict(self.retries),
        }

    def write_report(self, path):
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh, indent=2, sort_keys=True)
        return path


def derive_status(success_count, failed_count):
    if success_count == 0 and failed_count == 0:
        return "no_data"
    if failed_count == 0:
        return "success"
    if success_count == 0:
        return "failed"
    return "partial_success"
