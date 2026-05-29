"""Shared utilities: config loading, logging, checkpointing."""

import json
import logging
import sys
import yaml
from pathlib import Path


def load_config(yaml_path: str | Path) -> dict:
    with open(yaml_path) as f:
        return yaml.safe_load(f)


def setup_logging(name: str = "srg", level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s",
                                           datefmt="%H:%M:%S"))
    logger.addHandler(handler)
    logger.setLevel(level)
    return logger


# ── JSONL checkpoint helpers ───────────────────────────────────────────────────

def load_done_ids(jsonl_path: Path) -> set[str]:
    """Return set of file_ids already written to a JSONL checkpoint."""
    done = set()
    if not jsonl_path.exists():
        return done
    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                done.add(rec["file_id"])
            except Exception:
                pass
    return done


def append_result(jsonl_path: Path, record: dict) -> None:
    """Append one JSON record to a JSONL file."""
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    with open(jsonl_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_results(jsonl_path: Path) -> list[dict]:
    """Load all records from a JSONL file."""
    if not jsonl_path.exists():
        return []
    records = []
    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except Exception:
                    pass
    return records
