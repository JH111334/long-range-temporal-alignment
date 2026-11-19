"""Dataset and Data Handling for TickTack v3 Baseline on TempUN Small."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset


class TempUNTrainDataset(Dataset):
    """Dataset for Stage-wise Continual Pretraining on TempUN train_nwp.jsonl."""

    def __init__(
        self,
        records: List[Dict[str, Any]],
        tokenizer: Any,
        max_seq_len: int = 256,
    ) -> None:
        self.records = records
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len

    @classmethod
    def from_file(
        cls,
        path: Union[str, Path],
        tokenizer: Any,
        stage_id: Optional[str] = None,
        max_seq_len: int = 256,
    ) -> "TempUNTrainDataset":
        records = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rec = json.loads(line)
                    if stage_id is None or rec.get("stage_id") == stage_id:
                        records.append(rec)
        return cls(records, tokenizer, max_seq_len=max_seq_len)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        rec = self.records[idx]
        text = rec.get("input", "") or rec.get("text", "")
        year = rec.get("year", 1947)

        # Fallback year extraction if not explicit
        if year is None:
            cond = rec.get("conditioning", {})
            year = cond.get("effective_year", 1947)

        tokens = self.tokenizer(
            text,
            max_length=self.max_seq_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        input_ids = tokens["input_ids"].squeeze(0)
        attention_mask = tokens["attention_mask"].squeeze(0)

        # Labels for Next-Token Prediction (ignore pad tokens)
        labels = input_ids.clone()
        labels[attention_mask == 0] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "year": int(year),
            "stage_id": rec.get("stage_id", ""),
            "example_id": rec.get("example_id", f"sample_{idx}"),
        }


def collate_ticktack_batch(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Collate batch elements into padded tensors."""
    input_ids = torch.stack([item["input_ids"] for item in batch])
    attention_mask = torch.stack([item["attention_mask"] for item in batch])
    labels = torch.stack([item["labels"] for item in batch])
    years = [item["year"] for item in batch]
    year_tensor = torch.tensor(years, dtype=torch.long)
    stage_ids = [item["stage_id"] for item in batch]

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "years": years,
        "year_tensor": year_tensor,
        "stage_ids": stage_ids,
    }
