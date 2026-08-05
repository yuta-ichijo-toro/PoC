"""OCR結果のCSVエクスポート"""
import csv
from typing import List

from ..domain.entities import FrameResult


def write_csv(path: str, results: List[FrameResult]) -> None:
    """frame,value 形式でCSVに保存する"""
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["frame", "value"])
        for r in results:
            writer.writerow([r.frame_no, r.value])
