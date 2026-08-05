"""トラッキング結果のJSON保存"""
import json
from typing import Dict

from ..domain.entities import ROI


def write_tracking_json(path: str, tracking_results: Dict[int, ROI]) -> None:
    """フレーム番号→矩形座標のマッピングをJSONに保存する"""
    data = {
        str(frame_no): {"x": roi.x, "y": roi.y, "w": roi.w, "h": roi.h}
        for frame_no, roi in tracking_results.items()
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
