"""ドメインエンティティ"""
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ROI:
    """OCR対象の矩形領域（動画座標系）"""
    x: int
    y: int
    w: int
    h: int


@dataclass
class FrameResult:
    """1フレームのOCR結果"""
    frame_no: int
    value: Optional[int] = None
    edited: bool = False
