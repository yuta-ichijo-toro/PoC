"""OpenCVを使った動画読み込み（スレッドセーフ）"""
import threading
from typing import Optional

import cv2
import numpy as np


class VideoReader:
    def __init__(self, path: str) -> None:
        self._cap = cv2.VideoCapture(path)
        if not self._cap.isOpened():
            raise ValueError(f"動画ファイルを開けませんでした: {path}")
        self._lock = threading.Lock()
        self._frame_count = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self._fps = self._cap.get(cv2.CAP_PROP_FPS)
        self._width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self._height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    @property
    def frame_count(self) -> int:
        return self._frame_count

    @property
    def fps(self) -> float:
        return self._fps

    @property
    def width(self) -> int:
        return self._width

    @property
    def height(self) -> int:
        return self._height

    def read_frame(self, frame_no: int) -> Optional[np.ndarray]:
        with self._lock:
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
            ret, frame = self._cap.read()
        if not ret or frame is None:
            return None
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    def seek(self, frame_no: int) -> None:
        """指定フレームにシークする（シーケンシャル読み込みの開始位置設定に使用）"""
        with self._lock:
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)

    def read_next_frame(self) -> Optional[np.ndarray]:
        """現在位置から次のフレームを順次読み込む（シーク不要で高速）"""
        with self._lock:
            ret, frame = self._cap.read()
        if not ret or frame is None:
            return None
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    def release(self) -> None:
        with self._lock:
            self._cap.release()

    def __del__(self) -> None:
        try:
            self.release()
        except Exception:
            pass
