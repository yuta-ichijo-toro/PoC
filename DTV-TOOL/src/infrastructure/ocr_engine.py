"""EasyOCRを使った数字認識エンジン"""
import re
from typing import Dict, List, Optional

import cv2
import numpy as np


class EasyOCREngine:
    """EasyOCRで数字（および小数点）を認識する"""

    _ALLOWLIST = "0123456789."

    def __init__(
        self,
        use_preprocessing: bool = False,
        use_tuned_params: bool = False,
        use_gpu: bool = False,
    ) -> None:
        """
        Parameters
        ----------
        use_preprocessing : bool
            True のとき recognize() 内でグレースケール変換・拡大・CLAHE・Otsu二値化を実施する。
        use_tuned_params : bool
            True のとき readtext() に精度向上パラメータ（min_size, contrast_ths 等）を渡す。
        use_gpu : bool
            True のとき CUDA が利用可能であれば GPU 推論を使用する。
        """
        import easyocr  # 遅延インポートで起動を速くする
        _gpu = False
        if use_gpu:
            try:
                import torch
                _gpu = torch.cuda.is_available()
            except ImportError:
                pass
        self._reader = easyocr.Reader(["en"], gpu=_gpu)
        self._use_preprocessing = use_preprocessing
        self._use_tuned_params = use_tuned_params
        self._clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    # ------------------------------------------------------------------ #
    # 前処理（案1）
    # ------------------------------------------------------------------ #

    def _preprocess(self, image: np.ndarray) -> np.ndarray:
        """
        OCR前の画像前処理パイプライン。
        グレースケール → 拡大 → CLAHE → Otsu二値化 → RGB変換。
        """
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        # 最短辺が 200px を下回る場合は整数倍で拡大
        # 拡大後の最大辺が 4000px を超えないよう scale を制限しメモリ不足を防ぐ
        h, w = gray.shape
        min_side = min(h, w)
        if min_side > 0 and min_side < 200:
            scale = max(2, int(200 / min_side))
            max_side = max(h, w)
            if max_side > 0:
                scale = min(scale, max(1, 4000 // max_side))
            if scale > 1:
                gray = cv2.resize(
                    gray, (w * scale, h * scale), interpolation=cv2.INTER_CUBIC
                )

        # CLAHE でコントラスト強調
        gray = self._clahe.apply(gray)

        # Otsu 二値化
        _, binary = cv2.threshold(
            gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU
        )

        return cv2.cvtColor(binary, cv2.COLOR_GRAY2RGB)

    # ------------------------------------------------------------------ #
    # 認識
    # ------------------------------------------------------------------ #

    def recognize(self, image: np.ndarray) -> Optional[int]:
        """画像から数字を読み取り整数で返す。検出なしまたは変換不可の場合はNone。"""
        if self._use_preprocessing:
            try:
                image = self._preprocess(image)
            except cv2.error:
                pass  # 前処理失敗時は元画像のまま継続

        # 案2: 調整済みパラメータ
        extra_kwargs: Dict = (
            dict(
                min_size=5,
                contrast_ths=0.05,
                adjust_contrast=0.7,
                text_threshold=0.5,
                low_text=0.3,
                width_ths=0.3,
            )
            if self._use_tuned_params
            else {}
        )

        try:
            results: List = self._reader.readtext(
                image,
                allowlist=self._ALLOWLIST,
                detail=1,
                **extra_kwargs,
            )
        except cv2.error:
            return None
        if not results:
            return None
        # 信頼度が最も高いテキストを採用
        best = max(results, key=lambda r: r[2])
        text = best[1]
        # 末尾の . を削除し、数字以外の文字を除去する
        text = text.rstrip(".")
        text = re.sub(r"[^0-9]", "", text)
        if not text:
            return None
        return int(text)
