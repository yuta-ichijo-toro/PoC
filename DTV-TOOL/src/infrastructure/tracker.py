"""OpenCVを使ったROIトラッキング"""
from typing import Optional

import cv2
import numpy as np

from ..domain.entities import ROI


def _create_cv_tracker(algorithm: str):
    """バージョン差異を吸収してOpenCVトラッカーを生成する"""
    _factories = {
        "KCF": [
            lambda: cv2.TrackerKCF_create(),
            lambda: cv2.legacy.TrackerKCF_create(),
        ],
        "CSRT": [
            lambda: cv2.TrackerCSRT_create(),
            lambda: cv2.legacy.TrackerCSRT_create(),
        ],
    }
    for factory in _factories.get(algorithm, _factories["CSRT"]):
        try:
            return factory()
        except AttributeError:
            continue
    raise RuntimeError(
        f"OpenCVトラッカー({algorithm})の生成に失敗しました。"
        " opencv-contrib-python をインストールしてください。"
    )


class OpenCVTracker:
    """OpenCV TrackerCSRT/KCFによるROIトラッキング（テンプレートマッチングフォールバック付き）"""

    def __init__(
        self,
        algorithm: str = "CSRT",
        use_template_matching: bool = False,
        template_threshold: float = 0.5,
        template_margin_factor: float = 0.5,
    ) -> None:
        self._algorithm = algorithm
        self._use_template_matching = use_template_matching
        self._template_threshold = template_threshold
        self._template_margin_factor = template_margin_factor
        self._tracker = None
        self._template: Optional[np.ndarray] = None
        self._initial_roi: Optional[ROI] = None

    def init(self, frame_rgb: np.ndarray, roi: ROI) -> bool:
        """RGBフレームでトラッカーを初期化する"""
        bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        self._tracker = _create_cv_tracker(self._algorithm)
        if self._use_template_matching:
            # テンプレートは初回 init のみ保存する（reinit では更新しない）
            self._template = bgr[roi.y:roi.y + roi.h, roi.x:roi.x + roi.w].copy()
            self._initial_roi = roi
        return bool(self._tracker.init(bgr, (roi.x, roi.y, roi.w, roi.h)))

    def reinit(self, frame_rgb: np.ndarray, roi: ROI) -> bool:
        """テンプレートを保持したままトラッカーを再初期化する（案A/B用）"""
        bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        self._tracker = _create_cv_tracker(self._algorithm)
        return bool(self._tracker.init(bgr, (roi.x, roi.y, roi.w, roi.h)))

    def update(self, frame_rgb: np.ndarray) -> Optional[ROI]:
        """RGBフレームを渡してトラッキング結果を返す。失敗時はNone（案Cで回復した場合はROI）。"""
        if self._tracker is None:
            return None
        bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        success, bbox = self._tracker.update(bgr)
        if success:
            x, y, w, h = [max(0, int(v)) for v in bbox]
            return ROI(x=x, y=y, w=w, h=h)

        # 案C: テンプレートマッチングによるフォールバック
        if self._use_template_matching:
            matched = self._template_match_fallback(bgr)
            if matched is not None:
                # マッチ成功: そのROIでトラッカーを再初期化し次フレームから追跡再開
                self._tracker = _create_cv_tracker(self._algorithm)
                self._tracker.init(bgr, (matched.x, matched.y, matched.w, matched.h))
                return matched
        return None

    def _template_match_fallback(self, bgr: np.ndarray) -> Optional[ROI]:
        """テンプレートマッチングで初期ROI周辺を探索し、一致位置を返す"""
        if self._template is None or self._initial_roi is None or self._template.size == 0:
            return None
        h_img, w_img = bgr.shape[:2]
        roi = self._initial_roi
        margin_x = int(roi.w * self._template_margin_factor)
        margin_y = int(roi.h * self._template_margin_factor)
        x1 = max(0, roi.x - margin_x)
        y1 = max(0, roi.y - margin_y)
        x2 = min(w_img, roi.x + roi.w + margin_x)
        y2 = min(h_img, roi.y + roi.h + margin_y)
        search_area = bgr[y1:y2, x1:x2]

        if search_area.shape[0] < self._template.shape[0] or search_area.shape[1] < self._template.shape[1]:
            return None

        result = cv2.matchTemplate(search_area, self._template, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(result)
        if max_val < self._template_threshold:
            return None
        mx, my = max_loc
        return ROI(x=x1 + mx, y=y1 + my, w=roi.w, h=roi.h)
