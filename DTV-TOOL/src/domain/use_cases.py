"""ユースケース層"""
import concurrent.futures
import os
import pathlib
from collections import Counter
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from .entities import ROI, FrameResult

# プロセス並列時に子プロセスへ渡すプロジェクトルートパス
_PROJ_ROOT = str(pathlib.Path(__file__).resolve().parent.parent.parent)


class ProcessVideoUseCase:
    """動画全フレームに対してOCR＋トラッキングを実行し、結果リストとトラッキング結果を返す"""

    def __init__(
        self,
        video_reader,
        ocr_engine,
        tracker=None,
        tracking_reinit_on_failure: bool = True,
        tracking_drift_detection: bool = False,
        tracking_drift_factor: float = 1.5,
        segment_interval: int = 0,
        segment_rois: Optional[Dict[int, ROI]] = None,
    ) -> None:
        self._video_reader = video_reader
        self._ocr_engine = ocr_engine
        self._tracker = tracker
        self._tracking_reinit_on_failure = tracking_reinit_on_failure
        self._tracking_drift_detection = tracking_drift_detection
        self._tracking_drift_factor = tracking_drift_factor
        self._segment_interval = segment_interval
        self._segment_rois: Dict[int, ROI] = segment_rois or {}
        self._last_good_roi: Optional[ROI] = None
        self._initial_roi: Optional[ROI] = None

    def execute(
        self,
        roi: ROI,
        ocr_sub_roi: Optional[ROI] = None,
        progress_callback: Optional[Callable[[int, int], None]] = None,
        use_parallel: bool = False,
        use_process_parallel: bool = False,
        max_workers: Optional[int] = None,
        batch_size: Optional[int] = None,
        frame_skip: int = 1,
        use_sequential_read: bool = False,
        ocr_engine_params: Optional[Dict] = None,
    ) -> Tuple[List[FrameResult], Dict[int, ROI]]:
        if use_process_parallel:
            return self._execute_process_parallel(
                roi, ocr_sub_roi, progress_callback, max_workers, batch_size,
                frame_skip, use_sequential_read, ocr_engine_params or {},
            )
        if use_parallel:
            return self._execute_parallel(
                roi, ocr_sub_roi, progress_callback, max_workers, batch_size,
                frame_skip, use_sequential_read,
            )
        return self._execute_sequential(roi, ocr_sub_roi, progress_callback, frame_skip, use_sequential_read)

    # ------------------------------------------------------------------ #
    # 順次処理
    # ------------------------------------------------------------------ #

    def _execute_sequential(
        self,
        roi: ROI,
        ocr_sub_roi: Optional[ROI],
        progress_callback: Optional[Callable[[int, int], None]],
        frame_skip: int,
        use_sequential_read: bool,
    ) -> Tuple[List[FrameResult], Dict[int, ROI]]:
        results: List[FrameResult] = []
        tracking_results: Dict[int, ROI] = {}
        total = self._video_reader.frame_count
        last_value: Optional[int] = None

        if use_sequential_read:
            self._video_reader.seek(0)

        for i in range(total):
            if use_sequential_read:
                frame = self._video_reader.read_next_frame()
            else:
                frame = self._video_reader.read_frame(i)

            if frame is None:
                results.append(FrameResult(frame_no=i, value=None))
                tracking_results[i] = roi
            else:
                current_roi = self._update_tracking(i, frame, roi)
                tracking_results[i] = current_roi
                if i % frame_skip == 0:
                    crop = self._extract_crop(frame, current_roi, ocr_sub_roi)
                    last_value = self._ocr_engine.recognize(crop) if crop.size > 0 else None
                results.append(FrameResult(frame_no=i, value=last_value))

            if progress_callback:
                progress_callback(i + 1, total)

        return results, tracking_results

    # ------------------------------------------------------------------ #
    # 並列処理
    # ------------------------------------------------------------------ #

    def _execute_parallel(
        self,
        roi: ROI,
        ocr_sub_roi: Optional[ROI],
        progress_callback: Optional[Callable[[int, int], None]],
        max_workers: Optional[int],
        batch_size: Optional[int],
        frame_skip: int,
        use_sequential_read: bool,
    ) -> Tuple[List[FrameResult], Dict[int, ROI]]:
        total = self._video_reader.frame_count
        tracking_results: Dict[int, ROI] = {}
        results: List[Optional[FrameResult]] = [None] * total
        done_count = 0  # OCR完了フレーム累計
        skip_indices: set = set()  # フレームスキップ対象フレーム番号集合

        _max_workers = max_workers if max_workers is not None else os.cpu_count()
        # バッチサイズ: スレッド数×4 を基準に設定し、一度にメモリへ保持するクロップ数を制限する
        _batch_size = batch_size if batch_size is not None else max(1, (_max_workers or 1) * 4)
        # progress_callback の done/total スケール: 前半(0..total)=トラッキング、後半(total..total*2)=OCR
        _prog_total = total * 2

        def _ocr_task(item: Tuple[int, Optional[np.ndarray]]) -> Tuple[int, Optional[int]]:
            frame_no, crop = item
            if crop is None:
                return frame_no, None
            try:
                return frame_no, self._ocr_engine.recognize(crop)
            except Exception:
                return frame_no, None

        if use_sequential_read:
            self._video_reader.seek(0)

        with concurrent.futures.ThreadPoolExecutor(max_workers=_max_workers) as executor:
            for batch_start in range(0, total, _batch_size):
                batch_end = min(batch_start + _batch_size, total)
                crops: List[Tuple[int, Optional[np.ndarray]]] = []

                # フェーズ1: このバッチのトラッキング（順次）＋切り出し
                for i in range(batch_start, batch_end):
                    if use_sequential_read:
                        frame = self._video_reader.read_next_frame()
                    else:
                        frame = self._video_reader.read_frame(i)

                    if frame is None:
                        tracking_results[i] = roi
                        crops.append((i, None))
                    else:
                        current_roi = self._update_tracking(i, frame, roi)
                        tracking_results[i] = current_roi
                        if i % frame_skip == 0:
                            crop = self._extract_crop(frame, current_roi, ocr_sub_roi)
                            # copy() でフレームへの参照を切り、フレームのメモリをバッチ終了後に解放する
                            crops.append((i, crop.copy() if crop.size > 0 else None))
                        else:
                            skip_indices.add(i)
                            crops.append((i, None))

                    if progress_callback:
                        progress_callback(i + 1, _prog_total)

                # フェーズ2: このバッチのOCR並列実行
                future_map = {executor.submit(_ocr_task, item): item[0] for item in crops}
                for future in concurrent.futures.as_completed(future_map):
                    frame_no, value = future.result()
                    results[frame_no] = FrameResult(frame_no=frame_no, value=value)
                    done_count += 1
                    if progress_callback:
                        progress_callback(total + done_count, _prog_total)

        # スキップフレームを直前の非スキップフレームの値で補完
        if frame_skip > 1:
            last_value: Optional[int] = None
            for r in results:
                if r is None:
                    continue
                if r.frame_no not in skip_indices:
                    last_value = r.value
                elif last_value is not None:
                    r.value = last_value

        return results, tracking_results  # type: ignore[return-value]

    # ------------------------------------------------------------------ #
    # プロセスプール並列処理（GIL回避）
    # ------------------------------------------------------------------ #

    def _execute_process_parallel(
        self,
        roi: ROI,
        ocr_sub_roi: Optional[ROI],
        progress_callback: Optional[Callable[[int, int], None]],
        max_workers: Optional[int],
        batch_size: Optional[int],
        frame_skip: int,
        use_sequential_read: bool,
        ocr_engine_params: Dict,
    ) -> Tuple[List[FrameResult], Dict[int, ROI]]:
        from ..infrastructure.ocr_worker import init_ocr_worker, run_ocr_task  # noqa: PLC0415

        total = self._video_reader.frame_count
        tracking_results: Dict[int, ROI] = {}
        results: List[Optional[FrameResult]] = [None] * total
        done_count = 0
        skip_indices: set = set()

        _max_workers = max_workers if max_workers is not None else os.cpu_count()
        _batch_size = batch_size if batch_size is not None else max(1, (_max_workers or 1) * 4)
        _prog_total = total * 2

        initargs = (
            _PROJ_ROOT,
            ocr_engine_params.get("use_preprocessing", False),
            ocr_engine_params.get("use_tuned_params", False),
            ocr_engine_params.get("use_gpu", False),
        )

        if use_sequential_read:
            self._video_reader.seek(0)

        with concurrent.futures.ProcessPoolExecutor(
            max_workers=_max_workers,
            initializer=init_ocr_worker,
            initargs=initargs,
        ) as executor:
            for batch_start in range(0, total, _batch_size):
                batch_end = min(batch_start + _batch_size, total)
                crops: List[Tuple[int, Optional[np.ndarray]]] = []

                # フェーズ1: トラッキング（順次）＋クロップ収集
                for i in range(batch_start, batch_end):
                    if use_sequential_read:
                        frame = self._video_reader.read_next_frame()
                    else:
                        frame = self._video_reader.read_frame(i)

                    if frame is None:
                        tracking_results[i] = roi
                        crops.append((i, None))
                    else:
                        current_roi = self._update_tracking(i, frame, roi)
                        tracking_results[i] = current_roi
                        if i % frame_skip == 0:
                            crop = self._extract_crop(frame, current_roi, ocr_sub_roi)
                            crops.append((i, crop.copy() if crop.size > 0 else None))
                        else:
                            skip_indices.add(i)
                            crops.append((i, None))

                    if progress_callback:
                        progress_callback(i + 1, _prog_total)

                # フェーズ2: OCR をワーカープロセスに投入（真の並列実行）
                future_map = {executor.submit(run_ocr_task, item): item[0] for item in crops}
                for future in concurrent.futures.as_completed(future_map):
                    frame_no, value = future.result()
                    results[frame_no] = FrameResult(frame_no=frame_no, value=value)
                    done_count += 1
                    if progress_callback:
                        progress_callback(total + done_count, _prog_total)

        if frame_skip > 1:
            last_value: Optional[int] = None
            for r in results:
                if r is None:
                    continue
                if r.frame_no not in skip_indices:
                    last_value = r.value
                elif last_value is not None:
                    r.value = last_value

        return results, tracking_results  # type: ignore[return-value]

    # ------------------------------------------------------------------ #
    # 共通ヘルパー
    # ------------------------------------------------------------------ #

    def _update_tracking(self, frame_no: int, frame: np.ndarray, roi: ROI) -> ROI:
        """トラッキングを更新し現在フレームのROIを返す。"""
        if self._tracker is None:
            return roi

        # セグメント境界での再初期化（frame_no==0 を含む）
        is_segment_start = frame_no == 0 or (
            self._segment_interval > 0 and frame_no % self._segment_interval == 0
        )
        if is_segment_start:
            if frame_no == 0:
                segment_roi = roi
            else:
                segment_roi = self._segment_rois.get(frame_no) or (
                    self._last_good_roi if self._last_good_roi is not None else roi
                )
            self._tracker.init(frame, segment_roi)
            self._initial_roi = segment_roi
            self._last_good_roi = segment_roi
            return segment_roi

        tracked = self._tracker.update(frame)  # 案Cはtracker内部で処理

        if tracked is not None:
            # 案B: ドリフト検出 — 初期ROI中心から許容距離を超えた場合は誤追跡と判定
            if self._tracking_drift_detection and self._initial_roi is not None:
                init_cx = self._initial_roi.x + self._initial_roi.w / 2
                init_cy = self._initial_roi.y + self._initial_roi.h / 2
                cx = tracked.x + tracked.w / 2
                cy = tracked.y + tracked.h / 2
                max_drift = max(self._initial_roi.w, self._initial_roi.h) * self._tracking_drift_factor
                if abs(cx - init_cx) > max_drift or abs(cy - init_cy) > max_drift:
                    fallback = self._last_good_roi if self._last_good_roi is not None else roi
                    if self._tracking_reinit_on_failure:
                        self._tracker.reinit(frame, fallback)
                    return fallback
            self._last_good_roi = tracked
            return tracked

        # 案A: tracker.update() 失敗時 — フォールバックROIで再初期化して次フレームから回復
        fallback = self._last_good_roi if self._last_good_roi is not None else roi
        if self._tracking_reinit_on_failure:
            self._tracker.reinit(frame, fallback)
        return fallback

    def _extract_crop(
        self, frame: np.ndarray, current_roi: ROI, ocr_sub_roi: Optional[ROI]
    ) -> np.ndarray:
        """フレームからOCR対象領域を切り出す。"""
        h, w = frame.shape[:2]
        if ocr_sub_roi is not None:
            ox = current_roi.x + ocr_sub_roi.x
            oy = current_roi.y + ocr_sub_roi.y
            x1 = max(0, ox)
            y1 = max(0, oy)
            x2 = min(w, ox + ocr_sub_roi.w)
            y2 = min(h, oy + ocr_sub_roi.h)
        else:
            x1 = max(0, current_roi.x)
            y1 = max(0, current_roi.y)
            x2 = min(w, current_roi.x + current_roi.w)
            y2 = min(h, current_roi.y + current_roi.h)
        return frame[y1:y2, x1:x2]


class AnalyzeUseCase:
    """OCR結果とGPSログCSVを突き合わせて解析を行う"""

    def execute(
        self,
        ocr_results: List[FrameResult],
        fps: float,
        start_timestamp: float,
        gps_rows: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """gps_rows の各行に ocr_result カラムを付加して返す。

        各行の timestamp 秒に対応するフレーム群のうち、最頻値を採用する。
        """
        result_rows: List[Dict[str, Any]] = []
        for row in gps_rows:
            ts = float(row["timestamp"])
            matching = [
                r.value
                for r in ocr_results
                if r.value is not None
                and start_timestamp + r.frame_no / fps >= ts
                and start_timestamp + r.frame_no / fps < ts + 1
            ]
            ocr_result: Optional[str] = (
                str(Counter(matching).most_common(1)[0][0]) if matching else None
            )
            out = dict(row)
            out["ocr_result"] = ocr_result
            result_rows.append(out)
        return result_rows

