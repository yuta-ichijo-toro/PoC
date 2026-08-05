"""ProcessPoolExecutor 用 OCR ワーカー関数（モジュールトップレベルで定義）"""
# このモジュールの関数はサブプロセスから pickle 経由で呼び出されるため、
# クラスやネスト関数ではなくモジュールレベルの関数として定義する必要がある。
from typing import Optional, Tuple

import numpy as np

_worker_engine = None


def init_ocr_worker(
    project_root: str,
    use_preprocessing: bool,
    use_tuned_params: bool,
    use_gpu: bool,
) -> None:
    """各ワーカープロセスで EasyOCR エンジンを初期化する（プロセス起動時に 1 回だけ実行される）"""
    import sys
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    global _worker_engine
    from src.infrastructure.ocr_engine import EasyOCREngine  # noqa: PLC0415
    _worker_engine = EasyOCREngine(
        use_preprocessing=use_preprocessing,
        use_tuned_params=use_tuned_params,
        use_gpu=use_gpu,
    )


def run_ocr_task(item: Tuple[int, Optional[np.ndarray]]) -> Tuple[int, Optional[int]]:
    """クロップ画像に対して OCR を実行しフレーム番号と結果を返す"""
    frame_no, crop = item
    if crop is None:
        return frame_no, None
    try:
        return frame_no, _worker_engine.recognize(crop)
    except Exception:
        return frame_no, None
