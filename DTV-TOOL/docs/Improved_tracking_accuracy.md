# トラッキング精度改善案

## 1. 現状の問題

### 症状
- トラッキングが一度外れると以降のフレームでも回復しない
- 選択した初期ROIの位置から大きく離れた位置を追跡し続ける

### 根本原因

#### `_update_tracking`（`src/domain/use_cases.py`）

```python
def _update_tracking(self, frame_no: int, frame: np.ndarray, roi: ROI) -> ROI:
    if frame_no == 0:
        self._tracker.init(frame, roi)
        return roi
    tracked = self._tracker.update(frame)
    return tracked if tracked is not None else roi  # ← fallbackのみ、再初期化なし
```

- `tracker.update()` が失敗すると **返り値は初期ROIに差し替えられる** が、トラッカーオブジェクト自体は再初期化されない
- 失敗後のトラッカーは内部状態が壊れたまま次フレームでも `update()` を呼び続けるため、**以降も永続的に失敗する**

#### `OpenCVTracker.update()`（`src/infrastructure/tracker.py`）

```python
def update(self, frame_rgb: np.ndarray) -> Optional[ROI]:
    success, bbox = self._tracker.update(bgr)
    if not success:
        return None   # 失敗通知のみ。再初期化・リカバリロジックなし
```

- 失敗を`None`で通知するだけで、回復処理がない

---

## 2. 改善案

### 案A（推奨）: 失敗時にトラッカーを再初期化する

**方針**: `tracker.update()` が失敗した時点で、フォールバックROI（初期ROIまたは直近成功ROI）でトラッカーを即時再初期化する。次フレームからトラッキングを再開できるようになる。

#### 実装箇所: `src/domain/use_cases.py` `_update_tracking`

```python
def _update_tracking(self, frame_no: int, frame: np.ndarray, roi: ROI) -> ROI:
    if self._tracker is None:
        return roi
    if frame_no == 0:
        self._tracker.init(frame, roi)
        return roi

    tracked = self._tracker.update(frame)
    if tracked is not None:
        self._last_good_roi = tracked  # 直近成功ROIを保持
        return tracked

    # 失敗時: フォールバックROIで再初期化して次フレームから回復を試みる
    fallback = getattr(self, '_last_good_roi', roi)
    self._tracker.init(frame, fallback)
    return fallback
```

**効果**: 失敗した次のフレームからトラッキングが再開され、「一度外れたら戻らない」問題を解消できる。

---

### 案B: 位置ずれ検出による強制再初期化（ドリフト防止）

**方針**: トラッキングが成功していても、追跡ROIの中心が初期ROIから許容距離以上離れた場合を「誤追跡」とみなして再初期化する。選択した範囲に近い位置にトラッキングを留める。

#### 実装箇所: `src/domain/use_cases.py` `_update_tracking`

```python
def _update_tracking(self, frame_no: int, frame: np.ndarray, roi: ROI) -> ROI:
    if self._tracker is None:
        return roi
    if frame_no == 0:
        self._tracker.init(frame, roi)
        self._initial_roi = roi
        return roi

    tracked = self._tracker.update(frame)

    if tracked is not None:
        # 初期ROI中心からの距離を計算
        init_cx = self._initial_roi.x + self._initial_roi.w / 2
        init_cy = self._initial_roi.y + self._initial_roi.h / 2
        cx = tracked.x + tracked.w / 2
        cy = tracked.y + tracked.h / 2
        max_drift = max(self._initial_roi.w, self._initial_roi.h) * 1.5  # 許容ドリフト係数

        if abs(cx - init_cx) <= max_drift and abs(cy - init_cy) <= max_drift:
            self._last_good_roi = tracked
            return tracked
        # 許容範囲を超えたドリフトは誤追跡として再初期化

    # 失敗 or ドリフト超過: 直近成功ROIで再初期化
    fallback = getattr(self, '_last_good_roi', roi)
    self._tracker.init(frame, fallback)
    return fallback
```

**効果**: 追跡ROIが選択範囲から大きく離れることを防止できる。許容係数（1.5）はパラメータとして外部から調整可能にするとよい。

---

### 案C: テンプレートマッチングによるフォールバック

**方針**: トラッカーが失敗した際、初期フレームから切り出したテンプレート画像を使って `cv2.matchTemplate()` で現フレームを探索し、最も近似する位置をROIとして返す。

#### 実装箇所: `src/infrastructure/tracker.py` に追加

```python
import cv2
import numpy as np
from ..domain.entities import ROI

class OpenCVTracker:
    def init(self, frame_rgb: np.ndarray, roi: ROI) -> bool:
        bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        self._tracker = _create_cv_tracker(self._algorithm)
        # テンプレートとして初期ROIの切り出し画像を保持
        self._template = bgr[roi.y:roi.y+roi.h, roi.x:roi.x+roi.w].copy()
        self._initial_roi = roi
        return bool(self._tracker.init(bgr, (roi.x, roi.y, roi.w, roi.h)))

    def update(self, frame_rgb: np.ndarray) -> Optional[ROI]:
        if self._tracker is None:
            return None
        bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        success, bbox = self._tracker.update(bgr)
        if success:
            x, y, w, h = [max(0, int(v)) for v in bbox]
            return ROI(x=x, y=y, w=w, h=h)

        # フォールバック: テンプレートマッチングで探索
        return self._template_match_fallback(bgr)

    def _template_match_fallback(self, bgr: np.ndarray) -> Optional[ROI]:
        """テンプレートマッチングで初期ROI周辺を探索する"""
        if self._template is None or self._template.size == 0:
            return None
        h_img, w_img = bgr.shape[:2]
        roi = self._initial_roi
        # 探索範囲: 初期ROI周辺 ±50% の余白
        margin_x = int(roi.w * 0.5)
        margin_y = int(roi.h * 0.5)
        x1 = max(0, roi.x - margin_x)
        y1 = max(0, roi.y - margin_y)
        x2 = min(w_img, roi.x + roi.w + margin_x)
        y2 = min(h_img, roi.y + roi.h + margin_y)
        search_area = bgr[y1:y2, x1:x2]

        if search_area.shape[0] < self._template.shape[0] or search_area.shape[1] < self._template.shape[1]:
            return None

        result = cv2.matchTemplate(search_area, self._template, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(result)
        if max_val < 0.5:  # 類似度閾値（調整可）
            return None
        mx, my = max_loc
        return ROI(x=x1 + mx, y=y1 + my, w=roi.w, h=roi.h)
```

**効果**: トラッカー失敗後も初期ROI付近でパターン探索するため、外見変化が少ない場合に有効。テンプレートの劣化には弱いため、案Aと組み合わせるのが望ましい。

---

## 3. 推奨する実装優先順位

| 優先度            | 案                              | 効果                                       | 実装コスト |
| ----------------- | ------------------------------- | ------------------------------------------ | ---------- |
| 1位（必須）       | 案A: 失敗時の再初期化           | 「一度外れると戻らない」問題の直接解消     | 低         |
| 2位（推奨）       | 案B: ドリフト検出               | 誤追跡による位置ずれを抑制                 | 低         |
| 3位（オプション） | 案C: テンプレートマッチング     | 追加のフォールバックで堅牢性向上           | 中         |
| 4位（手振れ対策） | 案D: セグメント分割トラッキング | カメラ移動が大きい動画での根本的な精度改善 | 中         |

### 実装の組み合わせ（最終形）

```
トラッキング更新フロー（フレームNごと）:
  0. セグメント境界チェック（案D）
       セグメント先頭フレームなら → 該当ROIでトラッカーをフル再初期化して以降の処理をスキップ
  1. tracker.update(frame)
  2. 成功かつドリフト許容範囲内 → 追跡ROIを採用（案B）
  3. 成功だがドリフト超過 → 直近成功ROIで再初期化（案A+B）
  4. 失敗 → テンプレートマッチングを試行（案C）
       4a. マッチ成功 → そのROIで再初期化
       4b. マッチ失敗 → 直近成功ROIで再初期化（案A）
```

---

## 4. 案D: セグメント分割トラッキング（手振れ対策）

### 背景と問題

手振れが激しい動画では、案A〜Cによる回復ロジックを適用しても、以下の問題が発生する:

- カメラが大きく移動すると、「直前の成功ROI」自体が既に誤位置を指している
- ドリフト検出（案B）は初期ROIからの累積ずれを検知できず、徐々に位置がずれると無効になる
- テンプレートマッチング（案C）は照明変化・遮蔽・ブラーに弱く、手振れ動画では誤マッチが頻発する

### 方針

動画を一定フレーム数（セグメント）ごとに分割し、各セグメント先頭でトラッカーをそのセグメント用の ROI で**完全再初期化**する。これにより：

- 累積誤差がセグメント境界でリセットされる
- セグメントごとに正しい初期位置を与えられるため、手振れ後の追跡精度が大幅に向上する

### 実装箇所

#### `src/domain/use_cases.py` `ProcessVideoUseCase`

```python
class ProcessVideoUseCase:
    def __init__(
        self,
        ...
        segment_interval: int = 0,           # 0 = 無効, >0 = セグメント間隔（フレーム数）
        segment_rois: Optional[Dict[int, ROI]] = None,  # 開始フレーム → ROI
    ) -> None:
        ...
        self._segment_interval = segment_interval
        self._segment_rois: Dict[int, ROI] = segment_rois or {}

    def _update_tracking(self, frame_no: int, frame: np.ndarray, roi: ROI) -> ROI:
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
                # 明示的に設定されたROI、なければ直前成功ROIを引き継ぐ
                segment_roi = self._segment_rois.get(frame_no) or (
                    self._last_good_roi if self._last_good_roi is not None else roi
                )
            self._tracker.init(frame, segment_roi)  # テンプレートも更新
            self._initial_roi = segment_roi
            self._last_good_roi = segment_roi
            return segment_roi
        # 以降は通常の案A/B/Cフロー
        ...
```

### GUI変更内容（`src/presentation/app.py`）

| 追加要素                                                | 説明                                                                   |
| ------------------------------------------------------- | ---------------------------------------------------------------------- |
| `[案D] セグメント分割トラッキング有効` チェックボックス | 機能のOn/Off切り替え（5段目オプションバー）                            |
| セグメント間隔スピンボックス                            | フレーム数を指定（デフォルト 180）                                     |
| 換算ラベル                                              | 「≈ X秒 / Y分割」を動画のFPSから自動計算して表示                       |
| `セグメントROI設定...` ボタン                           | セグメント一覧ダイアログを開く                                         |
| セグメントROI設定ダイアログ                             | セグメントごとにROIをドラッグ設定・クリアできるTreviewベースの管理画面 |

### 色コード（UI）

| 色                  | 意味                                                       |
| ------------------- | ---------------------------------------------------------- |
| 赤（`#ff4444`）     | メインROI（フレーム0 / セグメント0）                       |
| シアン（`#00cccc`） | セグメント用ROI（設定モード中のドラッグ枠 / オーバーレイ） |
| 青点線（`#44aaff`） | トラッキング結果のリアルタイム表示（既存）                 |

### 設定ファイル（`src/settings/performance_settings.json`）

```json
{
  "segment_tracking_enabled": false,
  "segment_interval": 180
}
```

### 効果と限界

| 項目     | 内容                                                                                   |
| -------- | -------------------------------------------------------------------------------------- |
| 効果     | セグメント先頭でリセットされるため、手振れ後の追跡ずれが蓄積しない                     |
| 追加設定 | セグメントごとのROI設定が必要（未設定時は直前ROI引継ぎ）                               |
| 限界     | セグメント内での手振れには効かない。間隔を短くすればするほどROI設定作業が増える        |
| 推奨設定 | FPS×10〜FPS×30 程度（30FPS で 300〜900フレーム）を目安に動画の手振れ周期に合わせて設定 |

---

## 5. 仕様書への反映が必要な箇所

`docs/spec.md` の 3.3節 に以下を追記する:

- トラッキング失敗時は直前の成功ROIでトラッカーを**再初期化**し、次フレームから回復を試みる
- 追跡ROIの中心が初期ROI中心から一定距離（初期ROI短辺の1.5倍など）を超えた場合は誤追跡とみなし再初期化する（ドリフト防止）
- ドリフト許容係数はパフォーマンス設定ファイル（`src/settings/performance_settings.json`）で調整可能とする
- **セグメント分割トラッキング**（案D）を 3.3.3 節として追記済み
