# OCR精度向上 修正案

## 現状の問題点

現在の実装（`src/infrastructure/ocr_engine.py`）は以下の構成になっている。

- EasyOCR を `allowlist="0123456789."` で実行
- `gpu=False`（CPUのみ）
- 結果のうち **信頼度スコアが最大のもの1件** だけを採用
- **画像前処理なし**（生のクロップ画像をそのまま投入）
- 後処理は末尾 `.` 除去と非数字文字の除去のみ

---

## 修正案一覧

### 案1: OCR実行前の画像前処理を追加する（優先度: 高）

**概要**  
クロップ画像を OCR に渡す前に OpenCV で前処理を行い、文字を際立たせる。

**実装箇所**: `ocr_engine.py` の `recognize()` 内、`readtext()` 呼び出し前

**推奨処理パイプライン例**:

```python
import cv2

def _preprocess(self, image: np.ndarray) -> np.ndarray:
    # 1. グレースケール変換
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    # 2. 画像を2〜4倍に拡大（小さいROIほど効果大）
    h, w = gray.shape
    scale = max(1, 200 // min(h, w))  # 最短辺が200px以上になるよう拡大
    gray = cv2.resize(gray, (w * scale, h * scale), interpolation=cv2.INTER_CUBIC)
    # 3. CLAHE でコントラスト強調
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    # 4. Otsu 二値化（背景が明るい場合は THRESH_BINARY_INV に変更）
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    # 5. RGBに戻す（EasyOCRはRGB入力を受け付ける）
    return cv2.cvtColor(binary, cv2.COLOR_GRAY2RGB)
```

**期待効果**: 低解像度・低コントラストROIでの認識率が大幅に向上。

---

### 案2: EasyOCR のパラメータを調整する（優先度: 高）

**概要**  
`readtext()` に追加パラメータを渡して検出精度を改善する。

**実装箇所**: `ocr_engine.py` の `readtext()` 呼び出し箇所

| パラメータ        | 現在              | 推奨値  | 効果                              |
| ----------------- | ----------------- | ------- | --------------------------------- |
| `min_size`        | デフォルト(10)    | `5`     | 小さい文字も検出                  |
| `contrast_ths`    | デフォルト(0.1)   | `0.05`  | 低コントラスト文字の検出強化      |
| `adjust_contrast` | デフォルト(0.5)   | `0.7`   | EasyOCR内部のコントラスト調整強化 |
| `text_threshold`  | デフォルト(0.7)   | `0.5`   | テキスト判定の閾値を下げる        |
| `low_text`        | デフォルト(0.4)   | `0.3`   | 低確信度テキストも拾う            |
| `width_ths`       | デフォルト(0.5)   | `0.3`   | 隣接テキストの結合を抑制          |
| `paragraph`       | デフォルト(False) | `False` | 数字は1ブロックで十分             |

```python
results = self._reader.readtext(
    image,
    allowlist=self._ALLOWLIST,
    detail=1,
    min_size=5,
    contrast_ths=0.05,
    adjust_contrast=0.7,
    text_threshold=0.5,
    low_text=0.3,
    width_ths=0.3,
)
```

**期待効果**: 見逃し（False Negative）の減少。

---

### 案3: 信頼度スコアによるフィルタリングを追加する（優先度: 中）

**概要**  
信頼度が低すぎる結果を `None` として捨て、誤認識（False Positive）を減らす。

**実装箇所**: `ocr_engine.py` の `recognize()` 内

```python
CONFIDENCE_THRESHOLD = 0.5  # チューニング可能

best = max(results, key=lambda r: r[2])
if best[2] < CONFIDENCE_THRESHOLD:
    return None  # 信頼度不足は無効扱い
```

**期待効果**: ノイズ・背景模様を誤認識したケースを除去できる。

---

### 案4: 時系列スムージング（外れ値除去）を追加する（優先度: 中）

**概要**  
前後フレームの OCR 結果を参照し、大きく外れた値を補間または `None` に置き換える。ユースケース層で実装する。

**実装箇所**: `src/domain/use_cases.py` の `execute()` 内、全フレーム処理後に後処理として追加

```python
def _smooth_results(
    self, results: List[FrameResult], window: int = 3
) -> List[FrameResult]:
    """前後 window フレームの中央値から大きく外れた値を None に置換"""
    values = [r.value for r in results]
    smoothed = []
    for i, v in enumerate(values):
        if v is None:
            smoothed.append(results[i])
            continue
        neighbors = [
            values[j]
            for j in range(max(0, i - window), min(len(values), i + window + 1))
            if values[j] is not None
        ]
        if not neighbors:
            smoothed.append(results[i])
            continue
        median = sorted(neighbors)[len(neighbors) // 2]
        # 中央値との差が閾値を超えたら外れ値とみなす
        if abs(v - median) > median * 0.3:
            smoothed.append(FrameResult(frame_no=results[i].frame_no, value=None))
        else:
            smoothed.append(results[i])
    return smoothed
```

**期待効果**: 一時的な誤認識（例: `1234` → `12B4`）によるスパイクを除去。

---

### 案5: ROIにパディングを加える（優先度: 低）

**概要**  
ユーザーが選択したROIの周囲に数ピクセルのパディングを加え、文字が端で切れるケースを防ぐ。

**実装箇所**: `src/domain/use_cases.py` の `execute()` 内、crop 計算部分

```python
PADDING = 4  # px

x1 = max(0, current_roi.x - PADDING)
y1 = max(0, current_roi.y - PADDING)
x2 = min(w, current_roi.x + current_roi.w + PADDING)
y2 = min(h, current_roi.y + current_roi.h + PADDING)
```

**期待効果**: ROI境界付近の文字欠損を防ぎ、認識率が向上。

---

### 案6: 認識エンジンを Tesseract に切り替えるオプションを追加する（優先度: 低）

**概要**  
EasyOCR と Tesseract を差し替え可能な設計にし、動画によって使い分けられるようにする。Tesseract は `--psm 7`（1行）＋ `digits` モードで数字専用認識が高速・高精度になるケースがある。

**実装箇所**: `ocr_engine.py` に `TesseractOCREngine` クラスを追加し、`app.py` でエンジン選択UIを設ける

```python
import pytesseract

class TesseractOCREngine:
    def recognize(self, image: np.ndarray) -> Optional[int]:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        config = r"--psm 7 -c tessedit_char_whitelist=0123456789"
        text = pytesseract.image_to_string(gray, config=config).strip()
        text = re.sub(r"[^0-9]", "", text)
        return int(text) if text else None
```

**期待効果**: 特定の動画（高コントラスト・大きい文字）では Tesseract の方が高速かつ高精度。

---

## 推奨する優先実装順

| 優先度 | 案                          | 実装コスト | 期待効果     |
| ------ | --------------------------- | ---------- | ------------ |
| 1      | 案1: 画像前処理             | 低         | 大           |
| 2      | 案2: EasyOCR パラメータ調整 | 低         | 中〜大       |
| 3      | 案3: 信頼度フィルタリング   | 低         | 中           |
| 4      | 案4: 時系列スムージング     | 中         | 中           |
| 5      | 案5: ROIパディング          | 低         | 小〜中       |
| 6      | 案6: Tesseract 切り替え     | 高         | ケースによる |

案1〜3 は `ocr_engine.py` のみの変更で済み、他モジュールへの影響がないため最初に試すことを推奨。
