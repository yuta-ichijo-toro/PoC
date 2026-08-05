# 処理速度向上 修正案

## 現状の処理フローと課題

現在の実装（`src/` 配下）における処理フローは以下の通り。

```
動画読み込み → フレームごとに順次処理
  ├─ フレーム取得 (VideoReader.read_frame)  ← ランダムアクセス（seek）
  ├─ BGR→RGB 変換
  ├─ トラッカー更新 (OpenCVTracker)         ← RGB→BGR 再変換あり
  ├─ ROI切り出し
  └─ OCR推論 (EasyOCREngine.recognize)      ← CPU推論・最大のボトルネック
```

並列処理（チェックボックス ON 時）は、バッチ内フレームの OCR 推論をスレッドプールで並列化しているが、
Python の GIL により CPU バウンドな EasyOCR 推論は真の並列実行にならない。

---

## 修正案一覧

### 案1: GPU 推論を有効にする（優先度: 最高）

**概要**  
`EasyOCREngine` の初期化で `gpu=False` となっており、GPU が利用可能な環境でも CPU 推論に固定されている。
CUDA 環境があれば `gpu=True` にするだけで OCR 推論速度が 5〜20 倍向上する。

**実装箇所**: `src/infrastructure/ocr_engine.py`

```python
# 現状
self._reader = easyocr.Reader(["en"], gpu=False)

# 案: CUDA が使える場合は自動で GPU を使う
import torch
_use_gpu = torch.cuda.is_available()
self._reader = easyocr.Reader(["en"], gpu=_use_gpu)
```

GUI に「GPU使用」チェックボックスを追加するか、起動時に自動検出して適用する。

**期待効果**: CUDA 環境では処理時間が大幅短縮（GPU なし環境では変化なし）。  
**依存**: `torch`（既存の EasyOCR 依存に含まれる）

---

### 案2: フレームスキップ機能を追加する（優先度: 高）

**概要**  
映像中の数字が1秒以上変化しない場合、全フレームを処理する必要はない。
「N フレームに1回だけ OCR を実行する」スキップ間隔を GUI で設定できるようにする。

**仕様追加例**:
- GUI に「フレームスキップ間隔（1=全フレーム、2=1フレームおき）」スピンボックスを追加
- スキップしたフレームは前フレームの OCR 結果をそのままコピーする（補完）
- トラッキングはスキップフレームでも継続して更新する（精度維持）

**実装箇所**: `src/domain/use_cases.py` の `_execute_sequential` / `_execute_parallel`

```python
# スキップ間隔 skip_step を引数で受け取り
for i in range(total):
    frame = self._video_reader.read_frame(i)
    current_roi = self._update_tracking(i, frame, roi)
    if i % skip_step == 0:
        crop = self._extract_crop(frame, current_roi, ocr_sub_roi)
        last_value = self._ocr_engine.recognize(crop)
    results.append(FrameResult(frame_no=i, value=last_value))
```

**期待効果**: スキップ間隔 N 倍で OCR 推論回数が約 1/N に削減。30fps 動画で数字が1秒間隔の場合、間隔 15 で処理時間を約 1/15 に短縮。

---

### 案3: 動画のシーケンシャル読み込みに変更する（優先度: 高）

**概要**  
`VideoReader.read_frame()` は毎回 `cap.set(CAP_PROP_POS_FRAMES, frame_no)` でシーク後に読み込んでいる。
H.264 等の圧縮コーデックではシーク（特に B フレーム）がデコードコストを大幅に増やす。
全フレーム順次処理の場合は `cap.read()` を連続呼び出しするシーケンシャルモードが高速。

**実装箇所**: `src/infrastructure/video_reader.py`

```python
def read_next_frame(self) -> Optional[np.ndarray]:
    """現在位置から次のフレームを連続読み込みする（順次処理専用）"""
    with self._lock:
        ret, frame = self._cap.read()
    if not ret or frame is None:
        return None
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
```

`ProcessVideoUseCase._execute_sequential` で全フレームを順次処理する際に `read_next_frame()` を使用する。

**期待効果**: 圧縮動画（mp4/h264）での読み込み速度が 20〜50% 向上する場合がある。

---

### 案4: OCR 推論にプロセスプール並列を使用する（優先度: 高）

**概要**  
現在の並列処理は `ThreadPoolExecutor` を使っているが、Python の GIL により EasyOCR 推論（CPU バウンド）は1スレッドずつしか実行されない。
`ProcessPoolExecutor` に切り替えることで真の並列実行が可能になる。

**課題**: EasyOCR の `Reader` オブジェクトはプロセス間で共有できないため、
ワーカープロセスごとに `Reader` を初期化するか、`initializer` 関数でプロセスローカルに保持する必要がある。

**実装方針**:
```python
# ワーカープロセス初期化関数
_worker_reader = None

def _init_worker(use_gpu: bool):
    global _worker_reader
    import easyocr
    _worker_reader = easyocr.Reader(["en"], gpu=use_gpu)

def _ocr_task_proc(crop_bytes: bytes, shape: tuple) -> Optional[int]:
    crop = np.frombuffer(crop_bytes, dtype=np.uint8).reshape(shape)
    # _worker_reader を使って推論
    ...
```

**期待効果**: CPU コア数に応じて推論スループットが線形に近いスケールアップが見込める（4コアで最大 3〜4 倍）。

---

### 案5: EasyOCR のバッチ推論を活用する（優先度: 中）

**概要**  
EasyOCR の `readtext()` は画像リストを一括入力するバッチ処理に対応している（ただし公式ドキュメント上は非推奨）。
代わりに、内部の `recognize()` API を直接呼び出してバッチ推論を行う方法がある。

または、フレームをまとめてバッチ化し、GPU 推論の効率を上げる（案1 と組み合わせることで効果大）。

**参考**: EasyOCR のモデルはバッチサイズを明示してバッチ推論できる。
`Reader` の内部属性を使うか、EasyOCR のソース層に手を入れる必要があるため、難易度は中〜高。

**期待効果**: GPU 使用時にバッチ効率が向上し、単独推論より 2〜5 倍のスループット改善が期待できる。

---

### 案6: 代替 OCR エンジン（PaddleOCR）への切り替え（優先度: 中）

**概要**  
EasyOCR は汎用性が高いが CPU 推論が遅い。
PaddleOCR は軽量モデルを持ち、数字認識に限定すれば EasyOCR より CPU でも高速に動作する事例が多い。

**比較**:

| エンジン  | CPU推論速度 | GPU対応 | 数字認識精度 |
| --------- | ----------- | ------- | ------------ |
| EasyOCR   | 遅い        | ○       | 高い         |
| PaddleOCR | 速い        | ○       | 高い         |
| Tesseract | 中程度      | △       | 前処理次第   |

**実装方針**: `EasyOCREngine` と同一インターフェース（`recognize(image) -> Optional[int]`）を持つ
`PaddleOCREngine` クラスを `ocr_engine.py` に追加し、GUI でエンジンを選択できるようにする。

**依存追加**: `paddlepaddle`, `paddleocr`

---

### 案7: 色空間変換の重複を削減する（優先度: 低）

**概要**  
現在の実装では以下の変換が毎フレーム発生している。

```
VideoReader: BGR → RGB (cv2.cvtColor)
OpenCVTracker: RGB → BGR (cv2.cvtColor)  ← 逆変換
```

内部処理を BGR 統一に変更することで不要な変換コストを削減できる。
ただし改修範囲が広く（VideoReader・Tracker・OCREngine すべて影響）、リスクと効果のバランスは低優先度。

**実装箇所**: `src/infrastructure/video_reader.py`, `src/infrastructure/tracker.py`, `src/infrastructure/ocr_engine.py`

**期待効果**: フレームあたりの CPU 時間を数 ms 削減（全体への寄与は小）。

---

### 案8: 前処理の CLAHE オブジェクトをキャッシュする（優先度: 低）

**概要**  
`EasyOCREngine._preprocess()` 内で `cv2.createCLAHE(...)` を毎フレーム生成している。
オブジェクト生成コストは小さいが、インスタンス変数として1回だけ生成するのが望ましい。

**実装箇所**: `src/infrastructure/ocr_engine.py`

```python
# __init__ 内でキャッシュ
self._clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

# _preprocess 内
gray = self._clahe.apply(gray)
```

---

## 施策の優先度まとめ

| 優先度 | 案  | 内容                        | 実装難易度 | 期待効果       |
| ------ | --- | --------------------------- | ---------- | -------------- |
| 最高   | 案1 | GPU推論を有効化             | 低         | 最大           |
| 高     | 案2 | フレームスキップ機能        | 低〜中     | 大（用途依存） |
| 高     | 案3 | シーケンシャル読み込み      | 低         | 中             |
| 高     | 案4 | プロセスプール並列OCR       | 高         | 大（CPU環境）  |
| 中     | 案5 | バッチ推論                  | 高         | 中（GPU環境）  |
| 中     | 案6 | 代替エンジン（PaddleOCR）   | 中         | 中〜大         |
| 低     | 案7 | 色空間変換削減              | 中         | 小             |
| 低     | 案8 | CLAHEオブジェクトキャッシュ | 低         | 微小           |

---

## 推奨する実装順序

1. **案1（GPU 有効化）**: `gpu=False` を1行変更するだけで CUDA 環境では劇的に改善。
2. **案2（フレームスキップ）**: 仕様・実装とも小規模な変更で大きな効果が見込める。
3. **案3（シーケンシャル読み込み）**: `VideoReader` に `read_next_frame()` を追加するだけ。
4. **案6（PaddleOCR）**: GPU なし CPU 環境でのスループット改善として有効。
5. **案4（プロセスプール）**: 効果は大きいが EasyOCR との組み合わせに実装上の工夫が必要。
