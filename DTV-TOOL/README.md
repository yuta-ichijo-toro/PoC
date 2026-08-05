# DTV-tools

動画から数値を OCR で読み取り、GPS ルートと組み合わせて解析・可視化する Tkinter GUIツールです。

## 機能

| タブ           | 機能                                                                                                                                      |
| -------------- | ----------------------------------------------------------------------------------------------------------------------------------------- |
| **OCR解析**    | 動画を開いてROI（矩形領域）を指定し、全フレームの数値をEasyOCRで認識。OpenCV CSRT/KCF トラッカーによるROI追従、結果のCSV/JSONエクスポート |
| **解析・統計** | OCR結果CSVと色設定を読み込み、緯度経度データをもとにルートマップを描画（matplotlib / folium + 地図タイル対応）                            |

### 主な特徴
- **画像前処理オプション**: グレースケール変換・CLAHE・Otsu二値化でOCR精度を向上
- **EasyOCRパラメータ調整オプション**: `min_size` / `contrast_ths` 等の調整モード
- **並列処理オプション**: `ThreadPoolExecutor` による OCR 並列実行
- **クリーンアーキテクチャ**: Domain / Infrastructure / Presentation の3層構成

## ディレクトリ構成

```
DTV-tools/
├── main.py                      # エントリーポイント
├── requirements.txt
├── data/                        # 入力データ・出力結果
├── docs/                        # 仕様・改善ドキュメント
└── src/
    ├── domain/
    │   ├── entities.py          # ROI, FrameResult
    │   └── use_cases.py         # ProcessVideoUseCase, AnalyzeUseCase
    ├── infrastructure/
    │   ├── ocr_engine.py        # EasyOCREngine
    │   ├── tracker.py           # OpenCVTracker (CSRT/KCF)
    │   ├── video_reader.py      # VideoReader
    │   ├── map_renderer.py      # matplotlib / folium ルートマップ
    │   ├── analysis_exporter.py # CSV・画像・HTML エクスポート
    │   ├── csv_writer.py
    │   └── json_writer.py
    └── presentation/
        └── app.py               # Tkinter GUIアプリケーション
```

## セットアップ

### 動作要件
- Python 3.10 以上

### インストール

```bash
pip install -r requirements.txt
```

> **注意**: `opencv-contrib-python` が必要です（トラッカー機能のため）。
> `opencv-python` が既にインストール済みの場合は先にアンインストールしてください。

```bash
pip uninstall opencv-python -y
pip install opencv-contrib-python --trusted-host pypi.org --trusted-host files.pythonhosted.org
```

## 使い方

```bash
python main.py
```

1. **OCR解析タブ** で動画ファイルを開く
2. 映像上でROIをドラッグして指定
3. 「解析開始」で全フレームのOCRを実行
4. 結果をCSV/JSONに保存
5. **解析・統計タブ** で保存したCSVと色設定を読み込み、ルートマップを表示・エクスポート

## トラブルシューティング

### `cannot import name 'io' from 'skimage' (unknown location)`

**原因**  
`scikit-image 0.25` で `skimage.io` モジュールが削除されました。`easyocr` が内部で `skimage.io` を使用しているため、0.25 以上をインストールするとこのエラーが発生します。

**解決方法**  
`scikit-image` を 0.24 系にダウングレードしてください。

```bash
pip install "scikit-image<0.25" pypi.org files.pythonhosted.org
```

---

### `OpenCVトラッカー(CSRT)の生成に失敗しました。opencv-contrib-python をインストールしてください。`

**原因**  
`opencv-python`（contrib なし）では TrackerCSRT が利用できないため、トラッキング処理の開始時にこのエラーが発生します。

**解決方法**  
`opencv-contrib-python` をインストールしてください。すでに `opencv-python` が入っている場合は先にアンインストールしてから再インストールしてください。

```bash
pip uninstall opencv-python -y
pip install opencv-contrib-python --trusted-host pypi.org --trusted-host files.pythonhosted.org
```

> SSL証明書エラー (`CERTIFICATE_VERIFY_FAILED`) が出る場合は `--trusted-host` オプションを付けて実行してください。

---

### `module 'cv2' has no attribute 'legacy'`

**原因**  
`opencv-python`（contrib なし）では `cv2.legacy` モジュールが存在しないため、TrackerCSRT / TrackerKCF の生成時にこのエラーが発生します。

**解決方法**  
`opencv-contrib-python` をインストールしてください。すでに `opencv-python` が入っている場合は先にアンインストールしてから再インストールしてください。

```bash
pip uninstall opencv-python -y
pip install opencv-contrib-python pypi.org files.pythonhosted.org
```

---

### `OpenCV(x.x.x) error: (-4:Insufficient memory) Failed to allocate ... bytes in function 'cv::OutOfMemoryError'`

**原因**  
OCR前処理（「画像前処理」チェックボックスをオン）で `cv2.resize()` が画像を過度に拡大しようとした際、OpenCV がメモリ確保に失敗します。  
ROIが極端に縦長・横長（例：幅が非常に大きく高さが1〜数ピクセル）の場合、スケール係数が大きくなりすぎることが原因です。

**解決方法**  
以下のいずれかを試してください。

1. **ROIのサイズを調整する** — OCR対象範囲（ROI）をより正方形に近い形に再選択してください。
2. **「画像前処理」チェックボックスをオフにする** — 前処理なしで認識できる場合はオフのまま使用してください。

> バージョン XX 以降では、リサイズ後の最大辺が 4000px を超えないよう制限されており、このエラーは大幅に発生しにくくなっています。