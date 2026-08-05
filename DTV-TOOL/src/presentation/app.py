"""Tkinter GUIアプリケーション"""
import csv
import json
import os
import tempfile
import threading
import tkinter as tk
import webbrowser
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Any, Dict, List, Optional, Tuple

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from PIL import Image, ImageTk

from ..domain.entities import FrameResult, ROI
from ..domain.use_cases import AnalyzeUseCase, ProcessVideoUseCase
from ..infrastructure.analysis_exporter import write_map_html, write_map_image, write_result_csv
from ..infrastructure.csv_writer import write_csv
from ..infrastructure.json_writer import write_tracking_json
from ..infrastructure.map_renderer import (
    render_route_map,
    render_route_map_with_tiles,
    save_folium_map_to_tempfile,
)
from ..infrastructure.ocr_engine import EasyOCREngine
from ..infrastructure.tracker import OpenCVTracker
from ..infrastructure.video_reader import VideoReader


class App(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("DTV-tools")
        self.geometry("1100x700")
        self.minsize(800, 500)

        self._video_reader: Optional[VideoReader] = None
        self._results: List[FrameResult] = []
        self._tracking_results: Dict[int, ROI] = {}
        self._roi: Optional[ROI] = None
        self._current_frame_no: int = 0
        self._scale: float = 1.0
        self._ocr_running: bool = False
        self._roi_start: Optional[Tuple[int, int]] = None
        self._tk_img: Optional[ImageTk.PhotoImage] = None  # GC対策
        self._show_tracking = tk.BooleanVar(value=True)
        self._tracking_enabled = tk.BooleanVar(value=True)
        self._ocr_sub_roi: Optional[ROI] = None
        self._draw_mode = tk.StringVar(value="roi")
        self._use_preprocessing = tk.BooleanVar(value=False)  # 案1: 画像前処理
        self._use_tuned_params = tk.BooleanVar(value=False)   # 案2: EasyOCRパラメータ調整
        self._use_parallel = tk.BooleanVar(value=False)        # 並列処理

        # --- 処理速度オプション状態 ---
        self._use_gpu = tk.BooleanVar(value=False)             # 速度案1: GPU推論
        self._frame_skip_var = tk.IntVar(value=1)              # 速度案2: フレームスキップ間隔
        self._use_sequential_read = tk.BooleanVar(value=True)  # 速度案3: シーケンシャル読み込み
        self._use_process_parallel = tk.BooleanVar(value=False) # 速度案4: プロセス並列OCR

        # --- トラッキング回復オプション状態 ---
        self._tracking_reinit = tk.BooleanVar(value=True)           # 案A: 失敗時に再初期化
        self._tracking_drift_detection = tk.BooleanVar(value=False) # 案B: ドリフト検出
        self._tracking_drift_factor_var = tk.StringVar(value="1.5") # 案B: ドリフト許容係数
        self._tracking_template_matching = tk.BooleanVar(value=False) # 案C: テンプレートマッチング
        self._tracking_template_threshold_var = tk.StringVar(value="0.5") # 案C: 類似度閾値
        self._tracking_template_margin_var = tk.StringVar(value="0.5")    # 案C: 探索余白係数

        # --- セグメント分割トラッキング状態 ---
        self._segment_rois: Dict[int, ROI] = {}
        self._segment_tracking_enabled = tk.BooleanVar(value=False)
        self._segment_interval_var = tk.IntVar(value=180)
        self._segment_edit_mode: Optional[int] = None
        self._segment_dialog: Optional[tk.Toplevel] = None
        self._seg_list_refresh = lambda: None

        # --- 解析・統計タブ状態 ---
        self._analysis_csv_path: Optional[str] = None
        self._color_config: Dict[str, str] = {}
        self._color_config_path: Optional[str] = None
        self._analysis_rows: List[Dict[str, Any]] = []
        self._map_figure: Optional[Figure] = None
        self._map_canvas_agg: Optional[FigureCanvasTkAgg] = None
        self._use_map_tiles = tk.BooleanVar(value=False)

        self._build_ui()
        self._load_default_color_config()
        self._load_perf_config()

    # ================================================================== #
    # UI構築
    # ================================================================== #

    def _build_ui(self) -> None:
        # ステータスバーは BOTTOM pack のため最初に配置する
        self._build_status_bar()

        notebook = ttk.Notebook(self)
        notebook.pack(fill=tk.BOTH, expand=True)

        ocr_tab = tk.Frame(notebook)
        notebook.add(ocr_tab, text="OCR解析")
        self._build_ocr_tab(ocr_tab)

        analysis_tab = tk.Frame(notebook)
        notebook.add(analysis_tab, text="解析・統計")
        self._build_analysis_tab(analysis_tab)

    def _build_status_bar(self) -> None:
        bottom = tk.Frame(self, bd=1, relief=tk.SUNKEN)
        bottom.pack(side=tk.BOTTOM, fill=tk.X)
        self._status_var = tk.StringVar(value="動画ファイルを開いてください")
        tk.Label(bottom, textvariable=self._status_var, anchor=tk.W).pack(
            fill=tk.X, padx=4, pady=1
        )
        self._progress_var = tk.DoubleVar(value=0)
        self._progressbar = ttk.Progressbar(
            bottom, variable=self._progress_var, maximum=100
        )
        self._progressbar.pack(fill=tk.X, padx=4, pady=2)

    # ------------------------------------------------------------------ #
    # OCR解析タブ
    # ------------------------------------------------------------------ #

    def _build_ocr_tab(self, parent: tk.Frame) -> None:
        # --- ツールバー ---
        toolbar = tk.Frame(parent, bd=1, relief=tk.RAISED)
        toolbar.pack(side=tk.TOP, fill=tk.X)

        self._btn_open = tk.Button(toolbar, text="動画を開く", command=self._open_video)
        self._btn_open.pack(side=tk.LEFT, padx=2, pady=2)

        self._btn_load_ocr_csv = tk.Button(
            toolbar, text="OCR結果を読み込む", command=self._load_ocr_csv, state=tk.DISABLED
        )
        self._btn_load_ocr_csv.pack(side=tk.LEFT, padx=2, pady=2)

        self._btn_ocr = tk.Button(
            toolbar, text="OCR実行", command=self._run_ocr, state=tk.DISABLED
        )
        self._btn_ocr.pack(side=tk.LEFT, padx=2, pady=2)

        self._btn_csv = tk.Button(
            toolbar, text="CSV保存", command=self._save_csv, state=tk.DISABLED
        )
        self._btn_csv.pack(side=tk.LEFT, padx=2, pady=2)

        self._btn_json = tk.Button(
            toolbar, text="JSON保存", command=self._save_json, state=tk.DISABLED
        )
        self._btn_json.pack(side=tk.LEFT, padx=2, pady=2)

        tk.Checkbutton(
            toolbar,
            text="トラッキング有効",
            variable=self._tracking_enabled,
        ).pack(side=tk.LEFT, padx=4)

        tk.Checkbutton(
            toolbar,
            text="トラッキング表示",
            variable=self._show_tracking,
            command=lambda: self._show_frame(self._current_frame_no),
        ).pack(side=tk.LEFT, padx=4)

        self._roi_label = tk.Label(
            toolbar, text="ROI: 未設定（動画上でドラッグして範囲を指定）", fg="gray"
        )
        self._roi_label.pack(side=tk.LEFT, padx=12)

        tk.Label(toolbar, text="|", fg="gray").pack(side=tk.LEFT, padx=2)
        tk.Label(toolbar, text="描画:").pack(side=tk.LEFT)
        tk.Radiobutton(
            toolbar, text="ROI選択", variable=self._draw_mode, value="roi"
        ).pack(side=tk.LEFT)
        tk.Radiobutton(
            toolbar, text="OCR範囲指定", variable=self._draw_mode, value="ocr"
        ).pack(side=tk.LEFT)
        tk.Button(
            toolbar, text="OCR範囲クリア", command=self._clear_ocr_sub_roi
        ).pack(side=tk.LEFT, padx=2, pady=2)
        self._ocr_sub_roi_label = tk.Label(
            toolbar, text="OCR範囲: 未設定（ROI全体を使用）", fg="gray"
        )
        self._ocr_sub_roi_label.pack(side=tk.LEFT, padx=8)

        # --- 精度オプションバー（2段目） ---
        acc_bar = tk.Frame(parent, bd=1, relief=tk.GROOVE, bg="#f0f0f0")
        acc_bar.pack(side=tk.TOP, fill=tk.X)
        tk.Label(acc_bar, text="OCR精度オプション:", bg="#f0f0f0", font=("" , 9, "bold")).pack(
            side=tk.LEFT, padx=(6, 2), pady=2
        )
        tk.Checkbutton(
            acc_bar,
            text="[案1] 画像前処理（グレースケール・拡大・CLAHE・二値化）",
            variable=self._use_preprocessing,
            bg="#f0f0f0",
        ).pack(side=tk.LEFT, padx=6)
        tk.Checkbutton(
            acc_bar,
            text="[案2] EasyOCRパラメータ調整（min_size / contrast_ths 等）",
            variable=self._use_tuned_params,
            bg="#f0f0f0",
        ).pack(side=tk.LEFT, padx=6)
        ttk.Separator(acc_bar, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=8, pady=3)
        tk.Checkbutton(
            acc_bar,
            text="並列処理（OCR高速化）",
            variable=self._use_parallel,
            bg="#f0f0f0",
        ).pack(side=tk.LEFT, padx=6)

        # --- 処理速度オプションバー（3段目） ---
        speed_bar = tk.Frame(parent, bd=1, relief=tk.GROOVE, bg="#e8f0e8")
        speed_bar.pack(side=tk.TOP, fill=tk.X)
        tk.Label(speed_bar, text="処理速度オプション:", bg="#e8f0e8", font=("", 9, "bold")).pack(
            side=tk.LEFT, padx=(6, 2), pady=2
        )
        tk.Checkbutton(
            speed_bar,
            text="[案1] GPU使用 (CUDA)",
            variable=self._use_gpu,
            bg="#e8f0e8",
            command=self._save_perf_config,
        ).pack(side=tk.LEFT, padx=6)
        ttk.Separator(speed_bar, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=6, pady=3)
        tk.Checkbutton(
            speed_bar,
            text="[案3] シーケンシャル読み込み",
            variable=self._use_sequential_read,
            bg="#e8f0e8",
            command=self._save_perf_config,
        ).pack(side=tk.LEFT, padx=6)
        ttk.Separator(speed_bar, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=6, pady=3)
        tk.Label(speed_bar, text="[案2] フレームスキップ間隔:", bg="#e8f0e8").pack(side=tk.LEFT)
        self._frame_skip_spinbox = tk.Spinbox(
            speed_bar,
            from_=1,
            to=30,
            width=4,
            textvariable=self._frame_skip_var,
            command=self._save_perf_config,
        )
        self._frame_skip_spinbox.pack(side=tk.LEFT, padx=4)
        tk.Label(speed_bar, text="フレームごとに1回OCR（1=全フレーム）", bg="#e8f0e8").pack(
            side=tk.LEFT, padx=(0, 8)
        )
        ttk.Separator(speed_bar, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=6, pady=3)
        tk.Checkbutton(
            speed_bar,
            text="[案4] プロセス並列OCR（GIL回避）",
            variable=self._use_process_parallel,
            bg="#e8f0e8",
            command=self._save_perf_config,
        ).pack(side=tk.LEFT, padx=6)

        # --- トラッキング回復オプションバー（4段目） ---
        tracking_bar = tk.Frame(parent, bd=1, relief=tk.GROOVE, bg="#e8e8f4")
        tracking_bar.pack(side=tk.TOP, fill=tk.X)
        tk.Label(tracking_bar, text="トラッキング回復:", bg="#e8e8f4", font=("", 9, "bold")).pack(
            side=tk.LEFT, padx=(6, 2), pady=2
        )
        tk.Checkbutton(
            tracking_bar,
            text="[案A] 失敗時に再初期化",
            variable=self._tracking_reinit,
            bg="#e8e8f4",
            command=self._save_perf_config,
        ).pack(side=tk.LEFT, padx=6)
        ttk.Separator(tracking_bar, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=6, pady=3)
        tk.Checkbutton(
            tracking_bar,
            text="[案B] ドリフト検出",
            variable=self._tracking_drift_detection,
            bg="#e8e8f4",
            command=self._save_perf_config,
        ).pack(side=tk.LEFT, padx=6)
        tk.Label(tracking_bar, text="係数:", bg="#e8e8f4").pack(side=tk.LEFT)
        tk.Spinbox(
            tracking_bar,
            from_=0.5, to=5.0, increment=0.5, format="%.1f",
            width=5,
            textvariable=self._tracking_drift_factor_var,
            command=self._save_perf_config,
        ).pack(side=tk.LEFT, padx=4)
        ttk.Separator(tracking_bar, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=6, pady=3)
        tk.Checkbutton(
            tracking_bar,
            text="[案C] テンプレートマッチング",
            variable=self._tracking_template_matching,
            bg="#e8e8f4",
            command=self._save_perf_config,
        ).pack(side=tk.LEFT, padx=6)
        tk.Label(tracking_bar, text="類似度閾値:", bg="#e8e8f4").pack(side=tk.LEFT)
        tk.Spinbox(
            tracking_bar,
            from_=0.1, to=1.0, increment=0.1, format="%.1f",
            width=5,
            textvariable=self._tracking_template_threshold_var,
            command=self._save_perf_config,
        ).pack(side=tk.LEFT, padx=4)

        # --- セグメント分割トラッキングバー（5段目） ---
        segment_bar = tk.Frame(parent, bd=1, relief=tk.GROOVE, bg="#e8f4e8")
        segment_bar.pack(side=tk.TOP, fill=tk.X)
        tk.Label(segment_bar, text="セグメント分割:", bg="#e8f4e8", font=("", 9, "bold")).pack(
            side=tk.LEFT, padx=(6, 2), pady=2
        )
        tk.Checkbutton(
            segment_bar,
            text="[案D] セグメント分割トラッキング有効",
            variable=self._segment_tracking_enabled,
            bg="#e8f4e8",
            command=self._save_perf_config,
        ).pack(side=tk.LEFT, padx=6)
        ttk.Separator(segment_bar, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=6, pady=3)
        tk.Label(segment_bar, text="間隔（フレーム数）:", bg="#e8f4e8").pack(side=tk.LEFT)
        tk.Spinbox(
            segment_bar,
            from_=1,
            to=99999,
            width=7,
            textvariable=self._segment_interval_var,
            command=self._save_perf_config,
        ).pack(side=tk.LEFT, padx=4)
        self._segment_interval_label = tk.Label(segment_bar, text="", bg="#e8f4e8", fg="#555555")
        self._segment_interval_label.pack(side=tk.LEFT, padx=4)
        ttk.Separator(segment_bar, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=6, pady=3)
        tk.Button(
            segment_bar,
            text="セグメントROI設定...",
            command=self._open_segment_roi_dialog,
        ).pack(side=tk.LEFT, padx=4, pady=2)
        self._segment_roi_count_label = tk.Label(segment_bar, text="ROI設定済み: 0", bg="#e8f4e8", fg="gray")
        self._segment_roi_count_label.pack(side=tk.LEFT, padx=6)

        # --- 左右ペイン ---
        paned = tk.PanedWindow(parent, orient=tk.HORIZONTAL, sashwidth=5)
        paned.pack(fill=tk.BOTH, expand=True)

        # 左: キャンバス + スライダー
        left = tk.Frame(paned)
        paned.add(left, stretch="always")

        self._canvas = tk.Canvas(left, bg="#1e1e1e", cursor="crosshair")
        self._canvas.pack(fill=tk.BOTH, expand=True)
        self._canvas.bind("<ButtonPress-1>", self._on_roi_press)
        self._canvas.bind("<B1-Motion>", self._on_roi_drag)
        self._canvas.bind("<ButtonRelease-1>", self._on_roi_release)
        self._canvas.bind("<Configure>", self._on_canvas_resize)

        slider_row = tk.Frame(left)
        slider_row.pack(fill=tk.X)
        self._slider_var = tk.IntVar(value=0)
        self._slider = ttk.Scale(
            slider_row,
            from_=0,
            to=0,
            orient=tk.HORIZONTAL,
            variable=self._slider_var,
            command=self._on_slider,
        )
        self._slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=4, pady=2)
        self._frame_pos_label = tk.Label(slider_row, text="0 / 0", width=12)
        self._frame_pos_label.pack(side=tk.RIGHT, padx=4)

        # 右: 情報パネル
        right = tk.Frame(paned, width=270)
        paned.add(right, stretch="never")

        tk.Label(right, text="フレーム情報", font=("", 11, "bold")).pack(pady=(10, 4))

        self._frame_no_label = tk.Label(right, text="フレーム: -")
        self._frame_no_label.pack()

        tk.Label(right, text="OCR結果:").pack(pady=(10, 2))
        self._ocr_value_var = tk.StringVar()
        self._ocr_entry = tk.Entry(
            right, textvariable=self._ocr_value_var, width=18, font=("", 16), justify=tk.CENTER
        )
        self._ocr_entry.pack(pady=4)
        self._ocr_entry.bind("<Return>", self._on_edit_commit)
        self._ocr_entry.bind("<FocusOut>", self._on_edit_commit)

        nav = tk.Frame(right)
        nav.pack(pady=6)
        tk.Button(nav, text="◀ 前", command=self._prev_frame, width=8).pack(
            side=tk.LEFT, padx=4
        )
        tk.Button(nav, text="次 ▶", command=self._next_frame, width=8).pack(
            side=tk.LEFT, padx=4
        )

        ttk.Separator(right, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=10, padx=6)

        tk.Label(right, text="結果一覧:").pack()
        list_frame = tk.Frame(right)
        list_frame.pack(fill=tk.BOTH, expand=True, padx=4, pady=(2, 4))
        sb = tk.Scrollbar(list_frame)
        sb.pack(side=tk.RIGHT, fill=tk.Y)
        self._listbox = tk.Listbox(
            list_frame, yscrollcommand=sb.set, font=("Courier", 9), activestyle="none"
        )
        self._listbox.pack(fill=tk.BOTH, expand=True)
        sb.config(command=self._listbox.yview)
        self._listbox.bind("<<ListboxSelect>>", self._on_list_select)

    # ------------------------------------------------------------------ #
    # 解析・統計タブ
    # ------------------------------------------------------------------ #

    def _build_analysis_tab(self, parent: tk.Frame) -> None:
        # --- コントロールパネル ---
        ctrl = tk.LabelFrame(parent, text="解析設定", padx=6, pady=4)
        ctrl.pack(side=tk.TOP, fill=tk.X, padx=6, pady=(6, 2))

        # 行1: GPS CSV読み込み
        row1 = tk.Frame(ctrl)
        row1.pack(fill=tk.X, pady=2)
        tk.Button(
            row1, text="GPS/時刻CSVを開く", command=self._load_analysis_csv, width=20
        ).pack(side=tk.LEFT)
        self._analysis_csv_label = tk.Label(row1, text="未選択", fg="gray", anchor=tk.W)
        self._analysis_csv_label.pack(side=tk.LEFT, padx=8, fill=tk.X, expand=True)

        # 行2: 動画開始タイムスタンプ
        row2 = tk.Frame(ctrl)
        row2.pack(fill=tk.X, pady=2)
        tk.Label(row2, text="動画開始 Unixタイムスタンプ（秒）:").pack(side=tk.LEFT)
        self._start_ts_var = tk.StringVar(value="0")
        tk.Entry(row2, textvariable=self._start_ts_var, width=16).pack(side=tk.LEFT, padx=6)

        # 行3: 色設定ファイル
        row3 = tk.Frame(ctrl)
        row3.pack(fill=tk.X, pady=2)
        tk.Button(
            row3, text="色設定ファイルを開く", command=self._load_color_config, width=20
        ).pack(side=tk.LEFT)
        self._color_config_label = tk.Label(
            row3, text="未選択", fg="gray", anchor=tk.W
        )
        self._color_config_label.pack(side=tk.LEFT, padx=8, fill=tk.X, expand=True)

        # 行4: 解析・出力ボタン
        row4 = tk.Frame(ctrl)
        row4.pack(fill=tk.X, pady=4)
        self._btn_analyze = tk.Button(
            row4, text="解析実行", command=self._run_analysis, width=14
        )
        self._btn_analyze.pack(side=tk.LEFT, padx=(0, 6))
        self._btn_export = tk.Button(
            row4, text="出力", command=self._export_results, state=tk.DISABLED, width=14
        )
        self._btn_export.pack(side=tk.LEFT, padx=6)
        self._btn_open_map = tk.Button(
            row4,
            text="ブラウザで地図を開く",
            command=self._open_map_in_browser,
            state=tk.DISABLED,
            width=18,
        )
        self._btn_open_map.pack(side=tk.LEFT, padx=6)
        self._analysis_status_label = tk.Label(row4, text="", fg="#555555", anchor=tk.W)
        self._analysis_status_label.pack(side=tk.LEFT, padx=10)

        # --- マップ表示コントロールバー ---
        map_ctrl = tk.Frame(parent)
        map_ctrl.pack(fill=tk.X, padx=6, pady=(2, 0))
        tk.Checkbutton(
            map_ctrl,
            text="地図タイル表示 (OpenStreetMap)",
            variable=self._use_map_tiles,
            command=self._on_map_view_change,
        ).pack(side=tk.LEFT)
        tk.Label(
            map_ctrl,
            text="※タイル取得にはインターネット接続が必要です",
            fg="#888888",
            font=("", 8),
        ).pack(side=tk.LEFT, padx=8)

        # --- マップ表示エリア ---
        self._map_frame = tk.Frame(parent, bg="#f4f4f4", bd=1, relief=tk.SUNKEN)
        self._map_frame.pack(fill=tk.BOTH, expand=True, padx=6, pady=(2, 6))

        self._map_placeholder = tk.Label(
            self._map_frame,
            text="解析を実行するとマップが表示されます",
            fg="#aaaaaa",
            bg="#f4f4f4",
            font=("", 12),
        )
        self._map_placeholder.pack(expand=True)

    # ------------------------------------------------------------------ #
    # 動画読み込み
    # ------------------------------------------------------------------ #

    def _open_video(self) -> None:
        if self._ocr_running:
            return
        path = filedialog.askopenfilename(
            title="動画ファイルを選択",
            filetypes=[
                ("動画ファイル", "*.mp4 *.avi *.mov *.mkv *.wmv *.flv *.ts"),
                ("すべて", "*.*"),
            ],
        )
        if not path:
            return
        try:
            if self._video_reader:
                self._video_reader.release()
            self._video_reader = VideoReader(path)
        except ValueError as e:
            messagebox.showerror("エラー", str(e))
            return

        self._results = []
        self._tracking_results = {}
        self._roi = None
        self._roi_label.config(
            text="ROI: 未設定（動画上でドラッグして範囲を指定）", fg="gray"
        )
        self._ocr_sub_roi = None
        self._ocr_sub_roi_label.config(text="OCR範囲: 未設定（ROI全体を使用）", fg="gray")
        self._segment_rois = {}
        self._segment_edit_mode = None
        self._update_segment_roi_count_label()
        self._update_segment_interval_label()
        self._current_frame_no = 0
        self._slider.config(to=max(0, self._video_reader.frame_count - 1))
        self._slider_var.set(0)
        self._btn_ocr.config(state=tk.NORMAL)
        self._btn_load_ocr_csv.config(state=tk.NORMAL)
        self._btn_csv.config(state=tk.DISABLED)
        self._btn_json.config(state=tk.DISABLED)
        self._listbox.delete(0, tk.END)
        self._progress_var.set(0)
        self._status_var.set(
            f"読み込み完了: {self._video_reader.frame_count} フレーム  "
            f"{self._video_reader.fps:.2f} fps  "
            f"{self._video_reader.width}×{self._video_reader.height}"
        )
        self.after(50, lambda: self._show_frame(0))

    # ------------------------------------------------------------------ #
    # フレーム表示
    # ------------------------------------------------------------------ #

    def _show_frame(self, frame_no: int) -> None:
        if self._video_reader is None:
            return
        frame = self._video_reader.read_frame(frame_no)
        if frame is None:
            return

        h, w = frame.shape[:2]
        cw = self._canvas.winfo_width() or 640
        ch = self._canvas.winfo_height() or 480
        self._scale = min(cw / w, ch / h)
        nw = max(1, int(w * self._scale))
        nh = max(1, int(h * self._scale))

        img = Image.fromarray(frame).resize((nw, nh), Image.LANCZOS)
        self._tk_img = ImageTk.PhotoImage(img)

        self._canvas.delete("all")
        self._canvas.create_image(0, 0, anchor=tk.NW, image=self._tk_img, tags="frame")

        if self._roi:
            self._redraw_roi()

        # トラッキング結果があればその位置を基準にOCR範囲を描画
        if self._ocr_sub_roi:
            base_roi = self._tracking_results.get(frame_no) if self._tracking_results else None
            self._redraw_ocr_sub_roi(base_roi=base_roi)

        # トラッキング範囲のオーバーレイ表示
        if self._show_tracking.get() and self._tracking_results:
            track_roi = self._tracking_results.get(frame_no)
            if track_roi is not None:
                s = self._scale
                tx1 = int(track_roi.x * s)
                ty1 = int(track_roi.y * s)
                tx2 = int((track_roi.x + track_roi.w) * s)
                ty2 = int((track_roi.y + track_roi.h) * s)
                self._canvas.create_rectangle(
                    tx1, ty1, tx2, ty2,
                    outline="#44aaff", width=2, dash=(6, 3), tags="tracking"
                )

        # セグメント分割ROIのオーバーレイ表示（シアン枠）
        if self._segment_tracking_enabled.get() and self._segment_rois:
            iv = self._segment_interval_var.get()
            if iv > 0:
                seg_start = (frame_no // iv) * iv
                if seg_start in self._segment_rois:
                    sr = self._segment_rois[seg_start]
                    s = self._scale
                    sx1 = int(sr.x * s)
                    sy1 = int(sr.y * s)
                    sx2 = int((sr.x + sr.w) * s)
                    sy2 = int((sr.y + sr.h) * s)
                    self._canvas.create_rectangle(
                        sx1, sy1, sx2, sy2,
                        outline="#00cccc", width=2, dash=(4, 3), tags="seg_roi"
                    )

        total = self._video_reader.frame_count
        self._frame_no_label.config(text=f"フレーム: {frame_no} / {total - 1}")
        self._frame_pos_label.config(text=f"{frame_no} / {total - 1}")

        if self._results and frame_no < len(self._results):
            self._ocr_value_var.set(self._results[frame_no].value)
        else:
            self._ocr_value_var.set("")

    def _redraw_roi(self) -> None:
        if self._roi is None:
            return
        s = self._scale
        x1 = int(self._roi.x * s)
        y1 = int(self._roi.y * s)
        x2 = int((self._roi.x + self._roi.w) * s)
        y2 = int((self._roi.y + self._roi.h) * s)
        self._canvas.delete("roi")
        self._canvas.create_rectangle(x1, y1, x2, y2, outline="#ff4444", width=2, tags="roi")

    def _redraw_ocr_sub_roi(self, base_roi: Optional[ROI] = None) -> None:
        if self._ocr_sub_roi is None or self._roi is None:
            return
        s = self._scale
        # トラッキング結果があればその位置を基準に、なければ初期ROIを使用
        ref = base_roi if base_roi is not None else self._roi
        abs_x = ref.x + self._ocr_sub_roi.x
        abs_y = ref.y + self._ocr_sub_roi.y
        x1 = int(abs_x * s)
        y1 = int(abs_y * s)
        x2 = int((abs_x + self._ocr_sub_roi.w) * s)
        y2 = int((abs_y + self._ocr_sub_roi.h) * s)
        self._canvas.delete("ocr_sub_roi")
        self._canvas.create_rectangle(x1, y1, x2, y2, outline="#ffdd00", width=2, tags="ocr_sub_roi")

    def _on_canvas_resize(self, event: tk.Event) -> None:  # type: ignore[type-arg]
        if not self._ocr_running:
            self._show_frame(self._current_frame_no)

    # ------------------------------------------------------------------ #
    # ROI選択
    # ------------------------------------------------------------------ #

    def _on_roi_press(self, event: tk.Event) -> None:  # type: ignore[type-arg]
        if self._video_reader is None or self._ocr_running:
            return
        if self._draw_mode.get() == "ocr" and self._roi is None:
            self._status_var.set("先にトラッキングROIを選択してください（描画モード: ROI選択）")
            return
        self._roi_start = (event.x, event.y)
        if self._draw_mode.get() == "roi":
            self._canvas.delete("roi")
        else:
            self._canvas.delete("ocr_sub_roi")

    def _on_roi_drag(self, event: tk.Event) -> None:  # type: ignore[type-arg]
        if self._roi_start is None:
            return
        x0, y0 = self._roi_start
        if self._draw_mode.get() == "roi":
            self._canvas.delete("roi")
            color = "#00cccc" if self._segment_edit_mode is not None else "#ff4444"
            self._canvas.create_rectangle(
                x0, y0, event.x, event.y, outline=color, width=2, tags="roi"
            )
        else:
            self._canvas.delete("ocr_sub_roi")
            self._canvas.create_rectangle(
                x0, y0, event.x, event.y, outline="#ffdd00", width=2, tags="ocr_sub_roi"
            )

    def _on_roi_release(self, event: tk.Event) -> None:  # type: ignore[type-arg]
        if self._roi_start is None:
            return
        x0, y0 = self._roi_start
        x1, y1 = event.x, event.y
        self._roi_start = None

        # 正規化
        if x0 > x1:
            x0, x1 = x1, x0
        if y0 > y1:
            y0, y1 = y1, y0

        if x1 - x0 < 5 or y1 - y0 < 5:
            return  # 小さすぎる選択は無視

        s = self._scale if self._scale > 0 else 1.0

        if self._draw_mode.get() == "roi":
            new_roi = ROI(
                x=int(x0 / s),
                y=int(y0 / s),
                w=int((x1 - x0) / s),
                h=int((y1 - y0) / s),
            )
            if self._segment_edit_mode is not None:
                # セグメントROI設定モード: メインROIには保存しない
                seg_frame = self._segment_edit_mode
                self._segment_rois[seg_frame] = new_roi
                self._segment_edit_mode = None
                self._update_segment_roi_count_label()
                self._status_var.set(
                    f"セグメント（フレーム {seg_frame}）のROI設定完了: "
                    f"({new_roi.x}, {new_roi.y})  {new_roi.w}×{new_roi.h} px"
                )
                if self._segment_dialog and self._segment_dialog.winfo_exists():
                    self._seg_list_refresh()
            else:
                self._roi = new_roi
                self._roi_label.config(
                    text=f"ROI: ({self._roi.x}, {self._roi.y})  {self._roi.w}×{self._roi.h} px",
                    fg="green",
                )
                # ROI変更時にOCR範囲をリセット
                self._ocr_sub_roi = None
                self._canvas.delete("ocr_sub_roi")
                self._ocr_sub_roi_label.config(text="OCR範囲: 未設定（ROI全体を使用）", fg="gray")
        else:
            # OCR範囲をROI相対座標で保存
            abs_x = int(x0 / s)
            abs_y = int(y0 / s)
            abs_w = int((x1 - x0) / s)
            abs_h = int((y1 - y0) / s)
            rel_x = abs_x - self._roi.x
            rel_y = abs_y - self._roi.y
            self._ocr_sub_roi = ROI(x=rel_x, y=rel_y, w=abs_w, h=abs_h)
            self._ocr_sub_roi_label.config(
                text=f"OCR範囲: ({rel_x:+d}, {rel_y:+d})  {abs_w}×{abs_h} px",
                fg="#aa8800",
            )

    # ------------------------------------------------------------------ #
    # OCR範囲クリア
    # ------------------------------------------------------------------ #

    def _clear_ocr_sub_roi(self) -> None:
        self._ocr_sub_roi = None
        self._canvas.delete("ocr_sub_roi")
        self._ocr_sub_roi_label.config(text="OCR範囲: 未設定（ROI全体を使用）", fg="gray")

    # ------------------------------------------------------------------ #
    # フレームナビゲーション
    # ------------------------------------------------------------------ #

    def _on_slider(self, val: str) -> None:
        frame_no = int(float(val))
        if frame_no != self._current_frame_no and not self._ocr_running:
            self._current_frame_no = frame_no
            self._show_frame(frame_no)
            self._sync_listbox(frame_no)

    def _prev_frame(self) -> None:
        if self._video_reader is None:
            return
        self._go_to_frame(max(0, self._current_frame_no - 1))

    def _next_frame(self) -> None:
        if self._video_reader is None:
            return
        self._go_to_frame(
            min(self._video_reader.frame_count - 1, self._current_frame_no + 1)
        )

    def _go_to_frame(self, n: int) -> None:
        self._current_frame_no = n
        self._slider_var.set(n)
        self._show_frame(n)
        self._sync_listbox(n)

    def _sync_listbox(self, n: int) -> None:
        if self._results and n < self._listbox.size():
            self._listbox.selection_clear(0, tk.END)
            self._listbox.selection_set(n)
            self._listbox.see(n)

    def _on_list_select(self, event: tk.Event) -> None:  # type: ignore[type-arg]
        sel = self._listbox.curselection()
        if not sel:
            return
        n = sel[0]
        self._current_frame_no = n
        self._slider_var.set(n)
        self._show_frame(n)

    # ------------------------------------------------------------------ #
    # OCR結果の編集
    # ------------------------------------------------------------------ #

    def _on_edit_commit(self, event: Optional[tk.Event] = None) -> None:  # type: ignore[type-arg]
        if not self._results:
            return
        idx = self._current_frame_no
        if idx >= len(self._results):
            return
        new_val = self._ocr_value_var.get()
        self._results[idx].value = new_val
        self._results[idx].edited = True
        # リストボックス更新
        self._listbox.delete(idx)
        self._listbox.insert(idx, f"{idx:>6}: {new_val}")
        self._listbox.selection_set(idx)

    # ------------------------------------------------------------------ #
    # OCR実行（バックグラウンドスレッド）
    # ------------------------------------------------------------------ #

    def _run_ocr(self) -> None:
        if self._video_reader is None:
            messagebox.showwarning("警告", "先に動画ファイルを開いてください。")
            return
        if self._roi is None:
            messagebox.showwarning(
                "警告",
                "OCR対象範囲を選択してください。\n動画上でマウスをドラッグして範囲を指定してください。",
            )
            return

        self._set_ui_ocr_mode(running=True)
        self._status_var.set("OCRエンジンを初期化中... (初回は数十秒かかる場合があります)")
        self._progress_var.set(0)

        def _worker() -> None:
            try:
                use_process_parallel = self._use_process_parallel.get()
                # プロセス並列時は各ワーカープロセスがエンジンを初期化するため、メインプロセスでの初期化をスキップする
                if use_process_parallel:
                    ocr_engine = None
                    self.after(0, lambda: self._status_var.set(
                        f"ワーカープロセスを起動中... (最大 {os.cpu_count()} プロセス、初回起動は数十秒かかる場合あり)"
                    ))
                else:
                    ocr_engine = EasyOCREngine(
                        use_preprocessing=self._use_preprocessing.get(),
                        use_tuned_params=self._use_tuned_params.get(),
                        use_gpu=self._use_gpu.get(),
                    )
                tracker = OpenCVTracker(
                    algorithm="CSRT",
                    use_template_matching=self._tracking_template_matching.get(),
                    template_threshold=float(self._tracking_template_threshold_var.get()),
                    template_margin_factor=float(self._tracking_template_margin_var.get()),
                ) if self._tracking_enabled.get() else None
                use_case = ProcessVideoUseCase(
                    self._video_reader,
                    ocr_engine,
                    tracker,
                    tracking_reinit_on_failure=self._tracking_reinit.get(),
                    tracking_drift_detection=self._tracking_drift_detection.get(),
                    tracking_drift_factor=float(self._tracking_drift_factor_var.get()),
                    segment_interval=self._segment_interval_var.get() if self._segment_tracking_enabled.get() else 0,
                    segment_rois=dict(self._segment_rois) if self._segment_tracking_enabled.get() else {},
                )

                def _progress(done: int, total: int) -> None:
                    pct = done / total * 100
                    self.after(0, lambda p=pct: self._progress_var.set(p))
                    frame_count = self._video_reader.frame_count if self._video_reader else 0
                    if frame_count > 0 and total == frame_count * 2:
                        # 並列モード: total = frame_count * 2（前半=トラッキング、後半=OCR）
                        if done <= frame_count:
                            phase, actual = "トラッキング中", done
                        else:
                            phase, actual = "OCR処理中", done - frame_count
                        msg = f"{phase}: {actual} / {frame_count} フレーム ({pct:.0f}%)"
                    else:
                        msg = f"OCR処理中: {done} / {total} フレーム"
                    self.after(0, lambda m=msg: self._status_var.set(m))

                results, tracking_results = use_case.execute(
                    self._roi,
                    ocr_sub_roi=self._ocr_sub_roi,
                    progress_callback=_progress,
                    use_parallel=self._use_parallel.get() and not use_process_parallel,
                    use_process_parallel=use_process_parallel,
                    frame_skip=max(1, self._frame_skip_var.get()),
                    use_sequential_read=self._use_sequential_read.get(),
                    ocr_engine_params={
                        "use_preprocessing": self._use_preprocessing.get(),
                        "use_tuned_params": self._use_tuned_params.get(),
                        "use_gpu": self._use_gpu.get(),
                    },
                )
                self.after(0, lambda: self._on_ocr_done(results, tracking_results))
            except RuntimeError as exc:
                msg = str(exc)
                self.after(0, lambda: messagebox.showerror("実行エラー", msg))
                self.after(0, lambda: self._status_var.set("実行エラーが発生しました"))
            except Exception as exc:
                msg = str(exc)
                self.after(0, lambda: messagebox.showerror("OCRエラー", msg))
                self.after(0, lambda: self._status_var.set("OCRエラーが発生しました"))
                self.after(0, lambda: self._set_ui_ocr_mode(running=False))

        threading.Thread(target=_worker, daemon=True).start()

    def _on_ocr_done(self, results: List[FrameResult], tracking_results: Dict[int, ROI]) -> None:
        self._results = results
        self._tracking_results = tracking_results
        self._progress_var.set(100)
        self._status_var.set(f"OCR完了: {len(results)} フレーム処理")
        self._listbox.delete(0, tk.END)
        for r in results:
            self._listbox.insert(tk.END, f"{r.frame_no:>6}: {r.value}")
        self._set_ui_ocr_mode(running=False)
        self._btn_csv.config(state=tk.NORMAL)
        self._btn_json.config(state=tk.NORMAL)
        self._show_frame(self._current_frame_no)

    def _set_ui_ocr_mode(self, running: bool) -> None:
        self._ocr_running = running
        state = tk.DISABLED if running else tk.NORMAL
        self._btn_open.config(state=state)
        self._btn_load_ocr_csv.config(state=state if self._video_reader else tk.DISABLED)
        self._btn_ocr.config(state=state)
        self._slider.config(state=state)
        if running:
            self._progressbar.config(mode="determinate")
        else:
            self._progressbar.config(mode="determinate")

    # ------------------------------------------------------------------ #
    # OCR結果CSV読み込み
    # ------------------------------------------------------------------ #

    def _load_ocr_csv(self) -> None:
        if self._video_reader is None:
            return
        path = filedialog.askopenfilename(
            title="過去のOCR結果CSVを選択",
            filetypes=[("CSVファイル", "*.csv"), ("すべて", "*.*")],
        )
        if not path:
            return
        try:
            with open(path, newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                rows = list(reader)
        except OSError as e:
            messagebox.showerror("読み込みエラー", str(e))
            return

        if not rows or "frame" not in rows[0] or "value" not in rows[0]:
            messagebox.showerror(
                "フォーマットエラー",
                "CSVに 'frame' と 'value' 列が必要です。",
            )
            return

        # frame番号をキーにしたマップを作成
        csv_map: Dict[int, Optional[int]] = {}
        for row in rows:
            try:
                frame_no = int(row["frame"])
            except (ValueError, KeyError):
                continue
            raw = row["value"].strip()
            try:
                csv_map[frame_no] = int(raw) if raw else None
            except ValueError:
                csv_map[frame_no] = None

        total = self._video_reader.frame_count
        if len(csv_map) != total:
            proceed = messagebox.askyesno(
                "フレーム数不一致",
                f"CSV のフレーム数 ({len(csv_map)}) と動画のフレーム数 ({total}) が異なります。\n読み込みを続けますか？",
            )
            if not proceed:
                return

        self._results = [
            FrameResult(frame_no=i, value=csv_map.get(i))
            for i in range(total)
        ]
        self._tracking_results = {}

        self._listbox.delete(0, tk.END)
        for r in self._results:
            self._listbox.insert(tk.END, f"{r.frame_no:>6}: {r.value}")

        self._btn_csv.config(state=tk.NORMAL)
        self._btn_json.config(state=tk.DISABLED)
        self._status_var.set(
            f"OCR結果を読み込みました: {Path(path).name}  ({len(self._results)} フレーム)"
        )
        self._show_frame(self._current_frame_no)
        self._sync_listbox(self._current_frame_no)

    # ------------------------------------------------------------------ #
    # CSV保存
    # ------------------------------------------------------------------ #

    def _save_csv(self) -> None:
        if not self._results:
            messagebox.showwarning("警告", "OCR結果がありません。先にOCRを実行してください。")
            return
        path = filedialog.asksaveasfilename(
            title="CSVファイルを保存",
            defaultextension=".csv",
            filetypes=[("CSVファイル", "*.csv"), ("すべて", "*.*")],
        )
        if not path:
            return
        try:
            write_csv(path, self._results)
            messagebox.showinfo("保存完了", f"CSVを保存しました:\n{path}")
        except OSError as e:
            messagebox.showerror("保存エラー", str(e))

    # ------------------------------------------------------------------ #
    # JSON保存（トラッキング結果）
    # ------------------------------------------------------------------ #

    def _save_json(self) -> None:
        if not self._tracking_results:
            messagebox.showwarning("警告", "トラッキング結果がありません。先にOCRを実行してください。")
            return
        path = filedialog.asksaveasfilename(
            title="JSONファイルを保存",
            defaultextension=".json",
            filetypes=[("JSONファイル", "*.json"), ("すべて", "*.*")],
        )
        if not path:
            return
        try:
            write_tracking_json(path, self._tracking_results)
            messagebox.showinfo("保存完了", f"JSONを保存しました:\n{path}")
        except OSError as e:
            messagebox.showerror("保存エラー", str(e))

    # ================================================================== #
    # 解析・統計タブ操作
    # ================================================================== #

    def _load_analysis_csv(self) -> None:
        path = filedialog.askopenfilename(
            title="GPS/時刻ログCSVを選択",
            filetypes=[("CSVファイル", "*.csv"), ("すべて", "*.*")],
        )
        if not path:
            return
        self._analysis_csv_path = path
        self._analysis_csv_label.config(text=path, fg="black")

        # 先頭行の timestamp を start_timestamp の初期値として自動設定する
        try:
            rows = self._read_csv_as_dicts(path)
            if rows and "timestamp" in rows[0]:
                self._start_ts_var.set(str(rows[0]["timestamp"]))
        except (OSError, ValueError):
            pass

    # ------------------------------------------------------------------ #
    # 処理速度設定のロード／セーブ
    # ------------------------------------------------------------------ #

    def _load_perf_config(self) -> None:
        """performance_settings.json を読み込んで各状態変数に反映する"""
        perf_path = Path(__file__).parent.parent / "settings" / "performance_settings.json"
        try:
            with open(perf_path, encoding="utf-8") as f:
                cfg = json.load(f)
            self._use_gpu.set(bool(cfg.get("use_gpu", False)))
            self._frame_skip_var.set(max(1, int(cfg.get("frame_skip", 1))))
            self._use_sequential_read.set(bool(cfg.get("use_sequential_read", True)))
            self._use_process_parallel.set(bool(cfg.get("use_process_parallel", False)))
            self._tracking_reinit.set(bool(cfg.get("tracking_reinit_on_failure", True)))
            self._tracking_drift_detection.set(bool(cfg.get("tracking_drift_detection", False)))
            self._tracking_drift_factor_var.set(str(cfg.get("tracking_drift_factor", 1.5)))
            self._tracking_template_matching.set(bool(cfg.get("tracking_template_matching", False)))
            self._tracking_template_threshold_var.set(str(cfg.get("tracking_template_threshold", 0.5)))
            self._tracking_template_margin_var.set(str(cfg.get("tracking_template_margin_factor", 0.5)))
            self._segment_tracking_enabled.set(bool(cfg.get("segment_tracking_enabled", False)))
            self._segment_interval_var.set(max(1, int(cfg.get("segment_interval", 180))))
        except (OSError, json.JSONDecodeError):
            pass  # ファイルがない場合はデフォルト値のまま

    def _save_perf_config(self) -> None:
        """現在の処理速度・トラッキング設定を performance_settings.json に保存する"""
        perf_path = Path(__file__).parent.parent / "settings" / "performance_settings.json"
        try:
            cfg = {
                "use_gpu": self._use_gpu.get(),
                "frame_skip": max(1, self._frame_skip_var.get()),
                "use_sequential_read": self._use_sequential_read.get(),
                "use_process_parallel": self._use_process_parallel.get(),
                "tracking_reinit_on_failure": self._tracking_reinit.get(),
                "tracking_drift_detection": self._tracking_drift_detection.get(),
                "tracking_drift_factor": float(self._tracking_drift_factor_var.get()),
                "tracking_template_matching": self._tracking_template_matching.get(),
                "tracking_template_threshold": float(self._tracking_template_threshold_var.get()),
                "tracking_template_margin_factor": float(self._tracking_template_margin_var.get()),
                "segment_tracking_enabled": self._segment_tracking_enabled.get(),
                "segment_interval": self._segment_interval_var.get(),
            }
            with open(perf_path, "w", encoding="utf-8") as f:
                json.dump(cfg, f, indent=4, ensure_ascii=False)
        except (OSError, ValueError):
            pass

    # ------------------------------------------------------------------ #
    # セグメント分割トラッキング
    # ------------------------------------------------------------------ #

    def _update_segment_interval_label(self) -> None:
        """セグメント間隔ラベルをFPS換算値で更新する"""
        if self._video_reader is None:
            self._segment_interval_label.config(text="")
            return
        try:
            iv = self._segment_interval_var.get()
            fps = self._video_reader.fps
            secs = iv / fps if fps > 0 else 0
            total = self._video_reader.frame_count
            n_segs = (total + iv - 1) // iv if iv > 0 else 0
            self._segment_interval_label.config(
                text=f"≈ {secs:.1f}秒  {n_segs}分割"
            )
        except Exception:
            self._segment_interval_label.config(text="")

    def _update_segment_roi_count_label(self) -> None:
        """設定済みセグメントROI数のラベルを更新する"""
        count = len(self._segment_rois)
        if count > 0:
            self._segment_roi_count_label.config(
                text=f"ROI設定済み: {count} セグメント", fg="#006600"
            )
        else:
            self._segment_roi_count_label.config(
                text="ROI設定済み: 0 セグメント", fg="gray"
            )

    def _open_segment_roi_dialog(self) -> None:
        """セグメント分割ROI設定ダイアログを開く"""
        if self._video_reader is None:
            messagebox.showwarning("警告", "先に動画ファイルを開いてください。")
            return
        if self._segment_dialog is not None and self._segment_dialog.winfo_exists():
            self._segment_dialog.lift()
            return

        fps = self._video_reader.fps
        total = self._video_reader.frame_count

        dlg = tk.Toplevel(self)
        dlg.title("セグメントROI設定")
        dlg.geometry("600x420")
        dlg.resizable(True, True)
        self._segment_dialog = dlg

        # 上部: 動画情報と間隔設定
        top = tk.Frame(dlg, padx=8, pady=6)
        top.pack(fill=tk.X)
        tk.Label(
            top,
            text=f"動画情報: {total} フレーム  FPS: {fps:.2f}",
            anchor=tk.W,
        ).pack(anchor=tk.W)

        row = tk.Frame(top)
        row.pack(fill=tk.X, pady=4)
        tk.Label(row, text="セグメント間隔（フレーム数）:").pack(side=tk.LEFT)
        dlg_interval_var = tk.IntVar(value=self._segment_interval_var.get())
        tk.Spinbox(row, from_=1, to=total, width=8, textvariable=dlg_interval_var).pack(
            side=tk.LEFT, padx=4
        )
        dlg_interval_info = tk.Label(row, text="", fg="#555555")
        dlg_interval_info.pack(side=tk.LEFT, padx=4)

        def _update_dlg_interval_info(*_):
            try:
                iv = dlg_interval_var.get()
                secs = iv / fps if fps > 0 else 0
                n = (total + iv - 1) // iv if iv > 0 else 0
                dlg_interval_info.config(text=f"≈ {secs:.1f}秒/セグメント  合計 {n} セグメント")
            except Exception:
                pass

        dlg_interval_var.trace_add("write", _update_dlg_interval_info)
        _update_dlg_interval_info()

        def _apply_interval():
            self._segment_interval_var.set(dlg_interval_var.get())
            self._save_perf_config()
            self._update_segment_interval_label()
            _refresh_list()

        tk.Button(row, text="適用", command=_apply_interval).pack(side=tk.LEFT, padx=4)

        ttk.Separator(dlg, orient=tk.HORIZONTAL).pack(fill=tk.X, padx=8, pady=2)

        # 中央: セグメント一覧
        list_frame = tk.Frame(dlg, padx=8)
        list_frame.pack(fill=tk.BOTH, expand=True)

        columns = ("segment", "start_frame", "time", "roi_status")
        tree = ttk.Treeview(list_frame, columns=columns, show="headings", height=12)
        tree.heading("segment", text="セグメント")
        tree.heading("start_frame", text="開始フレーム")
        tree.heading("time", text="時刻")
        tree.heading("roi_status", text="ROI状態")
        tree.column("segment", width=80, anchor=tk.CENTER)
        tree.column("start_frame", width=100, anchor=tk.CENTER)
        tree.column("time", width=90, anchor=tk.CENTER)
        tree.column("roi_status", width=290, anchor=tk.W)

        vsb = ttk.Scrollbar(list_frame, orient=tk.VERTICAL, command=tree.yview)
        tree.configure(yscrollcommand=vsb.set)
        tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        vsb.pack(side=tk.RIGHT, fill=tk.Y)

        def _refresh_list():
            tree.delete(*tree.get_children())
            iv = self._segment_interval_var.get()
            for seg_idx, start in enumerate(range(0, total, iv)):
                secs = start / fps if fps > 0 else 0
                if start == 0:
                    if self._roi:
                        status = f"メインROI ({self._roi.x},{self._roi.y} {self._roi.w}×{self._roi.h})"
                        tag = "has_roi"
                    else:
                        status = "メインROI 未設定"
                        tag = "no_roi"
                elif start in self._segment_rois:
                    r = self._segment_rois[start]
                    status = f"設定済み ✓ ({r.x},{r.y} {r.w}×{r.h})"
                    tag = "has_roi"
                else:
                    status = "未設定（直前ROI引継ぎ）"
                    tag = "no_roi"
                tree.insert(
                    "", tk.END, iid=str(start),
                    values=(f"Seg {seg_idx}", str(start), f"{secs:.1f}秒", status),
                    tags=(tag,),
                )
            tree.tag_configure("has_roi", foreground="#006600")
            tree.tag_configure("no_roi", foreground="#888888")

        self._seg_list_refresh = _refresh_list
        _refresh_list()

        ttk.Separator(dlg, orient=tk.HORIZONTAL).pack(fill=tk.X, padx=8, pady=2)

        # 下部: 操作ボタン
        btn_row = tk.Frame(dlg, padx=8, pady=6)
        btn_row.pack(fill=tk.X)

        def _set_roi():
            sel = tree.selection()
            if not sel:
                messagebox.showwarning("未選択", "セグメントを選択してください。", parent=dlg)
                return
            start_frame = int(sel[0])
            if start_frame == 0:
                messagebox.showinfo(
                    "情報",
                    "セグメント0（先頭）のROIはメイン画面でドラッグして設定してください。",
                    parent=dlg,
                )
                return
            self._go_to_frame(start_frame)
            self._segment_edit_mode = start_frame
            iv = self._segment_interval_var.get()
            seg_idx = start_frame // iv
            self._status_var.set(
                f"[Seg {seg_idx} / フレーム {start_frame}] "
                f"ROIをシアン枠でドラッグして設定してください"
            )
            dlg.lift()

        def _clear_roi():
            sel = tree.selection()
            if not sel:
                messagebox.showwarning("未選択", "セグメントを選択してください。", parent=dlg)
                return
            start_frame = int(sel[0])
            if start_frame == 0:
                messagebox.showinfo(
                    "情報",
                    "セグメント0のROIはメインROIです。メイン画面で変更してください。",
                    parent=dlg,
                )
                return
            if start_frame in self._segment_rois:
                del self._segment_rois[start_frame]
                self._update_segment_roi_count_label()
                _refresh_list()

        def _clear_all():
            if messagebox.askyesno("確認", "すべてのセグメントROI設定をクリアしますか？", parent=dlg):
                self._segment_rois.clear()
                self._update_segment_roi_count_label()
                _refresh_list()

        tk.Button(btn_row, text="このセグメントのROIを設定", command=_set_roi, width=22).pack(
            side=tk.LEFT, padx=4
        )
        tk.Button(btn_row, text="このセグメントのROIをクリア", command=_clear_roi, width=24).pack(
            side=tk.LEFT, padx=4
        )
        tk.Button(btn_row, text="全ROIクリア", command=_clear_all, width=12).pack(
            side=tk.LEFT, padx=4
        )
        tk.Button(btn_row, text="閉じる", command=dlg.destroy, width=10).pack(
            side=tk.RIGHT, padx=4
        )

        dlg.protocol("WM_DELETE_WINDOW", dlg.destroy)

    def _load_default_color_config(self) -> None:
        default_path = Path(__file__).parent.parent / "settings" / "color_settings.json"
        try:
            with open(default_path, encoding="utf-8") as f:
                self._color_config = json.load(f)
            self._color_config_path = str(default_path)
            self._color_config_label.config(
                text=f"{default_path.name}（デフォルト）", fg="#555555"
            )
        except (OSError, json.JSONDecodeError):
            pass

    def _load_color_config(self) -> None:
        path = filedialog.askopenfilename(
            title="色設定JSONファイルを選択",
            filetypes=[("JSONファイル", "*.json"), ("すべて", "*.*")],
        )
        if not path:
            return
        try:
            with open(path, encoding="utf-8") as f:
                self._color_config = json.load(f)
            self._color_config_path = path
            self._color_config_label.config(text=path, fg="black")
        except (OSError, json.JSONDecodeError) as e:
            messagebox.showerror("読み込みエラー", f"色設定ファイルの読み込みに失敗しました:\n{e}")

    def _run_analysis(self) -> None:
        if not self._results:
            messagebox.showwarning("警告", "先にOCR解析タブでOCRを実行してください。")
            return
        if self._analysis_csv_path is None:
            messagebox.showwarning("警告", "GPS/時刻ログCSVを選択してください。")
            return
        if self._video_reader is None:
            messagebox.showwarning("警告", "動画が読み込まれていません。")
            return

        try:
            start_ts = float(self._start_ts_var.get())
        except ValueError:
            messagebox.showerror("入力エラー", "動画開始タイムスタンプに有効な数値を入力してください。")
            return

        try:
            gps_rows = self._read_csv_as_dicts(self._analysis_csv_path)
        except (OSError, ValueError) as e:
            messagebox.showerror("CSVエラー", str(e))
            return

        for col in ("lat", "lon", "timestamp"):
            if col not in gps_rows[0]:
                messagebox.showerror("CSVエラー", f"必須カラム '{col}' がCSVに存在しません。")
                return

        use_case = AnalyzeUseCase()
        self._analysis_rows = use_case.execute(
            ocr_results=self._results,
            fps=self._video_reader.fps,
            start_timestamp=start_ts,
            gps_rows=gps_rows,
        )

        matched = sum(1 for r in self._analysis_rows if r.get("ocr_result") is not None)
        self._analysis_status_label.config(
            text=f"解析完了: {len(self._analysis_rows)} 行 / OCR対応: {matched} 行",
            fg="green",
        )
        self._btn_export.config(state=tk.NORMAL)
        self._btn_open_map.config(state=tk.NORMAL)
        self._update_map_display()

    def _on_map_view_change(self) -> None:
        if self._analysis_rows:
            self._update_map_display()

    def _update_map_display(self) -> None:
        if self._map_canvas_agg is not None:
            self._map_canvas_agg.get_tk_widget().destroy()
            self._map_canvas_agg = None
        if self._map_figure is not None:
            self._map_figure.clf()
            self._map_figure = None
        if self._map_placeholder.winfo_exists():
            self._map_placeholder.pack_forget()

        if self._use_map_tiles.get():
            self._map_figure = render_route_map_with_tiles(
                self._analysis_rows, self._color_config
            )
        else:
            self._map_figure = render_route_map(self._analysis_rows, self._color_config)

        self._map_canvas_agg = FigureCanvasTkAgg(self._map_figure, master=self._map_frame)
        self._map_canvas_agg.draw()
        self._map_canvas_agg.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def _open_map_in_browser(self) -> None:
        if not self._analysis_rows:
            return
        try:
            tmp_path = save_folium_map_to_tempfile(self._analysis_rows, self._color_config)
            webbrowser.open(f"file:///{tmp_path}")
        except Exception as e:
            messagebox.showerror("エラー", f"地図の表示に失敗しました:\n{e}")

    def _export_results(self) -> None:
        if not self._analysis_rows:
            messagebox.showwarning("警告", "解析結果がありません。先に解析を実行してください。")
            return

        out_dir = filedialog.askdirectory(title="出力先フォルダを選択")
        if not out_dir:
            return

        csv_path = os.path.join(out_dir, "result.csv")
        map_png_path = os.path.join(out_dir, "map.png")
        map_html_path = os.path.join(out_dir, "map.html")

        try:
            write_result_csv(csv_path, self._analysis_rows)
        except OSError as e:
            messagebox.showerror("保存エラー", f"CSVの保存に失敗しました:\n{e}")
            return

        try:
            fig = render_route_map(self._analysis_rows, self._color_config)
            write_map_image(map_png_path, fig)
            fig.clf()
        except Exception as e:
            messagebox.showerror("保存エラー", f"マップ画像の保存に失敗しました:\n{e}")
            return

        try:
            write_map_html(map_html_path, self._analysis_rows, self._color_config)
        except Exception as e:
            messagebox.showerror("保存エラー", f"地図HTMLの保存に失敗しました:\n{e}")
            return

        messagebox.showinfo(
            "出力完了",
            f"保存しました:\n  {csv_path}\n  {map_png_path}\n  {map_html_path}",
        )

    @staticmethod
    def _read_csv_as_dicts(path: str) -> List[Dict[str, Any]]:
        with open(path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = [dict(row) for row in reader]
        if not rows:
            raise ValueError("CSVにデータが存在しません。")
        return rows
