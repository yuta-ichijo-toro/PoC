"""解析結果（CSV・マップ画像）の出力"""
import csv
from typing import Any, Dict, List

from matplotlib.figure import Figure


def write_result_csv(path: str, rows: List[Dict[str, Any]]) -> None:
    """rows をそのままCSVに書き出す（カラム順は rows[0] のキー順）。"""
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_map_image(path: str, fig: Figure) -> None:
    """Figure をPNG画像として保存する。"""
    fig.savefig(path, dpi=150, bbox_inches="tight")


def write_map_html(path: str, rows: List[Dict[str, Any]], color_config: Dict[str, str]) -> None:
    """foliumマップをHTMLファイルとして保存する。"""
    from .map_renderer import save_folium_map_to_tempfile
    import shutil

    tmp_path = save_folium_map_to_tempfile(rows, color_config)
    shutil.move(tmp_path, path)
