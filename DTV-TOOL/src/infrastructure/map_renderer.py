"""マップ描画（matplotlib / folium）"""
import tempfile
from typing import Dict, List, Optional

from matplotlib.figure import Figure
from matplotlib.lines import Line2D

_DEFAULT_OTHER_COLOR = "#888888"


def _resolve_color(result: Optional[str], color_config: Dict[str, str]) -> str:
    if result is not None and result in color_config:
        return color_config[result]
    return color_config.get("Other", _DEFAULT_OTHER_COLOR)


def render_route_map(
    rows: List[Dict],
    color_config: Dict[str, str],
) -> Figure:
    """lat/lon の点を線で結び OCR結果で色分けしたルートマップを返す。"""
    fig = Figure(figsize=(8, 6))
    ax = fig.add_subplot(111)

    lats = [float(r["lat"]) for r in rows]
    lons = [float(r["lon"]) for r in rows]

    used: Dict[str, str] = {}
    for i in range(len(rows) - 1):
        result = rows[i].get("ocr_result")
        result_str = str(result) if result is not None else None
        color = _resolve_color(result_str, color_config)
        label = result_str if result_str is not None else "Other"
        used[label] = color
        ax.plot(
            [lons[i], lons[i + 1]],
            [lats[i], lats[i + 1]],
            color=color,
            linewidth=2,
        )

    ax.scatter(lons, lats, color="black", s=15, zorder=5)

    legend_elements = [
        Line2D([0], [0], color=c, linewidth=2, label=lbl)
        for lbl, c in sorted(used.items())
    ]
    if legend_elements:
        ax.legend(handles=legend_elements, loc="best", fontsize=8)

    ax.set_xlabel("経度 (Longitude)")
    ax.set_ylabel("緯度 (Latitude)")
    ax.set_title("ルートマップ")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def render_route_map_with_tiles(
    rows: List[Dict],
    color_config: Dict[str, str],
) -> Figure:
    """OSMタイルを背景に加えたルートマップを返す。タイル取得失敗時は通常マップにフォールバック。"""
    try:
        import contextily as ctx
    except ImportError:
        return render_route_map(rows, color_config)

    fig = Figure(figsize=(8, 6))
    ax = fig.add_subplot(111)

    lats = [float(r["lat"]) for r in rows]
    lons = [float(r["lon"]) for r in rows]

    used: Dict[str, str] = {}
    for i in range(len(rows) - 1):
        result = rows[i].get("ocr_result")
        result_str = str(result) if result is not None else None
        color = _resolve_color(result_str, color_config)
        label = result_str if result_str is not None else "Other"
        used[label] = color
        ax.plot(
            [lons[i], lons[i + 1]],
            [lats[i], lats[i + 1]],
            color=color,
            linewidth=2,
            zorder=3,
        )

    ax.scatter(lons, lats, color="black", s=15, zorder=5)

    legend_elements = [
        Line2D([0], [0], color=c, linewidth=2, label=lbl)
        for lbl, c in sorted(used.items())
    ]
    if legend_elements:
        ax.legend(handles=legend_elements, loc="best", fontsize=8)

    ax.set_xlabel("経度 (Longitude)")
    ax.set_ylabel("緯度 (Latitude)")
    ax.set_title("ルートマップ（地図タイル）")

    try:
        ctx.add_basemap(
            ax,
            crs="EPSG:4326",
            source=ctx.providers.OpenStreetMap.Mapnik,
            zoom="auto",
        )
    except Exception:
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    return fig


def render_folium_map(rows: List[Dict], color_config: Dict[str, str]) -> str:
    """foliumを使ってインタラクティブなHTMLマップを生成し、HTML文字列を返す。"""
    import folium

    lats = [float(r["lat"]) for r in rows]
    lons = [float(r["lon"]) for r in rows]
    center_lat = sum(lats) / len(lats) if lats else 0.0
    center_lon = sum(lons) / len(lons) if lons else 0.0

    m = folium.Map(location=[center_lat, center_lon], zoom_start=15)

    for i in range(len(rows) - 1):
        result = rows[i].get("ocr_result")
        result_str = str(result) if result is not None else None
        color = _resolve_color(result_str, color_config)
        folium.PolyLine(
            locations=[[lats[i], lons[i]], [lats[i + 1], lons[i + 1]]],
            color=color,
            weight=4,
            opacity=0.85,
        ).add_to(m)

    for row, lat, lon in zip(rows, lats, lons):
        result = row.get("ocr_result")
        result_str = str(result) if result is not None else "None"
        color = _resolve_color(str(result) if result is not None else None, color_config)
        folium.CircleMarker(
            location=[lat, lon],
            radius=5,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.9,
            popup=folium.Popup(
                f"lat={lat:.6f}, lon={lon:.6f}<br>OCR: {result_str}", max_width=200
            ),
        ).add_to(m)

    return m._repr_html_()


def save_folium_map_to_tempfile(rows: List[Dict], color_config: Dict[str, str]) -> str:
    """foliumマップをHTMLファイルとして一時ファイルに保存し、そのパスを返す。"""
    import folium

    lats = [float(r["lat"]) for r in rows]
    lons = [float(r["lon"]) for r in rows]
    center_lat = sum(lats) / len(lats) if lats else 0.0
    center_lon = sum(lons) / len(lons) if lons else 0.0

    m = folium.Map(location=[center_lat, center_lon], zoom_start=15)

    for i in range(len(rows) - 1):
        result = rows[i].get("ocr_result")
        result_str = str(result) if result is not None else None
        color = _resolve_color(result_str, color_config)
        folium.PolyLine(
            locations=[[lats[i], lons[i]], [lats[i + 1], lons[i + 1]]],
            color=color,
            weight=4,
            opacity=0.85,
        ).add_to(m)

    for row, lat, lon in zip(rows, lats, lons):
        result = row.get("ocr_result")
        result_str = str(result) if result is not None else "None"
        color = _resolve_color(str(result) if result is not None else None, color_config)
        folium.CircleMarker(
            location=[lat, lon],
            radius=5,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.9,
            popup=folium.Popup(
                f"lat={lat:.6f}, lon={lon:.6f}<br>OCR: {result_str}", max_width=200
            ),
        ).add_to(m)

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".html", delete=False, encoding="utf-8"
    ) as f:
        m.save(f.name)
        return f.name
