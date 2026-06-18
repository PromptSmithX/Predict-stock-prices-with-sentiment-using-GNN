"""Export slide-ready visualizations for daily stock-sector HGT graphs."""

from __future__ import annotations

import argparse
import html
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.features.graph_builder import build_daily_graphs  # noqa: E402


DEFAULT_STOCK_INPUT_PATH = PROJECT_ROOT / "data" / "processed" / "stock_node_features.csv"
DEFAULT_SECTOR_INPUT_PATH = PROJECT_ROOT / "data" / "processed" / "sector_feature_data.csv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "reports" / "figures"

DEFAULT_WIDTH = 1600
DEFAULT_HEIGHT = 1000
POSITIVE_CORR_COLOR = "#168363"
NEGATIVE_CORR_COLOR = "#c74343"
MEMBERSHIP_EDGE_COLOR = "#8c97a3"
STOCK_FILL = "#ffffff"
TEXT_COLOR = "#263238"
MUTED_TEXT_COLOR = "#667085"
BACKGROUND_COLOR = "#ffffff"
SECTOR_COLORS = {
    "Energy": "#f1b44c",
    "Financials": "#6172d8",
    "Healthcare": "#2fa6a0",
    "Technology": "#4f9be8",
}
FALLBACK_SECTOR_COLORS = [
    "#4f9be8",
    "#2fa6a0",
    "#6172d8",
    "#f1b44c",
    "#8a6fd1",
    "#db6f57",
]


@dataclass(frozen=True)
class VisualNode:
    """Drawable node extracted from a HeteroData graph."""

    node_id: str
    label: str
    node_type: str
    sector: str
    x: float
    y: float
    color: str


@dataclass(frozen=True)
class VisualEdge:
    """Drawable edge extracted from a HeteroData graph."""

    source: str
    target: str
    edge_type: str
    color: str
    width: float
    corr: float | None = None
    abs_corr: float | None = None


@dataclass(frozen=True)
class GraphVisualSpec:
    """Graph visualization payload independent from rendering backend."""

    date: str
    nodes: list[VisualNode]
    edges: list[VisualEdge]
    width: int = DEFAULT_WIDTH
    height: int = DEFAULT_HEIGHT
    title: str = "Daily Heterogeneous Stock-Sector Graph"


def _to_list(value: Any) -> list[Any]:
    if hasattr(value, "detach"):
        return value.detach().cpu().tolist()
    if hasattr(value, "tolist"):
        return value.tolist()
    return list(value)


def _edge_index_columns(edge_index: Any) -> list[tuple[int, int]]:
    values = _to_list(edge_index)
    if not values:
        return []
    if len(values) != 2:
        raise ValueError("edge_index must have shape [2, num_edges].")
    return [
        (int(source), int(target))
        for source, target in zip(values[0], values[1])
    ]


def _sector_color(sector: str, index: int) -> str:
    return SECTOR_COLORS.get(
        sector,
        FALLBACK_SECTOR_COLORS[index % len(FALLBACK_SECTOR_COLORS)],
    )


def _sector_centers(
    sectors: list[str],
    width: int,
    height: int,
) -> dict[str, tuple[float, float]]:
    if not sectors:
        return {}

    if len(sectors) <= 4:
        spacing = width / (len(sectors) + 1)
        return {
            sector: (spacing * (index + 1), height * 0.53)
            for index, sector in enumerate(sectors)
        }

    columns = min(4, math.ceil(math.sqrt(len(sectors))))
    rows = math.ceil(len(sectors) / columns)
    x_spacing = width / (columns + 1)
    y_spacing = (height * 0.55) / max(rows, 1)
    y_start = height * 0.32
    centers = {}
    for index, sector in enumerate(sectors):
        row = index // columns
        column = index % columns
        centers[sector] = (
            x_spacing * (column + 1),
            y_start + y_spacing * row,
        )
    return centers


def _stock_offsets(count: int, radius: float = 125.0) -> list[tuple[float, float]]:
    presets = {
        1: [(0.0, -radius)],
        2: [(-radius * 0.65, -radius * 0.85), (radius * 0.65, -radius * 0.85)],
        3: [(-radius * 0.75, -radius * 0.8), (radius * 0.75, -radius * 0.8), (0.0, radius)],
        4: [
            (-radius * 0.85, -radius * 0.8),
            (radius * 0.85, -radius * 0.8),
            (-radius * 0.85, radius * 0.75),
            (radius * 0.85, radius * 0.75),
        ],
    }
    if count in presets:
        return presets[count]

    return [
        (
            math.cos((2 * math.pi * index / count) - math.pi / 2) * radius,
            math.sin((2 * math.pi * index / count) - math.pi / 2) * radius,
        )
        for index in range(count)
    ]


def _graph_date(graph: Any) -> str:
    return str(getattr(graph, "date", "unknown"))


def _graph_sectors(graph: Any) -> list[str]:
    if hasattr(graph, "sectors"):
        return [str(sector) for sector in graph.sectors]
    return [str(sector) for sector in graph["industry"].sectors]


def _graph_tickers(graph: Any) -> list[str]:
    if hasattr(graph, "tickers"):
        return [str(ticker) for ticker in graph.tickers]
    return [str(ticker) for ticker in graph["stock"].tickers]


def extract_graph_visual_spec(
    graph: Any,
    width: int = DEFAULT_WIDTH,
    height: int = DEFAULT_HEIGHT,
) -> GraphVisualSpec:
    """Extract a renderable graph spec from a daily HeteroData graph."""
    tickers = _graph_tickers(graph)
    sectors = _graph_sectors(graph)
    sector_ids = [int(value) for value in _to_list(graph["stock"].sector_id)]
    if len(tickers) != len(sector_ids):
        raise ValueError("Ticker and sector_id counts do not match.")

    stock_to_sector = {
        ticker: sectors[sector_ids[index]]
        for index, ticker in enumerate(tickers)
    }
    sector_to_tickers: dict[str, list[str]] = {sector: [] for sector in sectors}
    for ticker in tickers:
        sector_to_tickers.setdefault(stock_to_sector[ticker], []).append(ticker)

    centers = _sector_centers(sectors, width=width, height=height)
    nodes: list[VisualNode] = []
    for sector_index, sector in enumerate(sectors):
        x, y = centers[sector]
        nodes.append(
            VisualNode(
                node_id=f"sector:{sector}",
                label=sector,
                node_type="sector",
                sector=sector,
                x=x,
                y=y,
                color=_sector_color(sector, sector_index),
            )
        )

        sector_tickers = sector_to_tickers.get(sector, [])
        for offset, ticker in zip(_stock_offsets(len(sector_tickers)), sector_tickers):
            nodes.append(
                VisualNode(
                    node_id=f"stock:{ticker}",
                    label=ticker,
                    node_type="stock",
                    sector=sector,
                    x=x + offset[0],
                    y=y + offset[1],
                    color=_sector_color(sector, sector_index),
                )
            )

    edges: list[VisualEdge] = []
    for ticker in tickers:
        sector = stock_to_sector[ticker]
        edges.append(
            VisualEdge(
                source=f"stock:{ticker}",
                target=f"sector:{sector}",
                edge_type="membership",
                color=MEMBERSHIP_EDGE_COLOR,
                width=2.0,
            )
        )

    corr_edge_index = graph["stock", "corr", "stock"].edge_index
    corr_edge_attr = graph["stock", "corr", "stock"].edge_attr
    corr_pairs: dict[tuple[int, int], tuple[float, float]] = {}
    for (source, target), attrs in zip(
        _edge_index_columns(corr_edge_index),
        _to_list(corr_edge_attr),
    ):
        if source == target:
            continue
        key = tuple(sorted((source, target)))
        corr = float(attrs[0])
        abs_corr = float(attrs[1])
        existing = corr_pairs.get(key)
        if existing is None or abs_corr > existing[1]:
            corr_pairs[key] = (corr, abs_corr)

    for (source, target), (corr, abs_corr) in sorted(corr_pairs.items()):
        edges.append(
            VisualEdge(
                source=f"stock:{tickers[source]}",
                target=f"stock:{tickers[target]}",
                edge_type="correlation",
                color=POSITIVE_CORR_COLOR if corr >= 0 else NEGATIVE_CORR_COLOR,
                width=1.2 + 4.0 * min(abs_corr, 1.0),
                corr=corr,
                abs_corr=abs_corr,
            )
        )

    return GraphVisualSpec(
        date=_graph_date(graph),
        nodes=nodes,
        edges=edges,
        width=width,
        height=height,
    )


def _node_lookup(spec: GraphVisualSpec) -> dict[str, VisualNode]:
    return {node.node_id: node for node in spec.nodes}


def _curved_path(
    source: VisualNode,
    target: VisualNode,
    curve_offset: float = 42.0,
) -> tuple[float, float, float, float, float, float]:
    x1, y1 = source.x, source.y
    x2, y2 = target.x, target.y
    mid_x = (x1 + x2) / 2.0
    mid_y = (y1 + y2) / 2.0
    dx = x2 - x1
    dy = y2 - y1
    length = math.hypot(dx, dy) or 1.0
    normal_x = -dy / length
    normal_y = dx / length
    control_x = mid_x + normal_x * curve_offset
    control_y = mid_y + normal_y * curve_offset
    return x1, y1, control_x, control_y, x2, y2


def _quadratic_points(
    x1: float,
    y1: float,
    cx: float,
    cy: float,
    x2: float,
    y2: float,
    steps: int = 24,
) -> list[tuple[float, float]]:
    points = []
    for step in range(steps + 1):
        t = step / steps
        one_minus_t = 1.0 - t
        x = one_minus_t * one_minus_t * x1 + 2 * one_minus_t * t * cx + t * t * x2
        y = one_minus_t * one_minus_t * y1 + 2 * one_minus_t * t * cy + t * t * y2
        points.append((x, y))
    return points


def _svg_text(
    text: str,
    x: float,
    y: float,
    size: int,
    fill: str = TEXT_COLOR,
    anchor: str = "middle",
    weight: int = 500,
) -> str:
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" text-anchor="{anchor}" '
        f'font-family="Segoe UI, Arial, sans-serif" font-size="{size}" '
        f'font-weight="{weight}" fill="{fill}">{html.escape(text)}</text>'
    )


def render_svg(spec: GraphVisualSpec) -> str:
    """Render a graph spec to an SVG string."""
    nodes = _node_lookup(spec)
    membership_edges = [edge for edge in spec.edges if edge.edge_type == "membership"]
    corr_edges = [edge for edge in spec.edges if edge.edge_type == "correlation"]

    parts = [
        (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{spec.width}" '
            f'height="{spec.height}" viewBox="0 0 {spec.width} {spec.height}">'
        ),
        f'<rect width="100%" height="100%" fill="{BACKGROUND_COLOR}"/>',
        _svg_text(spec.title, spec.width / 2, 58, 34, weight=700),
        _svg_text(
            f"Graph date: {spec.date}",
            spec.width / 2,
            94,
            18,
            fill=MUTED_TEXT_COLOR,
            weight=500,
        ),
    ]

    for edge in membership_edges:
        source = nodes[edge.source]
        target = nodes[edge.target]
        parts.append(
            f'<line x1="{source.x:.1f}" y1="{source.y:.1f}" '
            f'x2="{target.x:.1f}" y2="{target.y:.1f}" '
            f'stroke="{edge.color}" stroke-width="{edge.width:.1f}" '
            'stroke-opacity="0.42"/>'
        )

    for edge in corr_edges:
        source = nodes[edge.source]
        target = nodes[edge.target]
        x1, y1, cx, cy, x2, y2 = _curved_path(source, target)
        parts.append(
            f'<path d="M {x1:.1f} {y1:.1f} Q {cx:.1f} {cy:.1f} {x2:.1f} {y2:.1f}" '
            f'fill="none" stroke="{edge.color}" stroke-width="{edge.width:.1f}" '
            'stroke-opacity="0.74" stroke-linecap="round"/>'
        )

    for node in spec.nodes:
        if node.node_type == "sector":
            parts.append(
                f'<rect x="{node.x - 86:.1f}" y="{node.y - 28:.1f}" '
                'width="172" height="56" rx="8" '
                f'fill="{node.color}" stroke="#263238" stroke-opacity="0.18"/>'
            )
            parts.append(_svg_text(node.label, node.x, node.y + 7, 18, fill="#ffffff", weight=700))
        else:
            parts.append(
                f'<circle cx="{node.x:.1f}" cy="{node.y:.1f}" r="36" '
                f'fill="{STOCK_FILL}" stroke="{node.color}" stroke-width="5"/>'
            )
            parts.append(_svg_text(node.label, node.x, node.y + 7, 17, weight=700))

    stock_count = sum(1 for node in spec.nodes if node.node_type == "stock")
    sector_count = sum(1 for node in spec.nodes if node.node_type == "sector")
    stats = (
        f"{stock_count} stock nodes | {sector_count} sector nodes | "
        f"{len(corr_edges)} stock-correlation edges"
    )
    parts.append(_svg_text(stats, spec.width / 2, spec.height - 92, 18, fill=MUTED_TEXT_COLOR))

    legend_x = 440
    legend_y = spec.height - 46
    parts.append(_svg_text("Legend", legend_x - 110, legend_y + 6, 17, anchor="start", weight=700))
    parts.append(
        f'<line x1="{legend_x}" y1="{legend_y}" x2="{legend_x + 54}" '
        f'y2="{legend_y}" stroke="{MEMBERSHIP_EDGE_COLOR}" stroke-width="3" stroke-opacity="0.55"/>'
    )
    parts.append(_svg_text("stock-sector", legend_x + 64, legend_y + 6, 15, anchor="start", fill=MUTED_TEXT_COLOR))
    parts.append(
        f'<line x1="{legend_x + 210}" y1="{legend_y}" x2="{legend_x + 264}" '
        f'y2="{legend_y}" stroke="{POSITIVE_CORR_COLOR}" stroke-width="5" stroke-linecap="round"/>'
    )
    parts.append(_svg_text("positive corr", legend_x + 274, legend_y + 6, 15, anchor="start", fill=MUTED_TEXT_COLOR))
    parts.append(
        f'<line x1="{legend_x + 430}" y1="{legend_y}" x2="{legend_x + 484}" '
        f'y2="{legend_y}" stroke="{NEGATIVE_CORR_COLOR}" stroke-width="5" stroke-linecap="round"/>'
    )
    parts.append(_svg_text("negative corr", legend_x + 494, legend_y + 6, 15, anchor="start", fill=MUTED_TEXT_COLOR))
    parts.append("</svg>")
    return "\n".join(parts)


def write_svg(spec: GraphVisualSpec, output_path: str | Path) -> Path:
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(render_svg(spec), encoding="utf-8")
    return output_file


def _hex_to_rgb(color: str) -> tuple[int, int, int]:
    color = color.lstrip("#")
    return tuple(int(color[index : index + 2], 16) for index in (0, 2, 4))


def _load_font(size: int, bold: bool = False):
    from PIL import ImageFont

    candidates = [
        Path("C:/Windows/Fonts/segoeuib.ttf" if bold else "C:/Windows/Fonts/segoeui.ttf"),
        Path("C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return ImageFont.truetype(str(candidate), size=size)
    return ImageFont.load_default()


def _draw_centered_text(draw: Any, text: str, x: float, y: float, font: Any, fill: str) -> None:
    bbox = draw.textbbox((0, 0), text, font=font)
    width = bbox[2] - bbox[0]
    height = bbox[3] - bbox[1]
    draw.text((x - width / 2, y - height / 2), text, font=font, fill=_hex_to_rgb(fill))


def write_png(spec: GraphVisualSpec, output_path: str | Path) -> Path:
    """Render a graph spec to a PNG using Pillow."""
    from PIL import Image, ImageDraw

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGB", (spec.width, spec.height), _hex_to_rgb(BACKGROUND_COLOR))
    draw = ImageDraw.Draw(image)
    nodes = _node_lookup(spec)
    title_font = _load_font(34, bold=True)
    subtitle_font = _load_font(18)
    label_font = _load_font(17, bold=True)
    sector_font = _load_font(18, bold=True)
    small_font = _load_font(15)

    _draw_centered_text(draw, spec.title, spec.width / 2, 48, title_font, TEXT_COLOR)
    _draw_centered_text(draw, f"Graph date: {spec.date}", spec.width / 2, 86, subtitle_font, MUTED_TEXT_COLOR)

    for edge in spec.edges:
        if edge.edge_type != "membership":
            continue
        source = nodes[edge.source]
        target = nodes[edge.target]
        draw.line(
            [(source.x, source.y), (target.x, target.y)],
            fill=_hex_to_rgb(edge.color),
            width=int(edge.width),
        )

    for edge in spec.edges:
        if edge.edge_type != "correlation":
            continue
        source = nodes[edge.source]
        target = nodes[edge.target]
        points = _quadratic_points(*_curved_path(source, target))
        draw.line(points, fill=_hex_to_rgb(edge.color), width=max(1, int(edge.width)))

    for node in spec.nodes:
        if node.node_type == "sector":
            box = [node.x - 86, node.y - 28, node.x + 86, node.y + 28]
            draw.rounded_rectangle(box, radius=8, fill=_hex_to_rgb(node.color))
            _draw_centered_text(draw, node.label, node.x, node.y, sector_font, "#ffffff")
        else:
            box = [node.x - 36, node.y - 36, node.x + 36, node.y + 36]
            draw.ellipse(box, fill=_hex_to_rgb(STOCK_FILL), outline=_hex_to_rgb(node.color), width=5)
            _draw_centered_text(draw, node.label, node.x, node.y, label_font, TEXT_COLOR)

    corr_edges = [edge for edge in spec.edges if edge.edge_type == "correlation"]
    stock_count = sum(1 for node in spec.nodes if node.node_type == "stock")
    sector_count = sum(1 for node in spec.nodes if node.node_type == "sector")
    stats = (
        f"{stock_count} stock nodes | {sector_count} sector nodes | "
        f"{len(corr_edges)} stock-correlation edges"
    )
    _draw_centered_text(draw, stats, spec.width / 2, spec.height - 98, subtitle_font, MUTED_TEXT_COLOR)

    legend_x = 330
    legend_y = spec.height - 50
    draw.text((legend_x, legend_y - 12), "Legend", font=label_font, fill=_hex_to_rgb(TEXT_COLOR))
    draw.line([(legend_x + 110, legend_y), (legend_x + 164, legend_y)], fill=_hex_to_rgb(MEMBERSHIP_EDGE_COLOR), width=3)
    draw.text((legend_x + 174, legend_y - 11), "stock-sector", font=small_font, fill=_hex_to_rgb(MUTED_TEXT_COLOR))
    draw.line([(legend_x + 320, legend_y), (legend_x + 374, legend_y)], fill=_hex_to_rgb(POSITIVE_CORR_COLOR), width=5)
    draw.text((legend_x + 384, legend_y - 11), "positive corr", font=small_font, fill=_hex_to_rgb(MUTED_TEXT_COLOR))
    draw.line([(legend_x + 535, legend_y), (legend_x + 589, legend_y)], fill=_hex_to_rgb(NEGATIVE_CORR_COLOR), width=5)
    draw.text((legend_x + 599, legend_y - 11), "negative corr", font=small_font, fill=_hex_to_rgb(MUTED_TEXT_COLOR))

    image.save(output_file)
    return output_file


def build_graph_for_visualization(
    stock_features_path: str | Path = DEFAULT_STOCK_INPUT_PATH,
    sector_features_path: str | Path = DEFAULT_SECTOR_INPUT_PATH,
    date: str | None = None,
    corr_window: int = 20,
    top_k: int = 3,
    min_periods: int = 10,
) -> Any:
    stock_df = pd.read_csv(stock_features_path)
    sector_df = pd.read_csv(sector_features_path)
    graphs, _ = build_daily_graphs(
        stock_df=stock_df,
        industry_df=sector_df,
        max_days=None,
        corr_window=corr_window,
        top_k=top_k,
        min_corr_abs=None,
        min_periods=min_periods,
        fill_missing_features="zero",
    )
    if not graphs:
        raise ValueError("No daily graphs were created.")

    if date is None:
        return graphs[-1]

    requested_date = pd.Timestamp(date).date().isoformat()
    for graph in graphs:
        if _graph_date(graph) == requested_date:
            return graph
    available_start = _graph_date(graphs[0])
    available_end = _graph_date(graphs[-1])
    raise ValueError(
        f"No graph exists for {requested_date}. "
        f"Available graph date range: {available_start} to {available_end}."
    )


def export_graph_visualization(
    graph: Any,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    suffix: str = "latest",
    width: int = DEFAULT_WIDTH,
    height: int = DEFAULT_HEIGHT,
) -> tuple[Path, Path, GraphVisualSpec]:
    spec = extract_graph_visual_spec(graph, width=width, height=height)
    output_path = Path(output_dir)
    safe_suffix = suffix.replace("/", "-").replace("\\", "-")
    svg_path = output_path / f"daily_graph_{safe_suffix}.svg"
    png_path = output_path / f"daily_graph_{safe_suffix}.png"
    write_svg(spec, svg_path)
    write_png(spec, png_path)
    return svg_path, png_path, spec


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export SVG/PNG visualizations for a daily stock-sector graph.",
    )
    parser.add_argument("--stock-features", type=Path, default=DEFAULT_STOCK_INPUT_PATH)
    parser.add_argument("--sector-features", type=Path, default=DEFAULT_SECTOR_INPUT_PATH)
    parser.add_argument("--date", default=None, help="Graph date in YYYY-MM-DD format.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--corr-window", type=int, default=20)
    parser.add_argument("--min-periods", type=int, default=10)
    parser.add_argument("--width", type=int, default=DEFAULT_WIDTH)
    parser.add_argument("--height", type=int, default=DEFAULT_HEIGHT)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    graph = build_graph_for_visualization(
        stock_features_path=args.stock_features,
        sector_features_path=args.sector_features,
        date=args.date,
        corr_window=args.corr_window,
        top_k=args.top_k,
        min_periods=args.min_periods,
    )
    suffix = "latest" if args.date is None else pd.Timestamp(args.date).date().isoformat()
    svg_path, png_path, spec = export_graph_visualization(
        graph=graph,
        output_dir=args.output_dir,
        suffix=suffix,
        width=args.width,
        height=args.height,
    )
    corr_edges = [edge for edge in spec.edges if edge.edge_type == "correlation"]
    stock_nodes = [node for node in spec.nodes if node.node_type == "stock"]
    sector_nodes = [node for node in spec.nodes if node.node_type == "sector"]
    print(f"Graph date: {spec.date}")
    print(f"Stock nodes: {len(stock_nodes)}")
    print(f"Sector nodes: {len(sector_nodes)}")
    print(f"Correlation edges: {len(corr_edges)}")
    print(f"SVG written: {svg_path}")
    print(f"PNG written: {png_path}")


if __name__ == "__main__":
    main()
