"""
Plotly heatmap factory functions.
All heatmaps share the same axis structure: Y = JPX33 sector names (33 rows),
X = company size categories (3 columns), with row/column subtotals appended.
"""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go


def _add_subtotals(matrix: pd.DataFrame, agg: str = "sum") -> pd.DataFrame:
    """Append a '合計' column (right) and '合計' row (bottom)."""
    m = matrix.copy()
    if agg == "sum":
        m["合計"] = m.sum(axis=1)
        row_total = m.sum(axis=0)
    else:
        m["合計"] = m.mean(axis=1)
        row_total = m.mean(axis=0)
    row_total.name = "合計"
    return pd.concat([m, pd.DataFrame([row_total], index=["合計"])])


def _base_heatmap(
    matrix: pd.DataFrame,
    title: str,
    colorscale: str,
    unit: str,
    fmt: str = ".1f",
    zmid: float | None = None,
    agg: str = "sum",
) -> go.Figure:
    m = _add_subtotals(matrix, agg=agg)

    z = m.values
    x = list(m.columns)
    y = list(m.index)
    n_cols = len(x)
    n_rows = len(y)

    text  = [[f"{v:{fmt}}" for v in row] for row in z]
    hover = [[f"{v:{fmt}} {unit}" for v in row] for row in z]

    heatmap_kwargs: dict = dict(
        z=z,
        x=x,
        y=y,
        colorscale=colorscale,
        text=text,
        customdata=hover,
        texttemplate="%{text}",
        textfont={"size": 13},
        hovertemplate="<b>%{y}</b><br>%{x}<br>%{customdata}<extra></extra>",
        showscale=True,
    )
    if zmid is not None:
        heatmap_kwargs["zmid"] = zmid

    fig = go.Figure(go.Heatmap(**heatmap_kwargs))

    # Separator lines between data and subtotal row/column
    shapes = [
        dict(
            type="line",
            xref="x", yref="y",
            x0=n_cols - 1.5, x1=n_cols - 1.5,
            y0=-0.5, y1=n_rows - 0.5,
            line=dict(color="white", width=3),
        ),
        dict(
            type="line",
            xref="x", yref="y",
            x0=-0.5, x1=n_cols - 0.5,
            y0=n_rows - 1.5, y1=n_rows - 1.5,
            line=dict(color="white", width=3),
        ),
    ]

    fig.update_layout(
        title=dict(text=title, font=dict(size=13)),
        margin=dict(l=120, r=20, t=50, b=60),
        height=700,
        xaxis=dict(side="top"),
        yaxis=dict(autorange="reversed"),
        font=dict(family="Noto Sans JP, sans-serif"),
        shapes=shapes,
    )
    return fig


def build_gdp_heatmap(matrix: pd.DataFrame, title: str = "GDP寄与額（付加価値）") -> go.Figure:
    return _base_heatmap(matrix, title, colorscale="Blues", unit="十億円", fmt=",.0f")


def build_profit_heatmap(matrix: pd.DataFrame, title: str = "利益（営業利益）") -> go.Figure:
    return _base_heatmap(matrix, title, colorscale="Greens", unit="十億円", fmt=",.0f")


def build_company_count_heatmap(matrix: pd.DataFrame, title: str = "企業数分布") -> go.Figure:
    return _base_heatmap(matrix, title, colorscale="Oranges", unit="社", fmt=",.0f")


def build_employee_count_heatmap(matrix: pd.DataFrame, title: str = "従業者数分布") -> go.Figure:
    return _base_heatmap(matrix, title, colorscale="Purples", unit="人", fmt=",.0f")


def build_profit_rate_heatmap(
    matrix: pd.DataFrame,
    title: str = "利益率（営業利益/GDP比）",
) -> go.Figure:
    flat = matrix.values.flatten()
    median = float(pd.Series(flat[flat > 0]).median()) if (flat > 0).any() else 0.0
    return _base_heatmap(
        matrix, title, colorscale="RdYlGn", unit="%", fmt=",.1f", zmid=median, agg="mean"
    )


def build_per_unit_heatmap(
    matrix: pd.DataFrame,
    title: str,
    unit: str = "百万円/社",
) -> go.Figure:
    flat = matrix.values.flatten()
    median = float(pd.Series(flat[flat > 0]).median()) if (flat > 0).any() else 0.0
    return _base_heatmap(
        matrix, title, colorscale="RdYlGn", unit=unit, fmt=",.1f", zmid=median, agg="mean"
    )
