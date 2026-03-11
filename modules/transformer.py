"""
Transform raw source DataFrames into JPX33 (33 rows) × size (3 cols) matrices.

GDP distribution note:
  国民経済計算のGDPは業種単位のみ。企業規模別は経済センサスの付加価値比率で按分。
  gdp_by_size = gdp_sector * (census_value_added_by_size / census_value_added_total)
"""
from __future__ import annotations

import pandas as pd
import numpy as np

SIZES = ["大企業", "中小企業", "個人事業主"]
SECTOR_CODES = list(range(1, 34))


def _pivot(df: pd.DataFrame, value_col: str, fill: float = 0.0) -> pd.DataFrame:
    """Pivot to (jpx33_code x company_size_category) matrix."""
    pt = df.pivot_table(
        index="jpx33_code",
        columns="company_size_category",
        values=value_col,
        aggfunc="sum",
    )
    pt = pt.reindex(index=SECTOR_CODES)
    # Ensure all 3 size columns present
    for s in SIZES:
        if s not in pt.columns:
            pt[s] = fill
    pt = pt[SIZES].fillna(fill)
    return pt


def build_gdp_matrix(df_census: pd.DataFrame, df_gdp: pd.DataFrame) -> pd.DataFrame:
    """
    Distribute sector-level GDP to size categories using census value-added ratios.
    Returns matrix indexed by jpx33_code, columns = SIZES, unit = billion JPY.
    """
    # Sector-level GDP (index = jpx33_code)
    gdp_by_sector = df_gdp.set_index("jpx33_code")["gdp_contribution_billion_jpy"]

    # Value-added totals per sector
    va_pivot = _pivot(df_census, "value_added_billion_jpy")
    va_total = va_pivot.sum(axis=1).replace(0, np.nan)

    # Distribute GDP proportionally
    gdp_matrix = va_pivot.div(va_total, axis=0).mul(gdp_by_sector, axis=0)
    gdp_matrix = gdp_matrix.fillna(0)
    gdp_matrix.index.name = "jpx33_code"
    return gdp_matrix


def build_profit_matrix(df_corp: pd.DataFrame) -> pd.DataFrame:
    """
    Operating profit matrix. 個人事業主 column filled with 0 (not in corporate stats).
    """
    pt = _pivot(df_corp, "operating_profit_billion_jpy")
    pt.index.name = "jpx33_code"
    return pt


def build_company_count_matrix(df_census: pd.DataFrame) -> pd.DataFrame:
    pt = _pivot(df_census, "company_count")
    pt.index.name = "jpx33_code"
    return pt


def build_employee_count_matrix(df_census: pd.DataFrame) -> pd.DataFrame:
    """従業者数分布マトリクス。単位: 人。"""
    pt = _pivot(df_census, "employee_count")
    pt.index.name = "jpx33_code"
    return pt


def build_per_company_matrix(df_census: pd.DataFrame) -> pd.DataFrame:
    """付加価値 / 企業数 → 億円/社 (= billion JPY / count → 百万円/社 なので×1000→万円/社)"""
    va   = _pivot(df_census, "value_added_billion_jpy")
    cnt  = _pivot(df_census, "company_count").replace(0, np.nan)
    # Convert to 億円/社 (10億 / 社数) → 百万円/社 (×1000)
    # billion JPY / 社数 * 1000 = million JPY per company
    per_co = (va / cnt * 1000).fillna(0)
    per_co.index.name = "jpx33_code"
    return per_co


def build_per_person_matrix(df_census: pd.DataFrame) -> pd.DataFrame:
    """付加価値 / 従業者数 → 万円/人 (billion JPY / persons * 10^9 / 10^4 = 10^5 万円... → *100000/count)"""
    va   = _pivot(df_census, "value_added_billion_jpy")
    emp  = _pivot(df_census, "employee_count").replace(0, np.nan)
    # billion JPY / persons = 10^9 JPY / persons
    # → 万円/人 = 10^9 / count / 10_000 = 100_000 / count (when va in billion)
    # va [billion JPY] / emp [人] * 1e9 / 1e4 = * 1e5 万円/人
    per_pp = (va / emp * 1e5).fillna(0)
    per_pp.index.name = "jpx33_code"
    return per_pp


def add_sector_names(matrix: pd.DataFrame, sectors_df: pd.DataFrame) -> pd.DataFrame:
    """Add jpx33_name column (Japanese) to matrix and set as index."""
    mapping = sectors_df.set_index("code")["name_ja"]
    result = matrix.copy()
    result.index = result.index.map(mapping)
    result.index.name = "業種"
    return result
