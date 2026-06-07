"""
Transform raw source DataFrames into JSIC (19 大分類) × size (2 cols) matrices.

GDP distribution note:
  国民経済計算のGDPは業種単位のみ。企業規模別は経済センサスの付加価値比率で按分。
  gdp_by_size = gdp_sector * (census_value_added_by_size / census_value_added_total)
"""
from __future__ import annotations

import pandas as pd
import numpy as np

SIZES = ["大企業", "中小企業"]
SECTOR_CODES = list("ABCDEFGHIJKLMNOPQRS")


def _pivot(df: pd.DataFrame, value_col: str, fill: float = 0.0) -> pd.DataFrame:
    """Pivot to (jsic_code x company_size_category) matrix."""
    pt = df.pivot_table(
        index="jsic_code",
        columns="company_size_category",
        values=value_col,
        aggfunc="sum",
    )
    pt = pt.reindex(index=SECTOR_CODES)
    for s in SIZES:
        if s not in pt.columns:
            pt[s] = fill
    pt = pt[SIZES].fillna(fill)
    return pt


def build_gdp_matrix(df_census: pd.DataFrame, df_gdp: pd.DataFrame) -> pd.DataFrame:
    """
    Distribute sector-level GDP to size categories using census value-added ratios.
    Returns matrix indexed by jsic_code, columns = SIZES, unit = billion JPY.
    """
    gdp_by_sector = df_gdp.set_index("jsic_code")["gdp_contribution_billion_jpy"]

    va_pivot = _pivot(df_census, "value_added_billion_jpy")
    va_total = va_pivot.sum(axis=1).replace(0, np.nan)

    gdp_matrix = va_pivot.div(va_total, axis=0).mul(gdp_by_sector, axis=0)
    gdp_matrix = gdp_matrix.fillna(0)
    gdp_matrix.index.name = "jsic_code"
    return gdp_matrix


def build_profit_matrix(df_corp: pd.DataFrame) -> pd.DataFrame:
    pt = _pivot(df_corp, "operating_profit_billion_jpy")
    pt.index.name = "jsic_code"
    return pt


def build_company_count_matrix(df_census: pd.DataFrame) -> pd.DataFrame:
    pt = _pivot(df_census, "company_count")
    pt.index.name = "jsic_code"
    return pt


def build_employee_count_matrix(df_census: pd.DataFrame) -> pd.DataFrame:
    """従業者数分布マトリクス。単位: 人。"""
    pt = _pivot(df_census, "employee_count")
    pt.index.name = "jsic_code"
    return pt


def build_per_company_matrix(df_census: pd.DataFrame) -> pd.DataFrame:
    """付加価値 / 企業数 → 百万円/社 (billion JPY / 社 × 1000)."""
    va   = _pivot(df_census, "value_added_billion_jpy")
    cnt  = _pivot(df_census, "company_count").replace(0, np.nan)
    per_co = (va / cnt * 1000).fillna(0)
    per_co.index.name = "jsic_code"
    return per_co


def build_per_person_matrix(df_census: pd.DataFrame) -> pd.DataFrame:
    """付加価値 / 従業者数 → 万円/人 (billion JPY / 人 × 1e5)."""
    va   = _pivot(df_census, "value_added_billion_jpy")
    emp  = _pivot(df_census, "employee_count").replace(0, np.nan)
    per_pp = (va / emp * 1e5).fillna(0)
    per_pp.index.name = "jsic_code"
    return per_pp


# ── 法人企業統計ベース（資本金階級）──────────────────
# 大企業 = 資本金10億円以上、中小 = 1千万〜10億円未満。
# 母集団/従業員数/付加価値/営業利益すべて法人企業統計内で完結する。

def build_corp_company_count_matrix(df_corp: pd.DataFrame) -> pd.DataFrame:
    pt = _pivot(df_corp, "corp_company_count")
    pt.index.name = "jsic_code"
    return pt


def build_corp_employee_count_matrix(df_corp: pd.DataFrame) -> pd.DataFrame:
    pt = _pivot(df_corp, "corp_employee_count")
    pt.index.name = "jsic_code"
    return pt


def build_corp_value_added_matrix(df_corp: pd.DataFrame) -> pd.DataFrame:
    pt = _pivot(df_corp, "value_added_corp_billion_jpy")
    pt.index.name = "jsic_code"
    return pt


def build_corp_per_company_va_matrix(df_corp: pd.DataFrame) -> pd.DataFrame:
    """法人企業統計の付加価値 / 母集団 → 百万円/社（規模定義一致）。"""
    va  = _pivot(df_corp, "value_added_corp_billion_jpy")
    cnt = _pivot(df_corp, "corp_company_count").replace(0, np.nan)
    out = (va / cnt * 1000).fillna(0)
    out.index.name = "jsic_code"
    return out


def build_corp_per_person_va_matrix(df_corp: pd.DataFrame) -> pd.DataFrame:
    """法人企業統計の付加価値 / 期中平均従業員数 → 万円/人（規模定義一致）。"""
    va  = _pivot(df_corp, "value_added_corp_billion_jpy")
    emp = _pivot(df_corp, "corp_employee_count").replace(0, np.nan)
    out = (va / emp * 1e5).fillna(0)
    out.index.name = "jsic_code"
    return out


def build_corp_per_company_profit_matrix(df_corp: pd.DataFrame) -> pd.DataFrame:
    """営業利益 / 母集団 → 百万円/社（規模定義一致）。"""
    profit = _pivot(df_corp, "operating_profit_billion_jpy")
    cnt    = _pivot(df_corp, "corp_company_count").replace(0, np.nan)
    out = (profit / cnt * 1000).fillna(0)
    out.index.name = "jsic_code"
    return out


def build_corp_per_person_profit_matrix(df_corp: pd.DataFrame) -> pd.DataFrame:
    """営業利益 / 期中平均従業員数 → 万円/人（規模定義一致）。"""
    profit = _pivot(df_corp, "operating_profit_billion_jpy")
    emp    = _pivot(df_corp, "corp_employee_count").replace(0, np.nan)
    out = (profit / emp * 1e5).fillna(0)
    out.index.name = "jsic_code"
    return out


def build_per_company_profit_matrix(df_census: pd.DataFrame, df_corp: pd.DataFrame) -> pd.DataFrame:
    """営業利益 / 企業数 → 百万円/社.

    注意: profit は法人企業統計（資本金階級ベース）、count は経済センサス（従業者規模ベース）
    のため、規模区分の定義は厳密には一致しない（近似値）。
    """
    profit = _pivot(df_corp, "operating_profit_billion_jpy")
    cnt    = _pivot(df_census, "company_count").replace(0, np.nan)
    per_co = (profit / cnt * 1000).fillna(0)
    per_co.index.name = "jsic_code"
    return per_co


def build_per_person_profit_matrix(df_census: pd.DataFrame, df_corp: pd.DataFrame) -> pd.DataFrame:
    """営業利益 / 従業者数 → 万円/人.

    注意: profit は法人企業統計、employee は経済センサスのため規模区分の定義が異なる（近似値）。
    """
    profit = _pivot(df_corp, "operating_profit_billion_jpy")
    emp    = _pivot(df_census, "employee_count").replace(0, np.nan)
    per_pp = (profit / emp * 1e5).fillna(0)
    per_pp.index.name = "jsic_code"
    return per_pp


def add_sector_names(matrix: pd.DataFrame, sectors_df: pd.DataFrame) -> pd.DataFrame:
    """Add Japanese sector name to matrix index."""
    mapping = sectors_df.set_index("code")["name_ja"]
    result = matrix.copy()
    result.index = result.index.map(mapping)
    result.index.name = "業種"
    return result
