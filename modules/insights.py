"""
Cross-view analytical findings: detect structural mismatches in the Japanese economy.
"""
from __future__ import annotations

import pandas as pd
import numpy as np

SIZES = ["大企業", "中小企業"]


def compute_sector_mismatches(
    gdp_matrix: pd.DataFrame,
    profit_matrix: pd.DataFrame,
    count_matrix: pd.DataFrame,
    per_company_matrix: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    """
    Returns dict of named findings. Each value is a DataFrame of flagged sectors.
    All input matrices are indexed by sector name (Japanese), columns = SIZES.
    """
    findings: dict[str, pd.DataFrame] = {}

    gdp_total    = gdp_matrix.sum(axis=1)
    profit_total = profit_matrix[["大企業", "中小企業"]].sum(axis=1)

    # GDP share and profit share of each sector (vs universe)
    gdp_share    = gdp_total / gdp_total.sum()
    profit_share = profit_total / profit_total.replace(0, np.nan).sum()

    # 1. GDP大・利益薄: GDP占有率が高いのに利益率が低い
    margin = profit_total / gdp_total.replace(0, np.nan)
    threshold_gdp = gdp_share.quantile(0.60)
    threshold_margin = margin.quantile(0.35)
    f1 = pd.DataFrame({
        "GDP占有率(%)": (gdp_share * 100).round(2),
        "営業利益率(GDP比,%)": (margin * 100).round(2),
    })
    f1 = f1[(gdp_share >= threshold_gdp) & (margin <= threshold_margin)].sort_values("GDP占有率(%)", ascending=False)
    findings["GDP大・利益薄セクター"] = f1

    # 2. 利益大・GDP寄与小: 利益占有率がGDP占有率を大幅に上回る
    ratio = (profit_share / gdp_share.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)
    threshold_ratio = ratio.quantile(0.75)
    f2 = pd.DataFrame({
        "利益占有率(%)": (profit_share * 100).round(2),
        "GDP占有率(%)": (gdp_share * 100).round(2),
        "利益/GDP比率": ratio.round(2),
    })
    f2 = f2[ratio >= threshold_ratio].sort_values("利益/GDP比率", ascending=False)
    findings["利益大・GDP寄与小セクター"] = f2

    # 3-4: 「最大規模カラム」と「それ以外の合計」で大企業集中度を測る
    # 経済センサスベース → 大企業/中小企業, 法人企業統計ベース → 10億+/その他
    cols = list(profit_matrix.columns)
    top_col = cols[0] if cols else None

    if top_col is not None and top_col in gdp_matrix.columns:
        gdp_total_sec = gdp_matrix.sum(axis=1).replace(0, np.nan)
        prof_total_sec = profit_matrix.sum(axis=1).replace(0, np.nan)

        large_profit_share = profit_matrix[top_col] / prof_total_sec
        large_va_share = gdp_matrix[top_col] / gdp_total_sec
        gap = large_profit_share - large_va_share
        threshold_gap = gap.dropna().quantile(0.70) if gap.notna().any() else 0
        f3 = pd.DataFrame({
            f"{top_col} 利益シェア(%)": (large_profit_share * 100).round(1),
            f"{top_col} 付加価値シェア(%)": (large_va_share * 100).round(1),
            "シェア差分(pp)": (gap * 100).round(1),
        })
        f3 = f3[gap.fillna(-np.inf) >= threshold_gap].sort_values("シェア差分(pp)", ascending=False)
        findings[f"{top_col}利益集中セクター"] = f3

        # 残り（中小・小規模）側の付加価値支援度
        rest_cols = cols[1:]
        if rest_cols:
            sme_va_share = gdp_matrix[rest_cols].sum(axis=1) / gdp_total_sec
            sme_prof_share = profit_matrix[rest_cols].sum(axis=1) / prof_total_sec
            gap2 = sme_va_share - sme_prof_share
            threshold_sme = gap2.dropna().quantile(0.65) if gap2.notna().any() else 0
            f4 = pd.DataFrame({
                "下位規模 GDP占有率(%)": (sme_va_share * 100).round(1),
                "下位規模 利益シェア(%)": (sme_prof_share * 100).round(1),
                "格差(pp)": (gap2 * 100).round(1),
            })
            f4 = f4[gap2.fillna(-np.inf) >= threshold_sme].sort_values("格差(pp)", ascending=False)
            findings["下位規模GDP支援セクター"] = f4

    # 5. 企業数多・生産性低: 企業数シェアが高いが1社あたり付加価値が低い
    count_total_per_sector = count_matrix.sum(axis=1)
    count_share = count_total_per_sector / count_total_per_sector.sum()
    per_co_avg = per_company_matrix.mean(axis=1)  # average across sizes
    threshold_count = count_share.quantile(0.65)
    threshold_per_co = per_co_avg.quantile(0.40)
    f5 = pd.DataFrame({
        "企業数占有率(%)": (count_share * 100).round(2),
        "1社あたり付加価値(百万円)": per_co_avg.round(1),
    })
    f5 = f5[(count_share >= threshold_count) & (per_co_avg <= threshold_per_co)].sort_values("企業数占有率(%)", ascending=False)
    findings["企業数多・生産性低セクター"] = f5

    return findings
