"""
CSV loading and normalization for GDP_distribution.
All reads use utf-8-sig to handle BOM from Excel / government data exports.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml


def load_jsic_sectors(config_path: Path) -> pd.DataFrame:
    """Returns DataFrame with columns: code, name_ja, name_en (JSIC 大分類 A〜S)."""
    with open(config_path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return pd.DataFrame(data["sectors"])


def load_census(data_dir: Path) -> pd.DataFrame:
    """Load economic census CSV (JSIC 大分類 × 規模)."""
    path = data_dir / "census.csv"
    df = pd.read_csv(path, encoding="utf-8-sig")
    _require_columns(df, [
        "jsic_code", "jsic_name", "company_size_category",
        "company_count", "employee_count",
        "sales_billion_jpy", "value_added_billion_jpy", "survey_year",
    ], path)
    df["company_size_category"] = df["company_size_category"].astype(str)
    return df


def load_corporate_stats(data_dir: Path) -> pd.DataFrame:
    """Load corporate enterprise statistics CSV (JSIC 大分類 × 資本金階級)."""
    path = data_dir / "corporate_stats.csv"
    df = pd.read_csv(path, encoding="utf-8-sig")
    _require_columns(df, [
        "jsic_code", "jsic_name", "company_size_category",
        "corp_company_count", "corp_employee_count",
        "value_added_corp_billion_jpy",
        "operating_profit_billion_jpy", "net_profit_billion_jpy",
        "total_assets_billion_jpy", "equity_billion_jpy", "survey_year",
    ], path)
    return df


def load_gdp(data_dir: Path) -> pd.DataFrame:
    """Load GDP by industry CSV (JSIC 大分類)."""
    path = data_dir / "gdp_by_industry.csv"
    df = pd.read_csv(path, encoding="utf-8-sig")
    _require_columns(df, [
        "jsic_code", "jsic_name",
        "gdp_contribution_billion_jpy", "gdp_share_pct", "survey_year",
    ], path)
    return df


def _require_columns(df: pd.DataFrame, cols: list[str], path: Path) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"{path.name}: missing columns {missing}")
