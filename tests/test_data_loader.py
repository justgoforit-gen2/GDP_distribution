"""Test that data_loader correctly loads JSIC-based CSVs in data/processed."""
from pathlib import Path

import pandas as pd
import pytest

from modules.data_loader import (
    load_census,
    load_corporate_stats,
    load_gdp,
    load_jsic_sectors,
)

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "processed"
CONFIG_DIR = ROOT / "config"


@pytest.fixture(scope="module")
def sectors() -> pd.DataFrame:
    return load_jsic_sectors(CONFIG_DIR / "jsic_sectors.yaml")


def test_jsic_sectors_19(sectors):
    assert len(sectors) == 19
    assert set(sectors.columns) >= {"code", "name_ja", "name_en"}
    assert list(sectors["code"]) == list("ABCDEFGHIJKLMNOPQRS")


def test_census_schema():
    if not (DATA_DIR / "census.csv").exists():
        pytest.skip("census.csv not built")
    df = load_census(DATA_DIR)
    assert {"jsic_code", "jsic_name", "company_size_category",
            "company_count", "employee_count",
            "sales_billion_jpy", "value_added_billion_jpy", "survey_year"} <= set(df.columns)
    assert set(df["company_size_category"].unique()) <= {"大企業", "中小企業"}
    assert df["company_count"].sum() > 1_000_000
    assert df["employee_count"].sum() > 10_000_000


def test_corporate_stats_schema():
    if not (DATA_DIR / "corporate_stats.csv").exists():
        pytest.skip("corporate_stats.csv not built")
    df = load_corporate_stats(DATA_DIR)
    assert {"jsic_code", "jsic_name", "company_size_category",
            "operating_profit_billion_jpy", "net_profit_billion_jpy",
            "total_assets_billion_jpy", "equity_billion_jpy", "survey_year"} <= set(df.columns)
    assert set(df["company_size_category"].unique()) == {"大企業", "中小企業"}


def test_gdp_schema():
    if not (DATA_DIR / "gdp_by_industry.csv").exists():
        pytest.skip("gdp_by_industry.csv not built")
    df = load_gdp(DATA_DIR)
    assert {"jsic_code", "jsic_name",
            "gdp_contribution_billion_jpy", "gdp_share_pct", "survey_year"} <= set(df.columns)
    # Real economy: total GDP should be roughly in the 500-650 trillion JPY range (= 500,000-650,000 billion)
    total = df["gdp_contribution_billion_jpy"].sum()
    assert 400_000 <= total <= 700_000, f"GDP total {total} out of plausible range"
