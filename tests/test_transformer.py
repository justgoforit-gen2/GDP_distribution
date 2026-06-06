"""Test transformer matrix builders against synthetic minimal input."""
import pandas as pd
import pytest

from modules.transformer import (
    SECTOR_CODES,
    SIZES,
    build_company_count_matrix,
    build_employee_count_matrix,
    build_gdp_matrix,
    build_per_company_matrix,
    build_profit_matrix,
)


def _mini_census() -> pd.DataFrame:
    """Two sectors (A,E), both sizes, minimal but realistic."""
    return pd.DataFrame([
        {"jsic_code": "A", "company_size_category": "大企業",   "company_count": 100, "employee_count": 10_000, "value_added_billion_jpy": 50,  "sales_billion_jpy": 200, "survey_year": 2021},
        {"jsic_code": "A", "company_size_category": "中小企業", "company_count": 1_000, "employee_count": 50_000, "value_added_billion_jpy": 100, "sales_billion_jpy": 300, "survey_year": 2021},
        {"jsic_code": "E", "company_size_category": "大企業",   "company_count": 500, "employee_count": 200_000, "value_added_billion_jpy": 5_000, "sales_billion_jpy": 12_000, "survey_year": 2021},
        {"jsic_code": "E", "company_size_category": "中小企業", "company_count": 5_000, "employee_count": 300_000, "value_added_billion_jpy": 3_000, "sales_billion_jpy": 8_000, "survey_year": 2021},
    ])


def _mini_corp() -> pd.DataFrame:
    return pd.DataFrame([
        {"jsic_code": "A", "company_size_category": "大企業",   "operating_profit_billion_jpy": 5,  "net_profit_billion_jpy": 3, "total_assets_billion_jpy": 200, "equity_billion_jpy": 80, "survey_year": 2023},
        {"jsic_code": "E", "company_size_category": "大企業",   "operating_profit_billion_jpy": 1_000, "net_profit_billion_jpy": 700, "total_assets_billion_jpy": 30_000, "equity_billion_jpy": 12_000, "survey_year": 2023},
        {"jsic_code": "E", "company_size_category": "中小企業", "operating_profit_billion_jpy": 400,   "net_profit_billion_jpy": 250, "total_assets_billion_jpy": 15_000, "equity_billion_jpy": 5_000, "survey_year": 2023},
    ])


def _mini_gdp() -> pd.DataFrame:
    return pd.DataFrame([
        {"jsic_code": "A", "gdp_contribution_billion_jpy": 200, "gdp_share_pct": 5.0, "survey_year": 2023},
        {"jsic_code": "E", "gdp_contribution_billion_jpy": 10_000, "gdp_share_pct": 25.0, "survey_year": 2023},
    ])


def test_sizes_dropped_solo_proprietor():
    assert SIZES == ["大企業", "中小企業"]


def test_sector_codes_19():
    assert len(SECTOR_CODES) == 19
    assert SECTOR_CODES[0] == "A" and SECTOR_CODES[-1] == "S"


def test_company_count_pivot():
    m = build_company_count_matrix(_mini_census())
    assert m.loc["A", "大企業"] == 100
    assert m.loc["E", "中小企業"] == 5_000
    # JSIC sectors with no input become 0 after reindex
    assert m.loc["B", "大企業"] == 0
    assert list(m.columns) == ["大企業", "中小企業"]


def test_gdp_matrix_distributes_by_va_ratio():
    m = build_gdp_matrix(_mini_census(), _mini_gdp())
    # A sector va ratio: 50/(50+100) = 1/3, total GDP A = 200 → 大企業≈66.7
    assert m.loc["A", "大企業"] == pytest.approx(200 * 50 / 150, rel=1e-6)
    assert m.loc["A", "中小企業"] == pytest.approx(200 * 100 / 150, rel=1e-6)
    # E sector: GDP 10,000 → 大企業:5000/8000, 中小:3000/8000
    assert m.loc["E", "大企業"] == pytest.approx(10_000 * 5_000 / 8_000)


def test_profit_matrix_individual_proprietor_absent():
    m = build_profit_matrix(_mini_corp())
    assert list(m.columns) == ["大企業", "中小企業"]
    # A 中小企業 has no row → 0
    assert m.loc["A", "中小企業"] == 0
    assert m.loc["E", "大企業"] == 1_000


def test_per_company_unit_conversion():
    m = build_per_company_matrix(_mini_census())
    # A 大企業: 50 billion / 100 社 * 1000 = 500 million / company
    assert m.loc["A", "大企業"] == pytest.approx(500.0)


def test_employee_pivot():
    m = build_employee_count_matrix(_mini_census())
    assert m.loc["E", "中小企業"] == 300_000
    assert m.loc["A", "大企業"] == 10_000
