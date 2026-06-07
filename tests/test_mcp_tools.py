"""MCP ツールの純粋ロジック (mcp_server/_tools.py) のユニットテスト.

server.py は FastMCP に依存し pywin32 ロードでブロックされる環境があるため、
ロジックは _tools.py に分離してここでテストする。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "mcp_server"))

from _tools import (  # noqa: E402
    CAPITAL_CLASSES,
    SECTOR_CODES,
    _CORP_METRIC_COLS,
    compare_growth,
    get_caveats,
    get_timeseries,
    list_metadata,
    query_metric,
    sector_breakdown,
)


# ── list_metadata ──

def test_list_metadata_has_sectors_and_capital_and_years():
    out = json.loads(list_metadata())
    assert {"sectors", "capital_classes", "available_years",
            "supported_metrics", "data_sources"} <= out.keys()
    assert len(out["sectors"]) == 19  # JSIC A-S
    assert out["capital_classes"] == list(CAPITAL_CLASSES)
    assert all(2020 <= y <= 2024 for y in out["available_years"])
    assert len(out["available_years"]) >= 5


def test_list_metadata_supported_metrics_complete():
    out = json.loads(list_metadata())
    keys = {m["key"] for m in out["supported_metrics"]}
    assert keys == set(_CORP_METRIC_COLS.keys())


# ── query_metric ──

def test_query_metric_valid_returns_float_value():
    out = json.loads(query_metric("E", "10億+", 2023, "operating_profit"))
    assert "value" in out
    assert isinstance(out["value"], float)
    assert out["jsic_code"] == "E"
    assert out["capital_class"] == "10億+"
    assert out["year"] == 2023
    assert out["unit"] == "billion JPY"


def test_query_metric_unknown_metric_returns_error():
    out = json.loads(query_metric("E", "10億+", 2023, "foo"))
    assert "error" in out
    assert "supported" in out


def test_query_metric_unknown_jsic_returns_error():
    out = json.loads(query_metric("Z", "10億+", 2023, "operating_profit"))
    assert "error" in out


def test_query_metric_unknown_capital_returns_error():
    out = json.loads(query_metric("E", "存在しない区分", 2023, "operating_profit"))
    assert "error" in out


def test_query_metric_unknown_year_returns_error():
    out = json.loads(query_metric("E", "10億+", 1990, "operating_profit"))
    assert "error" in out
    assert "available_years" in out


# ── get_timeseries ──

def test_get_timeseries_returns_5_year_series():
    out = json.loads(get_timeseries("E", "10億+", "operating_profit"))
    assert "series" in out
    assert len(out["series"]) >= 5
    years = [p["year"] for p in out["series"]]
    assert years == sorted(years)  # 年度昇順
    assert all(isinstance(p["value"], float) for p in out["series"])


def test_get_timeseries_growth_pct_computed():
    out = json.loads(get_timeseries("E", "10億+", "operating_profit"))
    assert "growth_pct_first_to_last" in out


def test_get_timeseries_unknown_metric_returns_error():
    out = json.loads(get_timeseries("E", "10億+", "bogus"))
    assert "error" in out


# ── compare_growth ──

def test_compare_growth_default_returns_ranking():
    out = json.loads(compare_growth())
    assert "ranking" in out
    assert isinstance(out["ranking"], list)
    assert len(out["ranking"]) <= 10  # top_n=10 default
    assert "compared_years" in out
    assert len(out["compared_years"]) == 2


def test_compare_growth_ranking_sorted_desc():
    out = json.loads(compare_growth(top_n=20))
    growths = [r["growth_pct"] for r in out["ranking"] if r["growth_pct"] is not None]
    assert growths == sorted(growths, reverse=True)


def test_compare_growth_respects_min_threshold():
    out_hi = json.loads(compare_growth(min_value_billion_jpy=10000.0, top_n=50))
    out_lo = json.loads(compare_growth(min_value_billion_jpy=1.0, top_n=50))
    assert len(out_hi["ranking"]) <= len(out_lo["ranking"])


def test_compare_growth_unknown_metric_returns_error():
    out = json.loads(compare_growth(metric="bogus"))
    assert "error" in out


# ── sector_breakdown ──

def test_sector_breakdown_returns_19_sectors():
    out = json.loads(sector_breakdown(2023, "operating_profit"))
    assert len(out["breakdown"]) == 19
    codes = [b["jsic_code"] for b in out["breakdown"]]
    assert codes == list(SECTOR_CODES)


def test_sector_breakdown_has_all_capital_classes():
    out = json.loads(sector_breakdown(2023, "operating_profit"))
    for b in out["breakdown"]:
        assert set(b["by_capital"].keys()) == set(CAPITAL_CLASSES)


def test_sector_breakdown_sector_total_matches_sum():
    out = json.loads(sector_breakdown(2023, "operating_profit"))
    for b in out["breakdown"]:
        expected = sum(b["by_capital"].values())
        assert b["sector_total"] == pytest.approx(expected, abs=0.5)


def test_sector_breakdown_unknown_year_returns_error():
    out = json.loads(sector_breakdown(1990))
    assert "error" in out


# ── get_caveats ──

def test_get_caveats_has_expected_sections():
    out = json.loads(get_caveats())
    assert "data_sources" in out
    assert "known_zeros" in out
    assert "differences_with_sna_gdp" in out
    assert "size_definition_mismatch" in out


def test_get_caveats_known_zero_sectors_documented():
    out = json.loads(get_caveats())
    keys = " ".join(out["known_zeros"].keys())
    assert "金融" in keys
    assert "公務" in keys
