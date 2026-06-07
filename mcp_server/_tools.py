"""MCP ツールの純粋ロジック (MCP 非依存).

server.py からは @mcp.tool() で薄くラップするだけ。
このモジュールはユニットテストから直接呼べる (FastMCP 不要)。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# GDP_distribution の modules/ を import 可能にする
_GDP_ROOT = Path(__file__).resolve().parents[1]
if str(_GDP_ROOT) not in sys.path:
    sys.path.insert(0, str(_GDP_ROOT))

from modules.data_loader import (  # noqa: E402
    load_corporate_stats,
    load_census,
    load_gdp,
    load_jsic_sectors,
)
from modules.transformer import (  # noqa: E402
    CAPITAL_CLASSES,
    SECTOR_CODES,
)

DATA_DIR = _GDP_ROOT / "data" / "processed"
CONFIG_DIR = _GDP_ROOT / "config"


# ── Cached data accessors ──
_corp_df = None
_census_df = None
_gdp_df = None
_sectors_df = None


def _corp():
    global _corp_df
    if _corp_df is None:
        _corp_df = load_corporate_stats(DATA_DIR)
    return _corp_df


def _census():
    global _census_df
    if _census_df is None:
        _census_df = load_census(DATA_DIR)
    return _census_df


def _gdp():
    global _gdp_df
    if _gdp_df is None:
        _gdp_df = load_gdp(DATA_DIR)
    return _gdp_df


def _sectors():
    global _sectors_df
    if _sectors_df is None:
        _sectors_df = load_jsic_sectors(CONFIG_DIR / "jsic_sectors.yaml")
    return _sectors_df


# ── Metric registry (法人企業統計) ──
_CORP_METRIC_COLS = {
    "operating_profit": "operating_profit_billion_jpy",
    "value_added":      "value_added_corp_billion_jpy",
    "company_count":    "corp_company_count",
    "employee_count":   "corp_employee_count",
    "net_profit":       "net_profit_billion_jpy",
    "total_assets":     "total_assets_billion_jpy",
}

_METRIC_UNITS = {
    "operating_profit": "billion JPY",
    "value_added":      "billion JPY",
    "company_count":    "法人",
    "employee_count":   "人",
    "net_profit":       "billion JPY",
    "total_assets":     "billion JPY",
}


def list_metadata() -> str:
    sectors_df = _sectors()
    corp_df = _corp()
    years = sorted(corp_df["survey_year"].unique().tolist())

    out = {
        "sectors": [
            {"jsic_code": r["code"], "name_ja": r["name_ja"], "name_en": r["name_en"]}
            for _, r in sectors_df.iterrows()
        ],
        "capital_classes": CAPITAL_CLASSES,
        "available_years": years,
        "supported_metrics": [
            {"key": k, "column": v, "unit": _METRIC_UNITS[k]}
            for k, v in _CORP_METRIC_COLS.items()
        ],
        "data_sources": {
            "corporate_stats": "法人企業統計年報 (財務省, statsDataId=0003060791)",
            "census":          "経済センサス活動調査 2021 (経産省・総務省, 0004006327)",
            "gdp":             "国民経済計算 2023 (内閣府, 0004028489)",
        },
    }
    return json.dumps(out, ensure_ascii=False, indent=2)


def query_metric(jsic_code: str, capital_class: str, year: int, metric: str = "operating_profit") -> str:
    if metric not in _CORP_METRIC_COLS:
        return json.dumps({"error": f"unknown metric: {metric}",
                           "supported": list(_CORP_METRIC_COLS.keys())}, ensure_ascii=False)
    if jsic_code not in SECTOR_CODES:
        return json.dumps({"error": f"unknown jsic_code: {jsic_code}",
                           "supported": SECTOR_CODES}, ensure_ascii=False)
    if capital_class not in CAPITAL_CLASSES:
        return json.dumps({"error": f"unknown capital_class: {capital_class}",
                           "supported": CAPITAL_CLASSES}, ensure_ascii=False)

    df = _corp()
    col = _CORP_METRIC_COLS[metric]
    row = df[
        (df["jsic_code"] == jsic_code)
        & (df["capital_class"] == capital_class)
        & (df["survey_year"] == year)
    ]
    if row.empty:
        return json.dumps({"error": f"no data: year={year}",
                           "available_years": sorted(df["survey_year"].unique().tolist())},
                          ensure_ascii=False)

    sectors_df = _sectors()
    name_ja = sectors_df.set_index("code").loc[jsic_code, "name_ja"]

    return json.dumps({
        "jsic_code": jsic_code,
        "jsic_name": name_ja,
        "capital_class": capital_class,
        "year": year,
        "metric": metric,
        "value": float(row.iloc[0][col]),
        "unit": _METRIC_UNITS[metric],
    }, ensure_ascii=False, indent=2)


def get_timeseries(jsic_code: str, capital_class: str, metric: str = "operating_profit") -> str:
    if metric not in _CORP_METRIC_COLS:
        return json.dumps({"error": f"unknown metric: {metric}"}, ensure_ascii=False)

    df = _corp()
    col = _CORP_METRIC_COLS[metric]
    subset = df[
        (df["jsic_code"] == jsic_code) & (df["capital_class"] == capital_class)
    ].sort_values("survey_year")

    if subset.empty:
        return json.dumps({"error": "no data",
                           "jsic_code": jsic_code,
                           "capital_class": capital_class}, ensure_ascii=False)

    sectors_df = _sectors()
    name_ja = sectors_df.set_index("code").loc[jsic_code, "name_ja"]

    series = [
        {"year": int(r["survey_year"]), "value": float(r[col])}
        for _, r in subset.iterrows()
    ]

    growth_pct = None
    if len(series) >= 2 and series[0]["value"]:
        growth_pct = round((series[-1]["value"] / series[0]["value"] - 1) * 100, 1)

    return json.dumps({
        "jsic_code": jsic_code,
        "jsic_name": name_ja,
        "capital_class": capital_class,
        "metric": metric,
        "unit": _METRIC_UNITS[metric],
        "series": series,
        "growth_pct_first_to_last": growth_pct,
    }, ensure_ascii=False, indent=2)


def compare_growth(metric: str = "operating_profit", top_n: int = 10,
                   min_value_billion_jpy: float = 100.0) -> str:
    if metric not in _CORP_METRIC_COLS:
        return json.dumps({"error": f"unknown metric: {metric}"}, ensure_ascii=False)

    df = _corp()
    col = _CORP_METRIC_COLS[metric]
    years = sorted(df["survey_year"].unique().tolist())
    if len(years) < 2:
        return json.dumps({"error": "need >= 2 years of data"}, ensure_ascii=False)

    first_year, last_year = years[0], years[-1]
    first = df[df["survey_year"] == first_year].set_index(["jsic_code", "capital_class"])[col]
    last = df[df["survey_year"] == last_year].set_index(["jsic_code", "capital_class"])[col]

    joined = first.to_frame("first").join(last.to_frame("last"), how="inner")
    joined = joined[joined["last"] >= min_value_billion_jpy]
    joined["growth_pct"] = (joined["last"] / joined["first"].replace(0, float("nan")) - 1) * 100
    joined = joined.sort_values("growth_pct", ascending=False).head(top_n)

    sectors_df = _sectors()
    name_map = sectors_df.set_index("code")["name_ja"]

    ranking = []
    for (jsic, cap), r in joined.iterrows():
        ranking.append({
            "jsic_code": jsic,
            "jsic_name": name_map.get(jsic, ""),
            "capital_class": cap,
            f"value_{first_year}": round(float(r["first"]), 2),
            f"value_{last_year}":  round(float(r["last"]), 2),
            "growth_pct": round(float(r["growth_pct"]), 1) if r["growth_pct"] == r["growth_pct"] else None,
        })

    return json.dumps({
        "metric": metric,
        "unit": _METRIC_UNITS[metric],
        "compared_years": [first_year, last_year],
        "min_value_threshold": min_value_billion_jpy,
        "ranking": ranking,
    }, ensure_ascii=False, indent=2)


def sector_breakdown(year: int, metric: str = "operating_profit") -> str:
    if metric not in _CORP_METRIC_COLS:
        return json.dumps({"error": f"unknown metric: {metric}"}, ensure_ascii=False)

    df = _corp()
    if year not in df["survey_year"].unique():
        return json.dumps({"error": f"no data for year {year}",
                           "available_years": sorted(df["survey_year"].unique().tolist())},
                          ensure_ascii=False)

    col = _CORP_METRIC_COLS[metric]
    df_y = df[df["survey_year"] == year]

    sectors_df = _sectors()
    name_map = sectors_df.set_index("code")["name_ja"]

    breakdown = []
    for jsic in SECTOR_CODES:
        sub = df_y[df_y["jsic_code"] == jsic]
        by_capital = {cap: 0.0 for cap in CAPITAL_CLASSES}
        for _, r in sub.iterrows():
            by_capital[r["capital_class"]] = float(r[col])
        breakdown.append({
            "jsic_code": jsic,
            "jsic_name": name_map.get(jsic, ""),
            "by_capital": by_capital,
            "sector_total": round(sum(by_capital.values()), 2),
        })

    return json.dumps({
        "year": year,
        "metric": metric,
        "unit": _METRIC_UNITS[metric],
        "capital_classes": CAPITAL_CLASSES,
        "breakdown": breakdown,
    }, ensure_ascii=False, indent=2)


def get_caveats() -> str:
    return json.dumps({
        "data_sources": {
            "corporate_stats": {
                "name": "法人企業統計年報 (財務省)",
                "statsDataId": "0003060791",
                "years": "2020〜2024年度 (5年)",
                "scope": "営利法人。金融業・保険業を除外。",
                "size_definition": "資本金階級 6区分 (10億+ / 1億-10億 / 5千万-1億 / 2千万-5千万 / 1千万-2千万 / 1千万未満)",
            },
            "census": {
                "name": "経済センサス 活動調査",
                "statsDataId": "0004006327",
                "years": "2021年 (5年周期、次回2026)",
                "scope": "民営事業所。個人経営含む。",
                "size_definition": "従業者規模 (50人以上=大企業の便宜近似)",
            },
            "gdp": {
                "name": "国民経済計算 経済活動別国内総生産 (名目)",
                "statsDataId": "0004028489",
                "years": "2023年 (暦年)",
                "scope": "SNAベース。家計サービス・帰属家賃含む。",
            },
        },
        "known_zeros": {
            "J 金融業，保険業":             "法人企業統計の別表のため本データでは 0。財務省 0003061947 参照",
            "Q 複合サービス事業":            "法人企業統計の対象外 (郵便局・農協)",
            "S 公務（他に分類されるものを除く）": "営利法人ではないため法人企業統計対象外",
            "R サービス業（他に分類されないもの）": "サービス業集約コードが他業種と重複するため除外、本データでは 0",
        },
        "differences_with_sna_gdp": {
            "法人企業統計の付加価値合計 (2023)": "約 318兆円",
            "SNA GDP 2023 (名目)":            "約 588兆円",
            "差 270兆円の主な内訳": [
                "帰属家賃 約60兆円",
                "個人事業主の所得 約35兆円",
                "公務 約30兆円",
                "金融保険業 約28兆円",
                "公立学校・国公立病院 約40兆円",
                "減価償却 約50兆円",
                "その他 (純間接税, NPO等) 約27兆円",
            ],
            "注意": "法人企業統計の付加価値は『営利法人セクターのみ』。SNA GDP と直接比較するのは誤り。",
        },
        "rebound_vs_structural": {
            "注意": "2020 は COVID 底のため、運輸・宿泊飲食・小売の伸びはリバウンド成分が大きい。構造改善とリバウンドを切り分けるには 2019 比較や利益率(margin) を参照すべき。",
        },
        "size_definition_mismatch": {
            "注意": "経済センサス (従業者規模) と 法人企業統計 (資本金階級) は規模の定義が違う。"
                    "経済センサスの大企業=50人+ ≈ 16万事業所、法人企業統計の大企業=資本金10億+ ≈ 4,521社 と 34倍の差。"
                    "1社あたり利益を算出する際は分子分母を同じデータソースから取ること (本MCPは法人企業統計内で完結)。",
        },
    }, ensure_ascii=False, indent=2)
