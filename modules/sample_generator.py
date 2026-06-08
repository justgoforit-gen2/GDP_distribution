"""
DEPRECATED. オフライン/APIキー無し環境向けのフォールバック専用。
生成データは「全業種で企業数が等分」など現実に反した合成値である（line 103 参照）。
通常運用では scripts/build_estat_dataset.py で e-Stat から実データを取得すること。
出力スキーマは旧 JPX33 × 3規模 のまま残しているが、現行アプリは JSIC × 2規模 を要求するため
このスクリプトをそのまま流すとアプリは起動しない。

Synthetic sample data generator for GDP_distribution.
Produces structurally realistic data for all 33 JPX sectors × 3 size categories.
seed=42 for reproducibility.

Size distribution priors (approximate real-world Japanese economy):
  大企業:   利益72%, 付加価値65%, 雇用48%, 企業数 2%
  中小企業: 利益25%, 付加価値28%, 雇用42%, 企業数38%
  個人事業主: 利益 3%, 付加価値 7%, 雇用10%, 企業数60%
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path

# ------------------------------------------------------------
# Sector relative weights: (gdp_weight, profit_weight, employment_weight)
# Proportional to approximate real-world contribution of each JPX33 sector
# ------------------------------------------------------------
SECTOR_WEIGHTS: dict[int, tuple[float, float, float]] = {
    1:  (0.008, 0.004, 0.012),  # 水産・農林業
    2:  (0.003, 0.004, 0.002),  # 鉱業
    3:  (0.060, 0.025, 0.080),  # 建設業
    4:  (0.025, 0.020, 0.025),  # 食料品
    5:  (0.008, 0.006, 0.010),  # 繊維製品
    6:  (0.005, 0.004, 0.005),  # パルプ・紙
    7:  (0.035, 0.042, 0.030),  # 化学
    8:  (0.020, 0.038, 0.015),  # 医薬品
    9:  (0.010, 0.008, 0.005),  # 石油・石炭製品
    10: (0.007, 0.008, 0.007),  # ゴム製品
    11: (0.010, 0.009, 0.010),  # ガラス・土石製品
    12: (0.015, 0.010, 0.015),  # 鉄鋼
    13: (0.010, 0.009, 0.010),  # 非鉄金属
    14: (0.015, 0.012, 0.018),  # 金属製品
    15: (0.040, 0.045, 0.035),  # 機械
    16: (0.055, 0.062, 0.050),  # 電気機器
    17: (0.070, 0.065, 0.060),  # 輸送用機器
    18: (0.010, 0.012, 0.010),  # 精密機器
    19: (0.015, 0.012, 0.018),  # その他製品
    20: (0.025, 0.015, 0.012),  # 電気・ガス業
    21: (0.030, 0.018, 0.040),  # 陸運業
    22: (0.008, 0.012, 0.005),  # 海運業
    23: (0.005, 0.006, 0.006),  # 空運業
    24: (0.010, 0.008, 0.010),  # 倉庫・運輸関連業
    25: (0.080, 0.095, 0.065),  # 情報・通信業
    26: (0.065, 0.040, 0.070),  # 卸売業
    27: (0.055, 0.025, 0.090),  # 小売業
    28: (0.040, 0.055, 0.025),  # 銀行業
    29: (0.012, 0.025, 0.010),  # 証券、商品先物取引業
    30: (0.015, 0.030, 0.010),  # 保険業
    31: (0.020, 0.035, 0.012),  # その他金融業
    32: (0.045, 0.070, 0.015),  # 不動産業
    33: (0.100, 0.040, 0.130),  # サービス業
}

SECTOR_NAMES: dict[int, str] = {
    1: "水産・農林業", 2: "鉱業", 3: "建設業", 4: "食料品", 5: "繊維製品",
    6: "パルプ・紙", 7: "化学", 8: "医薬品", 9: "石油・石炭製品", 10: "ゴム製品",
    11: "ガラス・土石製品", 12: "鉄鋼", 13: "非鉄金属", 14: "金属製品", 15: "機械",
    16: "電気機器", 17: "輸送用機器", 18: "精密機器", 19: "その他製品",
    20: "電気・ガス業", 21: "陸運業", 22: "海運業", 23: "空運業",
    24: "倉庫・運輸関連業", 25: "情報・通信業", 26: "卸売業", 27: "小売業",
    28: "銀行業", 29: "証券、商品先物取引業", 30: "保険業", 31: "その他金融業",
    32: "不動産業", 33: "サービス業",
}

SIZES = ["大企業", "中小企業", "個人事業主"]

# Value-added split (large, sme, sole)
VA_SPLIT    = (0.65, 0.28, 0.07)
PROFIT_SPLIT = (0.72, 0.25, 0.03)
COUNT_SPLIT  = (0.02, 0.38, 0.60)
EMP_SPLIT    = (0.48, 0.42, 0.10)

# Total scale anchors (billion JPY, approximate Japan 2022)
TOTAL_GDP_BILLION    = 550_000   # ~550兆円
TOTAL_PROFIT_BILLION = 80_000    # ~80兆円
TOTAL_COMPANIES      = 3_800_000 # ~380万社
TOTAL_EMPLOYEES      = 65_000_000 # ~6500万人


def _noise(rng: np.random.Generator, base: float, pct: float = 0.10) -> float:
    """Add ±pct% noise to base value."""
    return float(base * (1 + rng.uniform(-pct, pct)))


def generate_census_csv(output_path: Path, seed: int = 42) -> None:
    rng = np.random.default_rng(seed)
    gdp_weights = np.array([SECTOR_WEIGHTS[c][0] for c in range(1, 34)])
    gdp_weights /= gdp_weights.sum()

    emp_weights = np.array([SECTOR_WEIGHTS[c][2] for c in range(1, 34)])
    emp_weights /= emp_weights.sum()

    rows = []
    for code in range(1, 34):
        gw = gdp_weights[code - 1]
        ew = emp_weights[code - 1]

        sector_va  = TOTAL_GDP_BILLION * gw
        sector_emp = TOTAL_EMPLOYEES   * ew
        sector_cos = TOTAL_COMPANIES   * (1 / 33)  # rough equal split of companies before size split

        for i, size in enumerate(SIZES):
            va   = _noise(rng, sector_va   * VA_SPLIT[i])
            emp  = _noise(rng, sector_emp  * EMP_SPLIT[i])
            cnt  = _noise(rng, sector_cos  * COUNT_SPLIT[i])
            sales = va * _noise(rng, 2.5, 0.3)  # sales ~2-3x value-added
            rows.append({
                "jpx33_code":             code,
                "jpx33_name":             SECTOR_NAMES[code],
                "company_size_category":  size,
                "company_count":          max(1, int(cnt)),
                "employee_count":         max(1, int(emp)),
                "sales_billion_jpy":      round(sales, 2),
                "value_added_billion_jpy": round(va, 2),
                "survey_year":            2022,
            })

    pd.DataFrame(rows).to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"[OK] census_sample.csv -> {output_path} ({len(rows)} rows)")


def generate_corporate_stats_csv(output_path: Path, seed: int = 42) -> None:
    rng = np.random.default_rng(seed + 1)
    profit_weights = np.array([SECTOR_WEIGHTS[c][1] for c in range(1, 34)])
    profit_weights /= profit_weights.sum()

    rows = []
    for code in range(1, 34):
        pw = profit_weights[code - 1]
        sector_profit = TOTAL_PROFIT_BILLION * pw

        for i, size in enumerate(["大企業", "中小企業"]):  # 個人事業主は法人企業統計外
            op  = _noise(rng, sector_profit * PROFIT_SPLIT[i])
            net = op * _noise(rng, 0.72, 0.10)
            assets = op * _noise(rng, 8.0, 0.30)
            equity = assets * _noise(rng, 0.38, 0.15)
            rows.append({
                "jpx33_code":                 code,
                "jpx33_name":                 SECTOR_NAMES[code],
                "company_size_category":      size,
                "operating_profit_billion_jpy": round(op, 2),
                "net_profit_billion_jpy":      round(net, 2),
                "total_assets_billion_jpy":    round(assets, 2),
                "equity_billion_jpy":          round(equity, 2),
                "survey_year":                 2022,
            })

    pd.DataFrame(rows).to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"[OK] corporate_stats_sample.csv -> {output_path} ({len(rows)} rows)")


def generate_gdp_csv(output_path: Path, seed: int = 42) -> None:
    rng = np.random.default_rng(seed + 2)
    gdp_weights = np.array([SECTOR_WEIGHTS[c][0] for c in range(1, 34)])
    gdp_weights /= gdp_weights.sum()

    rows = []
    for code in range(1, 34):
        gw  = gdp_weights[code - 1]
        gdp = _noise(rng, TOTAL_GDP_BILLION * gw)
        rows.append({
            "jpx33_code":                  code,
            "jpx33_name":                  SECTOR_NAMES[code],
            "gdp_contribution_billion_jpy": round(gdp, 2),
            "gdp_share_pct":               round(gw * 100, 4),
            "survey_year":                 2022,
        })

    pd.DataFrame(rows).to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"[OK] gdp_by_industry_sample.csv -> {output_path} ({len(rows)} rows)")
