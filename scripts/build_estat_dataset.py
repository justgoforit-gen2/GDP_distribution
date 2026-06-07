"""Build GDP_distribution dataset from real e-Stat tables.

Pulls 3 datasets:
  - 経済センサス‐活動調査 2021 (0004006327): 産業大分類 × 従業者規模 × 事業所数・従事者数・純付加価値
  - 国民経済計算 (0004028489): 経済活動別国内総生産 名目 暦年（2023年データ含む）
  - 法人企業統計調査 年度次 金融業・保険業以外 (0003060791):
        業種 × 資本金階級 × 営業利益等（最新年度）

Writes 3 CSVs to data/processed/:
  - census.csv               (19 sectors × 2 sizes = ≤38 rows)
  - corporate_stats.csv      (19 sectors × 2 sizes = ≤38 rows; 金融保険業 J は空)
  - gdp_by_industry.csv      (19 sectors)

Run:
  $env:ESTAT_API_KEY = "<key>"  # or use estat_client/.env
  uv run python scripts/build_estat_dataset.py --year 2023
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SIBLING_ESTAT_CLIENT = PROJECT_ROOT.parent / "estat_client"
sys.path.insert(0, str(SIBLING_ESTAT_CLIENT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from estat_client import EStatClient  # noqa: E402
from pull_estat_statsdata import _build_code_to_label_maps, _values_to_long_rows  # noqa: E402


# ============================================================
# StatsData IDs
# ============================================================
CENSUS_ID    = "0004006327"  # 経済センサス‐活動調査 2021 産業大分類×従業者規模×純付加価値
SNA_ID       = "0004028489"  # 国民経済計算 経済活動別国内総生産 名目-暦年
CORP_ID      = "0003060791"  # 法人企業統計調査 年度次 金融業・保険業以外


# ============================================================
# JSIC mapping helpers
# ============================================================
JSIC_NAMES = {
    "A": "農業，林業",
    "B": "漁業",
    "C": "鉱業，採石業，砂利採取業",
    "D": "建設業",
    "E": "製造業",
    "F": "電気・ガス・熱供給・水道業",
    "G": "情報通信業",
    "H": "運輸業，郵便業",
    "I": "卸売業，小売業",
    "J": "金融業，保険業",
    "K": "不動産業，物品賃貸業",
    "L": "学術研究，専門・技術サービス業",
    "M": "宿泊業，飲食サービス業",
    "N": "生活関連サービス業，娯楽業",
    "O": "教育，学習支援業",
    "P": "医療，福祉",
    "Q": "複合サービス事業",
    "R": "サービス業（他に分類されないもの）",
    "S": "公務（他に分類されるものを除く）",
}

# 経済センサス 2021 cat01 → JSIC
# 注意: censusでは A,Bは "AB 農林漁業" として level=1 統合済み (level=2 に A単独/B単独なし)
CENSUS_CAT01_TO_JSIC = {
    "AB": ["A", "B"],  # 農林漁業 → A,B に均等分割
    "C":  ["C"],
    "D":  ["D"],
    "E":  ["E"],
    "F":  ["F"],
    "G":  ["G"],
    "H":  ["H"],
    "I":  ["I"],
    "J":  ["J"],
    "K":  ["K"],
    "L":  ["L"],
    "M":  ["M"],
    "N":  ["N"],
    "O":  ["O"],
    "P":  ["P"],
    "Q":  ["Q"],
    "R":  ["R"],
    # S 公務 は経済センサスでは「全産業（S_公務を除く）」なので存在しない
}

# SNA 経済活動別国内総生産 cat01 (level=4 大分類) → JSIC
SNA_CAT01_TO_JSIC = {
    "100": ["A", "B"],  # 農林水産業 → A,B 均等分割
    "140": ["C"],        # 鉱業
    "150": ["E"],        # 製造業
    "310": ["F"],        # 電気・ガス・水道・廃棄物処理業
    "340": ["D"],        # 建設業
    "350": ["I"],        # 卸売・小売業
    "380": ["H"],        # 運輸・郵便業
    "390": ["M"],        # 宿泊・飲食サービス業
    "400": ["G"],        # 情報通信業
    "430": ["J"],        # 金融・保険業
    "440": ["K"],        # 不動産業
    "470": ["L"],        # 専門・科学技術、業務支援サービス業
    "480": ["S"],        # 公務
    "490": ["O"],        # 教育
    "500": ["P"],        # 保健衛生・社会事業
    "510": ["Q", "R"],   # その他のサービス → Q,R 均等分割
}

# 法人企業統計 cat02 (業種) → JSIC
# 注意: 137 サービス業(集約) は 152/153/156/157/161 と重複するため除外。
# 結果として R その他サービス は法人企業統計データでは 0（Q,S,J と同じ扱い）。
# 合計母集団 ≈ 282万社（全産業除金融保険 2.99M のうち 94%カバー）
CORP_CAT02_TO_JSIC = {
    "105": ["A", "B"],  # 農林水産業(集約)
    "106": ["C"],        # 鉱業
    "107": ["D"],        # 建設業
    "108": ["E"],        # 製造業
    "135": ["F"],        # 電気業
    "136": ["F"],        # ガス・熱供給・水道業 (Fに加算)
    "142": ["G"],        # 情報通信業
    "134": ["H"],        # 運輸業、郵便業
    "129": ["I"],        # 卸売業・小売業
    "155": ["K"],        # 不動産業・物品賃貸業
    "161": ["L"],        # 学術研究、専門・技術サービス業
    "156": ["M"],        # 宿泊業、飲食サービス業
    "157": ["N"],        # 生活関連サービス業、娯楽業
    "153": ["O"],        # 教育、学習支援業
    "152": ["P"],        # 医療、福祉業
    # 137 サービス業(集約) は除外（上記サブ業種と重複）→ R = 0
    # J (金融保険) は別表のため 0、Q 複合サービス・S 公務は対象外
}

# サイズマッピング
# 経済センサス 従業者規模 cat02
CENSUS_LARGE_CODE = "6"      # 50人以上
CENSUS_SME_CODES = ["1", "2", "3", "4", "5"]  # 1-4, 5-9, 10-19, 20-29, 30-49

# 法人企業統計 資本金階級 cat03
# 大企業 = 資本金10億円以上、中小企業 = それ未満（全規模からの差分）
# 16: 1千万円未満（約209万社の小法人）も中小企業に編入することで、
# 法人企業統計の母集団 約290万社 と整合する
CORP_LARGE_CODE = "25"             # 10億円以上 (約4,688社)
CORP_SME_CODES = ["24", "19", "16"]  # 1億-10億 + 1千万-1億 + 1千万未満 = 約290万社


# ============================================================
# Helpers
# ============================================================
def _fetch_long(client: EStatClient, stats_data_id: str, *,
                filters: dict[str, str] | None = None) -> pd.DataFrame:
    """Fetch full statsData and return long-form DataFrame with dim labels + value."""
    meta = client.get_meta_info(stats_data_id)
    class_obj = meta.get("CLASS_OBJ", [])
    if isinstance(class_obj, dict):
        class_obj = [class_obj]
    dim_maps = _build_code_to_label_maps(class_obj)

    values = client.get_all_stats_data(stats_data_id, filters=filters or {})
    rows = _values_to_long_rows(values, dim_maps)
    df = pd.DataFrame(rows)
    # value → numeric
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    return df


def _info(msg: str) -> None:
    # avoid mojibake on Windows shells - write to stderr (PowerShell tends to handle it)
    print(msg, file=sys.stderr)


# ============================================================
# Pull functions
# ============================================================
def pull_census(client: EStatClient) -> pd.DataFrame:
    """Pull census (event 2021): 産業大分類 × 従業者規模 × 事業所数 / 従事者数 / 純付加価値"""
    _info(f"\n[census] fetching {CENSUS_ID} ...")
    df = _fetch_long(
        client, CENSUS_ID,
        filters={"cdArea": "00000"},  # 全国のみ
    )
    _info(f"[census] long-form rows: {len(df)}")

    # cat01 (産業大分類), cat02 (従業者規模), tab (表章項目)
    # サイズ集約: 6 → 大企業, 1+2+3+4+5 → 中小企業
    df["size_category"] = df["cat02"].map(
        lambda c: "大企業" if c == CENSUS_LARGE_CODE
        else ("中小企業" if c in CENSUS_SME_CODES else None)
    )
    df = df[df["size_category"].notna()].copy()

    # 産業大分類: cat01 が CENSUS_CAT01_TO_JSIC のキーに限定
    df = df[df["cat01"].isin(CENSUS_CAT01_TO_JSIC.keys())].copy()

    # tab(表章項目) コードを指標へ（codeは "102-2021" のように年号サフィックス付き）
    METRIC_MAP = {
        "102-2021": "company_count",
        "148-2021": "employee_count",
        "158-2021": "value_added_million_jpy",
    }
    df["metric"] = df["tab"].map(METRIC_MAP)
    df = df[df["metric"].notna()].copy()

    # cat01 を JSIC 1 件以上に展開（AB のような複合キーは均等分割）
    rows = []
    for _, r in df.iterrows():
        jsic_codes = CENSUS_CAT01_TO_JSIC[r["cat01"]]
        share = 1.0 / len(jsic_codes)
        for j in jsic_codes:
            rows.append({
                "jsic_code": j,
                "size_category": r["size_category"],
                "metric": r["metric"],
                "value": (r["value"] or 0) * share,
            })
    df2 = pd.DataFrame(rows)

    # 集計（同 jsic × size × metric を合算）
    agg = df2.groupby(["jsic_code", "size_category", "metric"], as_index=False)["value"].sum()
    pivot = agg.pivot_table(
        index=["jsic_code", "size_category"],
        columns="metric",
        values="value",
        aggfunc="sum",
        fill_value=0,
    ).reset_index()

    # 単位変換: 付加価値 (百万円 → billion JPY: ÷ 1000)
    pivot["value_added_billion_jpy"] = pivot.get("value_added_million_jpy", 0) / 1000.0

    # sales 列は census 0004006327 には無いので 0 を入れる（後で別表で取得するか、空のまま）
    pivot["sales_billion_jpy"] = 0.0

    # 必要列
    out = pd.DataFrame({
        "jsic_code":               pivot["jsic_code"],
        "jsic_name":               pivot["jsic_code"].map(JSIC_NAMES),
        "company_size_category":   pivot["size_category"],
        "company_count":           pivot.get("company_count", 0).astype("int64", errors="ignore").astype(int),
        "employee_count":          pivot.get("employee_count", 0).astype("int64", errors="ignore").astype(int),
        "sales_billion_jpy":       pivot["sales_billion_jpy"].round(2),
        "value_added_billion_jpy": pivot["value_added_billion_jpy"].round(2),
        "survey_year":             2021,
    })

    _info(f"[census] sectors: {out['jsic_code'].nunique()} / sizes: {sorted(out['company_size_category'].unique())}")
    _info(f"[census] total company_count: {out['company_count'].sum():,}")
    _info(f"[census] total employee_count: {out['employee_count'].sum():,}")
    _info(f"[census] total value_added: {out['value_added_billion_jpy'].sum():,.1f} billion JPY")
    return out


def pull_gdp_by_industry(client: EStatClient, year: int) -> pd.DataFrame:
    """Pull SNA: 経済活動別国内総生産 名目 暦年 year. Returns JSIC-coded GDP."""
    _info(f"\n[gdp] fetching {SNA_ID} year={year} ...")
    time_code = f"{year}000000"
    df = _fetch_long(
        client, SNA_ID,
        filters={"cdTime": time_code, "cdTab": "11"},  # tab=11 金額
    )
    _info(f"[gdp] long-form rows: {len(df)}")

    # cat01 大分類 (level=4) のみ
    df = df[df["cat01"].isin(SNA_CAT01_TO_JSIC.keys())].copy()

    rows = []
    for _, r in df.iterrows():
        jsic_codes = SNA_CAT01_TO_JSIC[r["cat01"]]
        share = 1.0 / len(jsic_codes)
        for j in jsic_codes:
            rows.append({
                "jsic_code": j,
                "value": (r["value"] or 0) * share,
            })
    df2 = pd.DataFrame(rows)
    agg = df2.groupby("jsic_code", as_index=False)["value"].sum()

    # 単位: SNA tab=11 (金額) は「10億円」表示 = billion JPY 相当（変換不要）
    agg["gdp_contribution_billion_jpy"] = agg["value"]

    total = agg["gdp_contribution_billion_jpy"].sum()
    agg["gdp_share_pct"] = agg["gdp_contribution_billion_jpy"] / total * 100.0 if total > 0 else 0

    # JSIC全業種にreindex（欠損は0）
    all_codes = list(JSIC_NAMES.keys())
    agg = agg.set_index("jsic_code").reindex(all_codes, fill_value=0).reset_index()

    out = pd.DataFrame({
        "jsic_code":                    agg["jsic_code"],
        "jsic_name":                    agg["jsic_code"].map(JSIC_NAMES),
        "gdp_contribution_billion_jpy": agg["gdp_contribution_billion_jpy"].round(2),
        "gdp_share_pct":                agg["gdp_share_pct"].round(4),
        "survey_year":                  year,
    })
    _info(f"[gdp] total: {out['gdp_contribution_billion_jpy'].sum():,.1f} billion JPY")
    return out


def pull_corporate_stats(client: EStatClient, fiscal_year: int) -> pd.DataFrame:
    """Pull 法人企業統計 年度次: 業種 × 資本金階級 × 営業利益等.
    fiscal_year=2024 → time code "20240" (2024年度)
    """
    _info(f"\n[corp] fetching {CORP_ID} fy={fiscal_year} ...")
    time_code = f"{fiscal_year}0"
    df = _fetch_long(
        client, CORP_ID,
        filters={"cdTime": time_code},
    )
    _info(f"[corp] long-form rows: {len(df)}")

    # cat03 (規模) フィルタ
    df["size_category"] = df["cat03"].map(
        lambda c: "大企業" if c == CORP_LARGE_CODE
        else ("中小企業" if c in CORP_SME_CODES else None)
    )
    df = df[df["size_category"].notna()].copy()

    # cat02 (業種) フィルタ
    df = df[df["cat02"].isin(CORP_CAT02_TO_JSIC.keys())].copy()

    # cat01 (調査項目) → metric
    METRIC_MAP = {
        "001": "corp_company_count",         # 母集団（法人数）社
        "072": "corp_employee_count",        # 期中平均従業員数 人
        "073": "value_added_corp_million_jpy",  # 付加価値（法人企業統計の定義）百万円
        "048": "operating_profit_million_jpy",
        "056": "net_profit_million_jpy",
        "022": "total_assets_million_jpy",
        "036": "capital_million_jpy",        # 資本金
        "037": "capital_reserve_million_jpy",  # 資本準備金
        "038": "earnings_reserve_million_jpy",  # 利益準備金
        "041": "retained_earnings_million_jpy",  # 繰越利益剰余金
    }
    df["metric"] = df["cat01"].map(METRIC_MAP)
    df = df[df["metric"].notna()].copy()

    # 展開
    rows = []
    for _, r in df.iterrows():
        jsic_codes = CORP_CAT02_TO_JSIC[r["cat02"]]
        share = 1.0 / len(jsic_codes)
        for j in jsic_codes:
            rows.append({
                "jsic_code": j,
                "size_category": r["size_category"],
                "metric": r["metric"],
                "value": (r["value"] or 0) * share,
            })
    df2 = pd.DataFrame(rows)
    agg = df2.groupby(["jsic_code", "size_category", "metric"], as_index=False)["value"].sum()
    pivot = agg.pivot_table(
        index=["jsic_code", "size_category"],
        columns="metric",
        values="value",
        aggfunc="sum",
        fill_value=0,
    ).reset_index()

    # 単位: 百万円 → billion JPY (÷ 1000)
    for col in [
        "operating_profit_million_jpy", "net_profit_million_jpy",
        "total_assets_million_jpy",
        "capital_million_jpy", "capital_reserve_million_jpy",
        "earnings_reserve_million_jpy", "retained_earnings_million_jpy",
        "value_added_corp_million_jpy",
        "corp_company_count", "corp_employee_count",
    ]:
        if col not in pivot.columns:
            pivot[col] = 0.0

    pivot["operating_profit_billion_jpy"]   = pivot["operating_profit_million_jpy"] / 1000.0
    pivot["net_profit_billion_jpy"]         = pivot["net_profit_million_jpy"] / 1000.0
    pivot["total_assets_billion_jpy"]       = pivot["total_assets_million_jpy"] / 1000.0
    pivot["value_added_corp_billion_jpy"]   = pivot["value_added_corp_million_jpy"] / 1000.0
    pivot["equity_billion_jpy"] = (
        pivot["capital_million_jpy"]
        + pivot["capital_reserve_million_jpy"]
        + pivot["earnings_reserve_million_jpy"]
        + pivot["retained_earnings_million_jpy"]
    ) / 1000.0

    # 全 JSIC × {大企業, 中小企業} の組合せにreindex（金融保険業 J は0で残す）
    all_codes = list(JSIC_NAMES.keys())
    index = pd.MultiIndex.from_product(
        [all_codes, ["大企業", "中小企業"]],
        names=["jsic_code", "company_size_category"],
    )
    pivot = pivot.rename(columns={"size_category": "company_size_category"})
    pivot = pivot.set_index(["jsic_code", "company_size_category"]).reindex(index, fill_value=0).reset_index()

    out = pd.DataFrame({
        "jsic_code":                       pivot["jsic_code"],
        "jsic_name":                       pivot["jsic_code"].map(JSIC_NAMES),
        "company_size_category":           pivot["company_size_category"],
        "corp_company_count":              pivot["corp_company_count"].astype(int),
        "corp_employee_count":             pivot["corp_employee_count"].astype(int),
        "value_added_corp_billion_jpy":    pivot["value_added_corp_billion_jpy"].round(2),
        "operating_profit_billion_jpy":    pivot["operating_profit_billion_jpy"].round(2),
        "net_profit_billion_jpy":          pivot["net_profit_billion_jpy"].round(2),
        "total_assets_billion_jpy":        pivot["total_assets_billion_jpy"].round(2),
        "equity_billion_jpy":              pivot["equity_billion_jpy"].round(2),
        "survey_year":                     fiscal_year,
    })

    _info(f"[corp] total corp_company_count: {out['corp_company_count'].sum():,}")
    _info(f"[corp] total corp_employee_count: {out['corp_employee_count'].sum():,}")
    _info(f"[corp] total value_added (corp def): {out['value_added_corp_billion_jpy'].sum():,.1f} billion JPY")
    _info(f"[corp] total operating_profit: {out['operating_profit_billion_jpy'].sum():,.1f} billion JPY")
    return out


# ============================================================
# CLI
# ============================================================
def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--year", type=int, default=2023,
                   help="GDP対象暦年（SNA, default 2023）。法人企業統計は同じ番号を年度として使用。")
    p.add_argument("--no-cache", action="store_true",
                   help="estat_client のディスクキャッシュを無効化")
    p.add_argument("--output-dir", type=str,
                   default=str(PROJECT_ROOT / "data" / "processed"),
                   help="CSV 出力先")
    args = p.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cache_dir = None if args.no_cache else (SIBLING_ESTAT_CLIENT / "cache")
    client = EStatClient(cache_dir=cache_dir)

    # 1) Census (always 2021 — 5年周期の最新確報)
    census = pull_census(client)
    census_path = output_dir / "census.csv"
    census.to_csv(census_path, index=False, encoding="utf-8-sig")
    _info(f"[census] wrote {census_path}")

    # 2) GDP by industry (SNA, args.year)
    gdp = pull_gdp_by_industry(client, args.year)
    gdp_path = output_dir / "gdp_by_industry.csv"
    gdp.to_csv(gdp_path, index=False, encoding="utf-8-sig")
    _info(f"[gdp] wrote {gdp_path}")

    # 3) Corporate stats (法人企業統計, args.year as fiscal year)
    corp = pull_corporate_stats(client, args.year)
    corp_path = output_dir / "corporate_stats.csv"
    corp.to_csv(corp_path, index=False, encoding="utf-8-sig")
    _info(f"[corp] wrote {corp_path}")

    _info("\n=== Summary ===")
    _info(f"census           rows: {len(census)}  sectors: {census['jsic_code'].nunique()}")
    _info(f"gdp_by_industry  rows: {len(gdp)}")
    _info(f"corporate_stats  rows: {len(corp)}  sectors: {corp['jsic_code'].nunique()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
