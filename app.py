"""GDP Distribution — 日本経済 セクター別・規模別 分布分析."""
from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st
import pandas as pd
import plotly.graph_objects as go

sys.path.insert(0, str(Path(__file__).resolve().parent))

from modules.data_loader import load_census, load_corporate_stats, load_gdp, load_jsic_sectors
from modules.transformer import (
    CAPITAL_CLASSES,
    build_gdp_matrix,
    build_profit_matrix,
    build_company_count_matrix,
    build_employee_count_matrix,
    build_per_company_matrix,
    build_per_person_matrix,
    build_per_company_profit_matrix,
    build_per_person_profit_matrix,
    build_corp_company_count_by_capital,
    build_corp_employee_count_by_capital,
    build_corp_value_added_by_capital,
    build_corp_profit_by_capital,
    build_corp_per_company_va_by_capital,
    build_corp_per_person_va_by_capital,
    build_corp_per_company_profit_by_capital,
    build_corp_per_person_profit_by_capital,
    add_sector_names,
)
from modules.heatmap_builder import (
    build_gdp_heatmap,
    build_profit_heatmap,
    build_profit_rate_heatmap,
    build_company_count_heatmap,
    build_employee_count_heatmap,
    build_per_unit_heatmap,
)
from modules.insights import compute_sector_mismatches

DATA_DIR   = Path(__file__).parent / "data" / "processed"
CONFIG_DIR = Path(__file__).parent / "config"

st.set_page_config(
    page_title="日本経済 GDP分布分析",
    page_icon=":bar_chart:",
    layout="wide",
)

# ─── Sidebar ────────────────────────────────────────────────
with st.sidebar:
    st.title("フィルタ")

    db_source = st.radio(
        "規模区分のデータソース",
        [
            "経済センサス（従業者50人で区切る・約478万事業所）",
            "法人企業統計（資本金10億円で区切る・約282万法人）",
        ],
        index=0,
        help=(
            "経済センサス: 事業所ベース。大企業=従業者50人以上。\n"
            "法人企業統計: 法人ベース。大企業=資本金10億円以上。\n"
            "母集団・規模の定義が根本的に違うため、選んだソース内で全指標が完結する。"
        ),
        key="db_source_radio",
    )
    USE_CORP = db_source.startswith("法人企業統計")

    normalize = st.checkbox("正規化表示 (0–100%)", value=False,
                            help="各指標を0–100%に正規化して相対的な大小を比較します")

    unit_scale = st.radio("表示単位", ["十億円", "兆円"], horizontal=True,
                          help="ヒートマップの金額単位を切替")

    sectors_df = load_jsic_sectors(CONFIG_DIR / "jsic_sectors.yaml")
    all_sector_names = sectors_df["name_ja"].tolist()
    sel_sectors = st.multiselect("業種フィルタ（空=全業種）", options=all_sector_names, default=[])

    if USE_CORP:
        sel_sizes = st.multiselect(
            "資本金階級フィルタ（6区分）",
            options=CAPITAL_CLASSES,
            default=CAPITAL_CLASSES,
        )
    else:
        sel_sizes = st.multiselect(
            "規模フィルタ",
            options=["大企業", "中小企業"],
            default=["大企業", "中小企業"],
        )
    if not sel_sizes:
        st.warning("規模を1つ以上選択してください")
        st.stop()

# ─── Data loading (cached) ──────────────────────────────────
@st.cache_data(show_spinner="データ読込中...")
def _load_all(data_dir: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    d = Path(data_dir)
    return load_census(d), load_corporate_stats(d), load_gdp(d)


@st.cache_data(show_spinner="行列構築中...")
def _build_matrices(
    census_json: str,
    corp_json: str,
    gdp_json: str,
    sectors_json: str,
) -> dict[str, pd.DataFrame]:
    df_census = pd.read_json(census_json, orient="split")
    df_corp   = pd.read_json(corp_json,   orient="split")
    df_gdp    = pd.read_json(gdp_json,    orient="split")
    sectors   = pd.read_json(sectors_json, orient="split")

    raw = {
        # 経済センサスベース（従業者規模 大/中の2列）
        "gdp":         build_gdp_matrix(df_census, df_gdp),
        "profit":      build_profit_matrix(df_corp),
        "count":       build_company_count_matrix(df_census),
        "emp":         build_employee_count_matrix(df_census),
        "per_co":      build_per_company_matrix(df_census),
        "per_pp":      build_per_person_matrix(df_census),
        "per_co_pft":  build_per_company_profit_matrix(df_census, df_corp),
        "per_pp_pft":  build_per_person_profit_matrix(df_census, df_corp),
        # 法人企業統計ベース（資本金階級 6列）
        "corp_count":      build_corp_company_count_by_capital(df_corp),
        "corp_emp":        build_corp_employee_count_by_capital(df_corp),
        "corp_va":         build_corp_value_added_by_capital(df_corp),
        "corp_profit":     build_corp_profit_by_capital(df_corp),
        "corp_per_co_va":  build_corp_per_company_va_by_capital(df_corp),
        "corp_per_pp_va":  build_corp_per_person_va_by_capital(df_corp),
        "corp_per_co_pft": build_corp_per_company_profit_by_capital(df_corp),
        "corp_per_pp_pft": build_corp_per_person_profit_by_capital(df_corp),
    }
    # Attach sector names as index
    named = {k: add_sector_names(v, sectors) for k, v in raw.items()}
    return named


try:
    df_census, df_corp, df_gdp = _load_all(str(DATA_DIR))
except Exception as e:
    st.error(f"データ読込エラー: {e}")
    st.info(
        "先に `python scripts/build_estat_dataset.py --year 2023` を実行して "
        "e-Stat から実データを取得してください（要 ESTAT_API_KEY）。"
    )
    st.stop()

# Serialize for cache key stability
matrices = _build_matrices(
    df_census.to_json(orient="split"),
    df_corp.to_json(orient="split"),
    df_gdp.to_json(orient="split"),
    sectors_df.to_json(orient="split"),
)


def _apply_filters(m: pd.DataFrame) -> pd.DataFrame:
    """Apply sector & size filters."""
    if sel_sectors:
        m = m[m.index.isin(sel_sectors)]
    cols = [s for s in sel_sizes if s in m.columns]
    return m[cols] if cols else m


def _scale(m: pd.DataFrame) -> tuple[pd.DataFrame, str, str]:
    """Apply unit conversion and optional normalization.
    Source unit: billion JPY (10億円 per 1 unit)
      → 億円: × 10
      → 兆円: / 1,000 (1兆円 = 1,000 billion JPY)
    """
    if unit_scale == "兆円":
        m = m / 1_000
        unit_label, fmt = "兆円", ".1f"
    else:
        unit_label, fmt = "十億円", ".0f"

    if normalize:
        total = m.values.sum()
        if total > 0:
            m = m / total * 100
        unit_label, fmt = "%", ".2f"
    return m, unit_label, fmt


# データソース切替: 法人企業統計選択時は count/emp/付加価値関連を法人企業統計版に差し替え
if USE_CORP:
    # 法人企業統計ベース・資本金階級6区分
    m_gdp        = _apply_filters(matrices["corp_va"])
    m_profit     = _apply_filters(matrices["corp_profit"])
    m_count      = _apply_filters(matrices["corp_count"])
    m_emp        = _apply_filters(matrices["corp_emp"])
    m_per_co     = _apply_filters(matrices["corp_per_co_va"])
    m_per_pp     = _apply_filters(matrices["corp_per_pp_va"])
    m_per_co_pft = _apply_filters(matrices["corp_per_co_pft"])
    m_per_pp_pft = _apply_filters(matrices["corp_per_pp_pft"])
else:
    m_gdp        = _apply_filters(matrices["gdp"])
    m_profit     = _apply_filters(matrices["profit"])
    m_count      = _apply_filters(matrices["count"])
    m_emp        = _apply_filters(matrices["emp"])
    m_per_co     = _apply_filters(matrices["per_co"])
    m_per_pp     = _apply_filters(matrices["per_pp"])
    m_per_co_pft = _apply_filters(matrices["per_co_pft"])
    m_per_pp_pft = _apply_filters(matrices["per_pp_pft"])

if m_gdp.empty:
    st.warning("フィルタ条件に該当するデータがありません。")
    st.stop()

# Scale GDP and profit (money)
m_gdp_s,    gdp_unit, _    = _scale(m_gdp)
m_profit_s, pft_unit, _    = _scale(m_profit)

# ─── KPI Bar ─────────────────────────────────────────────────
st.title("日本経済 セクター別・規模別 分布分析")
_src_label = "法人企業統計ベース（資本金10億で区切る）" if USE_CORP else "経済センサスベース（従業者50人で区切る）"
st.caption(
    f"JSIC大分類（約19業種） × 企業規模（大企業・中小企業）で分解 — "
    f"**現在のデータソース: {_src_label}**"
)
with st.expander("データソース・前提・近似", expanded=False):
    st.markdown(
        """
### 業種分類
- 日本標準産業分類（JSIC）大分類 A〜S（19業種）。東証 JPX33業種ではない。
- 一部業種（生活関連・娯楽 N、複合サービス Q 等）は SNA の経済活動分類に直接対応せず、近似または合算。

### サイドバーの「規模区分のデータソース」切替
**全指標が選んだソース内で完結**するように、2つの公的統計を切り替えられる：

| 項目 | 経済センサス（デフォルト） | 法人企業統計 |
|---|---|---|
| statsDataId | `0004006327` (2021) | `0003060791` (2023年度) |
| カバー範囲 | **民営事業所**（個人経営・法人すべて） | **営利法人のみ**（金融保険除く） |
| 集計単位 | 事業所（支店も別カウント） | 法人（企業単位） |
| 大企業の定義 | **従業者数 50人以上** | **資本金 10億円以上** |
| 中小企業の定義 | 50人未満 | 資本金 10億円未満 |
| 大企業の社数 | 約 16万事業所 | **約 4,521社** |
| 全体の母集団 | 約 478万事業所 | **約 282万法人** |

### 付加価値 vs GDP（KPI の値が大きく変わる理由）
データソース切替で「GDP寄与合計」の中身が変わり、合計値が **587兆円 ⇔ 317兆円** と動く：

| ソース選択 | KPI 表示 | 中身 | 合計 |
|---|---|---|---|
| 経済センサス | 「GDP寄与合計」 | **SNA（国民経済計算）の名目GDP** を経済センサスの付加価値比で規模別に按分 | **587兆円** |
| 法人企業統計 | 「付加価値合計（法人企業統計）」 | **法人企業統計の付加価値**（営業純益＋人件費＋支払利息＋賃借料＋租税公課） | **317兆円** |

両者の差 270兆円 ≈ 個人事業主・公務（S）・帰属家賃・家計サービス・金融保険業 など、**法人企業統計に含まれない経済活動の付加価値**。
法人企業統計の数字は「営利法人セクターだけの付加価値」を意味し、**国全体のGDPと混同しないこと**。

### 営業利益
- 法人企業統計 2023年度（金融業・保険業以外）。約 82兆円（全産業除く金融保険）。
- **金融業・保険業（J）は別表のため本データでは 0**。複合サービス（Q）・公務（S）・その他サービス（R）も 0。

### GDP（経済センサスモード時の按分元）
- 国民経済計算 経済活動別国内総生産（名目・暦年, `0004028489`）。
- 産業別までしかなく、企業規模別配分は経済センサスの付加価値比で按分（直接計測ではない）。

### 規模区分の近似
- 経済センサスの「大企業=50人以上」は便宜的閾値（中小企業基本法では業種により50〜300人と異なる）。
- 法人企業統計の「中小企業=10億円未満」には資本金1千万円未満の小法人 約209万社 も含む。
        """
    )

k1, k2, k3, k4, k5 = st.columns(5)

# データソースに応じてラベル・分母を動的に切替
if USE_CORP:
    k1_label = "付加価値合計（法人企業統計）"
    k1_help  = "法人企業統計の付加価値（営業純益＋人件費＋支払利息＋賃借料＋租税公課）合計。SNAのGDPとは別物。"
    k5_label = "利益率（営業利益／付加価値）"
    k5_help  = "営業利益 ÷ 法人企業統計の付加価値"
    k3_label = "法人数合計"
    k3_unit  = "法人"
    k4_label = "従業員数合計"
    k4_unit  = "人"
else:
    k1_label = "GDP寄与合計"
    k1_help  = "国民経済計算（SNA）の名目GDPを経済センサスの付加価値比で規模別に按分した合計。"
    k5_label = "平均利益率（GDP比）"
    k5_help  = "営業利益 ÷ GDP寄与額"
    k3_label = "事業所数合計"
    k3_unit  = "事業所"
    k4_label = "従業者数合計"
    k4_unit  = "人"

k1.metric(k1_label, f"{m_gdp.sum().sum() / 1_000:,.1f} 兆円", help=k1_help)
k2.metric("営業利益合計", f"{m_profit.sum().sum() / 1_000:,.1f} 兆円")
k3.metric(k3_label, f"{int(m_count.sum().sum()):,} {k3_unit}")
k4.metric(k4_label, f"{int(m_emp.sum().sum()):,} {k4_unit}")
k5.metric(k5_label,
          f"{m_profit.sum().sum() / m_gdp.sum().sum() * 100:.1f}%"
          if m_gdp.sum().sum() > 0 else "N/A",
          help=k5_help)

st.divider()

# ─── Tabs ─────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["4ヒートマップ", "詳細テーブル", "インサイト"])

# ── Tab 1: 4 Heatmaps ───────────────────────────────────────
with tab1:
    col1, col2 = st.columns(2)

    # データソースに応じて見出しを切替
    gdp_title  = f"付加価値（法人企業統計）[{gdp_unit}]" if USE_CORP else f"GDP寄与額（付加価値）[{gdp_unit}]"
    cnt_title  = "法人数分布 [法人]" if USE_CORP else "事業所数分布 [事業所]"
    emp_title  = "従業員数分布 [人]" if USE_CORP else "従業者数分布 [人]"
    rate_label = "GDP比率 (%)" if not USE_CORP else "付加価値比率 (%)"

    with col1:
        st.subheader(gdp_title)
        fig_gdp = build_gdp_heatmap(m_gdp_s, title="")
        st.plotly_chart(fig_gdp, use_container_width=True)

        st.subheader(cnt_title)
        fig_cnt = build_company_count_heatmap(m_count, title="")
        st.plotly_chart(fig_cnt, use_container_width=True)

        st.subheader(emp_title)
        fig_emp = build_employee_count_heatmap(m_emp, title="")
        st.plotly_chart(fig_emp, use_container_width=True)

    with col2:
        profit_mode = st.radio(
            "利益表示モード",
            ["金額", rate_label],
            horizontal=True,
            key="profit_mode_radio",
        )
        if profit_mode == rate_label:
            denom_label = "付加価値" if USE_CORP else "GDP"
            st.subheader(f"利益率（営業利益/{denom_label}比）[%]")
            profit_cols = [c for c in m_profit.columns if c in m_gdp.columns]
            m_profit_rate = m_profit[profit_cols].div(
                m_gdp[profit_cols].replace(0, float("nan"))
            ).fillna(0) * 100
            fig_pft = build_profit_rate_heatmap(m_profit_rate, title="")
        else:
            st.subheader(f"利益（営業利益）[{pft_unit}]")
            fig_pft = build_profit_heatmap(m_profit_s, title="")
        st.plotly_chart(fig_pft, use_container_width=True)

        st.subheader("1社あたり / 1人あたり指標")
        metric_choice = st.radio(
            "指標を選択",
            [
                "1社あたり付加価値 [百万円/社]",
                "1人あたり付加価値 [万円/人]",
                "1社あたり営業利益 [百万円/社]",
                "1人あたり営業利益 [万円/人]",
            ],
            horizontal=True,
            key="per_unit_radio",
        )
        PER_UNIT_MAP = {
            "1社あたり付加価値 [百万円/社]": (m_per_co,     "百万円/社"),
            "1人あたり付加価値 [万円/人]":  (m_per_pp,     "万円/人"),
            "1社あたり営業利益 [百万円/社]": (m_per_co_pft, "百万円/社"),
            "1人あたり営業利益 [万円/人]":  (m_per_pp_pft, "万円/人"),
        }
        m_per, per_unit = PER_UNIT_MAP[metric_choice]
        fig_per = build_per_unit_heatmap(m_per, title="", unit=per_unit)
        st.plotly_chart(fig_per, use_container_width=True)

# ── Tab 2: Detail Table ──────────────────────────────────────
with tab2:
    metric_opts = {
        "GDP寄与額（付加価値）[億円]":    m_gdp,
        "利益（営業利益）[億円]":         m_profit,
        "企業数 [社]":                   m_count,
        "従業者数 [人]":                 m_emp,
        "1社あたり付加価値 [百万円/社]":  m_per_co,
        "1人あたり付加価値 [万円/人]":   m_per_pp,
        "1社あたり営業利益 [百万円/社]":  m_per_co_pft,
        "1人あたり営業利益 [万円/人]":   m_per_pp_pft,
    }
    sel_metric = st.selectbox("表示指標", options=list(metric_opts.keys()))
    tbl = metric_opts[sel_metric].copy()
    tbl.insert(0, "合計", tbl.sum(axis=1))
    tbl = tbl.sort_values("合計", ascending=False)

    # Format numbers
    def _fmt(v: float) -> str:
        return f"{v:,.1f}"

    st.dataframe(tbl.style.format(_fmt), use_container_width=True)

    csv_bytes = tbl.to_csv(encoding="utf-8-sig").encode("utf-8-sig")
    st.download_button(
        "CSVダウンロード",
        data=csv_bytes,
        file_name=f"{sel_metric}.csv",
        mime="text/csv",
    )

# ── Tab 3: Insights ──────────────────────────────────────────
with tab3:
    st.subheader("セクター横断ミスマッチ分析")
    st.caption(
        "GDP・利益・企業数の分布のズレを検出します。"
        "各セクションはしきい値（分位数）ベースで上位セクターを抽出しています。"
    )

    findings = compute_sector_mismatches(
        matrices["gdp"],
        matrices["profit"],
        matrices["count"],
        matrices["per_co"],
    )

    DESCRIPTIONS = {
        "GDP大・利益薄セクター":
            "GDP占有率は高いが営業利益率（GDP比）が低い業種。価値を生み出しているが利益が残りにくい。",
        "利益大・GDP寄与小セクター":
            "GDP寄与は相対的に小さいが利益占有率が高い業種。少ない付加価値で大きな利益を確保。",
        "大企業利益集中セクター":
            "大企業の利益シェアが付加価値シェアを大幅に上回る業種。大企業が利益を取りやすい構造。",
        "10億+利益集中セクター":
            "資本金10億以上法人の利益シェアが付加価値シェアを大幅に上回る業種。",
        "下位規模GDP支援セクター":
            "下位規模の法人/事業所がGDP（付加価値）を支えているにもかかわらず、利益が薄い業種。",
        "企業数多・生産性低セクター":
            "企業数占有率が高いが1社あたり付加価値が低い業種。企業数の多さが生産性に直結していない。",
    }

    for name, df_finding in findings.items():
        with st.expander(f"{name}  （{len(df_finding)}業種）", expanded=True):
            st.caption(DESCRIPTIONS.get(name, ""))
            if df_finding.empty:
                st.info("該当業種なし")
            else:
                st.dataframe(df_finding.style.format("{:.2f}"), use_container_width=True)
