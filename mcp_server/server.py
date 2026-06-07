"""GDP_distribution MCP Server — 業種×企業規模×年度 の経済データを公開.

提供ツール:
  - gdp_list_metadata    : 業種・規模・年度・指標の一覧
  - gdp_query_metric     : 単一セルの値
  - gdp_get_timeseries   : 5年時系列
  - gdp_compare_growth   : 2020→2024 成長率トップN
  - gdp_sector_breakdown : 指定年の業種別 6規模ブレイクダウン
  - gdp_get_caveats      : データソースの前提・近似・制約

データロジックは _tools.py に分離 (MCP 非依存)。本ファイルは FastMCP の薄いラッパー。

起動 (Claude Code 設定例 ~/.claude/settings.json):
  {
    "mcpServers": {
      "gdp": {
        "command": "python",
        "args": ["C:/Users/justg/Documents/python_projects/dify_projects/GDP_distribution/mcp_server/server.py"]
      }
    }
  }

ui-hub 設定: ui-hub/app.py の mcp_servers dict に1エントリ追加.
"""

from __future__ import annotations

import logging

from mcp.server.fastmcp import FastMCP

from _tools import (
    compare_growth,
    get_caveats,
    get_timeseries,
    list_metadata,
    query_metric,
    sector_breakdown,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

mcp = FastMCP(
    "GDP分布分析",
    instructions=(
        "日本の業種×企業規模×年度別の経済指標 (営業利益・付加価値・法人数・従業員数) を分析するツール。"
        "データソース: 法人企業統計 2020-2024年度 (資本金階級6区分) + 経済センサス 2021。"
        "業種は JSIC 大分類 A-S (19業種)。"
        "最初に gdp_list_metadata で利用可能なコードを確認してから個別ツールを呼ぶこと。"
        "金融保険業 J・複合サービス Q・公務 S はデータ対象外で 0 が返る (gdp_get_caveats 参照)。"
    ),
)


@mcp.tool()
def gdp_list_metadata() -> str:
    """利用可能な業種コード(JSIC)・資本金階級・年度・指標一覧を返す."""
    return list_metadata()


@mcp.tool()
def gdp_query_metric(
    jsic_code: str,
    capital_class: str,
    year: int,
    metric: str = "operating_profit",
) -> str:
    """単一セルの値を返す.

    Args:
        jsic_code: JSIC 大分類 1文字 (A〜S)
        capital_class: 資本金階級 (例: "10億+", "1億-10億", "1千万未満")
        year: 年度 (2020〜2024)
        metric: 指標キー (operating_profit / value_added / company_count / employee_count / net_profit / total_assets)
    """
    return query_metric(jsic_code, capital_class, year, metric)


@mcp.tool()
def gdp_get_timeseries(
    jsic_code: str,
    capital_class: str,
    metric: str = "operating_profit",
) -> str:
    """指定セルの5年時系列を返す (2020-2024)."""
    return get_timeseries(jsic_code, capital_class, metric)


@mcp.tool()
def gdp_compare_growth(
    metric: str = "operating_profit",
    top_n: int = 10,
    min_value_billion_jpy: float = 100.0,
) -> str:
    """5年間の成長率トップN (業種×資本金階級グリッド).

    Args:
        metric: 比較する指標
        top_n: 上位件数
        min_value_billion_jpy: 最終年のフィルタしきい値
    """
    return compare_growth(metric, top_n, min_value_billion_jpy)


@mcp.tool()
def gdp_sector_breakdown(year: int, metric: str = "operating_profit") -> str:
    """指定年の業種別 6規模ブレイクダウンを返す."""
    return sector_breakdown(year, metric)


@mcp.tool()
def gdp_get_caveats() -> str:
    """データソースの前提・近似・制約一覧を返す."""
    return get_caveats()


if __name__ == "__main__":
    mcp.run()
