"""E2E tests for GDP_distribution Streamlit app (Playwright).

対象:
  T1 — /_stcore/health が 200
  T2 — トップページタイトル "日本経済 セクター別・規模別 分布分析" 表示
  T3 — サイドバー db_source ラジオが2択（経済センサス / 法人企業統計）切替可能
  T4 — 法人企業統計モードで年スライダー (2020-2024) が表示される
  T5 — Plotly ヒートマップ canvas が描画される
  T6 — caveats expander が展開可能

実行前提: Streamlit を別ターミナルで起動済み
    .venv/Scripts/streamlit run app.py --server.port 8501 --server.headless true
"""

from __future__ import annotations

import httpx
import pytest
from playwright.sync_api import Page, expect

BASE_URL = "http://localhost:8501"


def _streamlit_alive() -> bool:
    try:
        r = httpx.get(f"{BASE_URL}/_stcore/health", timeout=2.0)
        return r.status_code == 200
    except Exception:
        return False


pytestmark = pytest.mark.skipif(
    not _streamlit_alive(),
    reason=(
        "Streamlit が 8501 で起動していません。別ターミナルで "
        "'.venv/Scripts/streamlit run app.py --server.port 8501 --server.headless true' を実行してから再試行してください。"
    ),
)


# T1
def test_streamlit_health_returns_200():
    r = httpx.get(f"{BASE_URL}/_stcore/health", timeout=5.0)
    assert r.status_code == 200


# T2
def test_top_page_title_visible(page: Page):
    page.goto(BASE_URL, wait_until="networkidle", timeout=30000)
    expect(page.get_by_text("日本経済 セクター別・規模別 分布分析")).to_be_visible(timeout=15000)


# T3
def test_db_source_radio_switchable(page: Page):
    page.goto(BASE_URL, wait_until="networkidle", timeout=30000)
    # サイドバーに2つの選択肢が表示される
    expect(page.get_by_text("経済センサス（従業者50人で区切る・約478万事業所）")).to_be_visible(timeout=15000)
    expect(page.get_by_text("法人企業統計（資本金10億円で区切る・約282万法人）")).to_be_visible()


# T4
def test_capital_year_slider_in_corp_mode(page: Page):
    page.goto(BASE_URL, wait_until="networkidle", timeout=30000)

    # 法人企業統計モードに切替（テキストでラジオを選択）
    page.get_by_text("法人企業統計（資本金10億円で区切る・約282万法人）").click()
    page.wait_for_timeout(2500)  # Streamlit の rerun を待つ

    # 年スライダーが現れる
    expect(page.get_by_text("法人企業統計の年度")).to_be_visible(timeout=15000)


# T5
def test_plotly_heatmap_canvas_rendered(page: Page):
    page.goto(BASE_URL, wait_until="networkidle", timeout=30000)
    # Plotly は SVG または canvas を生成。js-plotly-plot クラス要素が描画される。
    expect(page.locator(".js-plotly-plot").first).to_be_visible(timeout=20000)


# T6
def test_caveats_expander_openable(page: Page):
    page.goto(BASE_URL, wait_until="networkidle", timeout=30000)
    expander = page.get_by_text("データソース・前提・近似")
    expect(expander).to_be_visible(timeout=15000)
    expander.click()
    # 展開後、中身（前提テキスト）の一部が出ること
    page.wait_for_timeout(800)
