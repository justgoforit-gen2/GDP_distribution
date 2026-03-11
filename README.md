# 日本経済 GDP分布分析ダッシュボード

JPX33業種 × 企業規模（大企業・中小企業・個人事業主）で日本経済を分解し、
GDP寄与・利益・企業数・従業者数・生産性の分布をインタラクティブに可視化するStreamlitアプリです。

---

## スクリーンショット

| タブ | 内容 |
|------|------|
| 4ヒートマップ | GDP・利益・企業数・従業者数・1社あたり/1人あたり指標 |
| 詳細テーブル | 全指標のソート可能テーブル + CSVダウンロード |
| インサイト | 5種類のセクター横断ミスマッチ分析 |

---

## 機能

### ヒートマップ（Tab 1）
- **GDP寄与額（付加価値）** — Blues カラースケール、単位: 十億円/兆円
- **利益（営業利益）** — Greens、金額 ↔ GDP比率(%) のトグル付き
- **企業数分布** — Oranges、単位: 社
- **従業者数分布** — Purples、単位: 人
- **1社あたり / 1人あたり付加価値** — RdYlGn（中央値基準）

各ヒートマップは業種・規模の小計行/列（合計）を自動付加。セル内は数値のみ表示、ホバーで単位付き詳細表示。

### サイドバーフィルタ
- 表示単位切替（十億円 / 兆円）
- 正規化表示（0–100%）
- 業種フィルタ（JPX33業種から複数選択）
- 規模フィルタ（大企業 / 中小企業 / 個人事業主）

### KPIバー
GDP寄与合計 / 営業利益合計 / 企業数合計 / 従業者数合計 / 平均利益率(GDP比)

### インサイト分析（Tab 3）
| 分析名 | 内容 |
|--------|------|
| GDP大・利益薄セクター | GDP占有率は高いが利益率が低い |
| 利益大・GDP寄与小セクター | 少ない付加価値で高い利益を確保 |
| 大企業利益集中セクター | 大企業の利益シェアが付加価値シェアを上回る |
| 中小・個人GDP支援セクター | 中小・個人がGDPを支えるが利益が薄い |
| 企業数多・生産性低セクター | 企業数は多いが1社あたり付加価値が低い |

---

## セットアップ

### 必要環境
- Python 3.11 以上
- [uv](https://github.com/astral-sh/uv)（推奨）または pip

### インストール

```bash
# uv の場合
uv sync

# pip の場合
pip install -r requirements.txt
```

### サンプルデータ生成

```bash
uv run python scripts/generate_samples.py
```

`data/samples/` に以下の3ファイルが生成されます。

| ファイル | 行数 | 内容 |
|----------|------|------|
| `census_sample.csv` | 99 | 経済センサス（33業種×3規模） |
| `corporate_stats_sample.csv` | 66 | 法人企業統計（33業種×2規模） |
| `gdp_by_industry_sample.csv` | 33 | 国民経済計算GDP（業種単位） |

> **注意**: サンプルデータは合成データ（seed=42）です。実際の統計データと差し替える場合は下記スキーマを参照してください。

### アプリ起動

```bash
uv run streamlit run app.py
```

ブラウザで `http://localhost:8501` を開きます。

---

## ディレクトリ構成

```
GDP_distribution/
├── app.py                              # Streamlit エントリーポイント
├── pyproject.toml
├── .streamlit/config.toml
├── config/
│   ├── jpx33_sectors.yaml              # JPX33業種コード→名称マッピング（33件）
│   └── size_thresholds.yaml            # 企業規模区分定義
├── data/
│   └── samples/                        # 合成サンプルCSV（generate_samples.py で生成）
├── modules/
│   ├── data_loader.py                  # CSV読込・バリデーション
│   ├── transformer.py                  # CSV → JPX33×規模 行列変換
│   ├── heatmap_builder.py              # Plotly ヒートマップファクトリ
│   └── insights.py                     # ミスマッチ検出ロジック
└── scripts/
    └── generate_samples.py             # サンプルデータ生成スクリプト
```

---

## データスキーマ

実際の統計データを使用する場合、各CSVに以下の列が必要です。

### census_sample.csv（経済センサス）
```
jpx33_code, jpx33_name, company_size_category,
company_count, employee_count,
sales_billion_jpy, value_added_billion_jpy, survey_year
```

### corporate_stats_sample.csv（法人企業統計）
```
jpx33_code, jpx33_name, company_size_category,
operating_profit_billion_jpy, net_profit_billion_jpy,
total_assets_billion_jpy, equity_billion_jpy, survey_year
```
`company_size_category` は `大企業` / `中小企業` のみ（個人事業主は対象外）。

### gdp_by_industry_sample.csv（国民経済計算）
```
jpx33_code, jpx33_name,
gdp_contribution_billion_jpy, gdp_share_pct, survey_year
```
業種単位のみ。企業規模別GDPはセンサスの付加価値比率で自動按分されます。

---

## 依存ライブラリ

| パッケージ | バージョン | 用途 |
|-----------|-----------|------|
| streamlit | ≥1.35 | Web UI |
| pandas | ≥2.0 | データ処理 |
| numpy | ≥1.26 | 数値計算 |
| plotly | ≥5.18 | ヒートマップ描画 |
| pyyaml | ≥6.0 | 設定ファイル読込 |
| openpyxl | ≥3.1 | Excel出力（オプション） |
