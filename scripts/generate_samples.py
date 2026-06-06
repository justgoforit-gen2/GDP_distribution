"""DEPRECATED. オフライン用フォールバック。出力は旧 JPX33 × 3規模 で現行アプリは読めない。
通常は scripts/build_estat_dataset.py を使うこと。

One-shot script to generate sample CSV files in data/samples/.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from modules.sample_generator import (
    generate_census_csv,
    generate_corporate_stats_csv,
    generate_gdp_csv,
)

OUTPUT_DIR = Path(__file__).resolve().parents[1] / "data" / "samples"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

generate_census_csv(OUTPUT_DIR / "census_sample.csv")
generate_corporate_stats_csv(OUTPUT_DIR / "corporate_stats_sample.csv")
generate_gdp_csv(OUTPUT_DIR / "gdp_by_industry_sample.csv")

print("[DONE] All sample files generated.")
