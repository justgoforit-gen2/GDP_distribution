"""Pull e-Stat StatsData (JSON) and save to data/raw/.

This script is intentionally minimal and dependency-free (stdlib only).

Usage (PowerShell):
  $env:ESTAT_APP_ID = "<your_app_id>"
  uv run python scripts/pull_estat_statsdata.py --statsDataId 0004006320

It will write:
  - data/raw/estat_<statsDataId>.json
  - data/processed/estat_<statsDataId>.csv   (long-form; if extract succeeds)

Notes:
- Do NOT hardcode appId in the script; use env var `ESTAT_APP_ID`.
- The response structure varies slightly by endpoint/version; this script extracts
  the common `GET_STATS_DATA.STATISTICAL_DATA.DATA_INF.VALUE` shape.
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlencode
from urllib.request import Request, urlopen


DEFAULT_ENDPOINT = "https://api.e-stat.go.jp/rest/3.0/app/json/getStatsData"


def _http_get_json(url: str, timeout: int = 60) -> dict[str, Any]:
    req = Request(url, headers={"User-Agent": "GDP_distribution/estat-puller"})
    with urlopen(req, timeout=timeout) as resp:
        raw = resp.read().decode("utf-8")
    return json.loads(raw)


def _safe_get(d: dict[str, Any], path: list[str]) -> Any:
    cur: Any = d
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return None
        cur = cur[key]
    return cur


def _extract_values_payload(payload: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return (values, class_obj_list).

    values: list of dicts like {"@cat01":..., "@time":..., "$":...}
    class_obj_list: list of dicts describing dimension code->label mappings
    """
    values = _safe_get(payload, ["GET_STATS_DATA", "STATISTICAL_DATA", "DATA_INF", "VALUE"]) or []
    if isinstance(values, dict):
        values = [values]

    class_obj = _safe_get(payload, ["GET_STATS_DATA", "STATISTICAL_DATA", "CLASS_INF", "CLASS_OBJ"]) or []
    if isinstance(class_obj, dict):
        class_obj = [class_obj]

    return values, class_obj


def _build_code_to_label_maps(class_obj_list: list[dict[str, Any]]) -> dict[str, dict[str, str]]:
    """Return mapping: dimension_key -> {code -> label}.

    dimension_key is e.g. "cat01", "time", "area" (without leading '@').
    """
    dim_maps: dict[str, dict[str, str]] = {}

    for obj in class_obj_list:
        # '@id' contains the attribute name used in VALUE entries (e.g. '@cat01')
        obj_id = obj.get("@id")
        if not isinstance(obj_id, str) or not obj_id.startswith("@"):
            continue
        dim_key = obj_id[1:]

        class_list = obj.get("CLASS")
        if class_list is None:
            continue
        if isinstance(class_list, dict):
            class_list = [class_list]
        if not isinstance(class_list, list):
            continue

        m: dict[str, str] = {}
        for c in class_list:
            if not isinstance(c, dict):
                continue
            code = c.get("@code")
            name = c.get("@name")
            if isinstance(code, str) and isinstance(name, str):
                m[code] = name
        if m:
            dim_maps[dim_key] = m

    return dim_maps


def _values_to_long_rows(values: list[dict[str, Any]], dim_maps: dict[str, dict[str, str]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    for v in values:
        if not isinstance(v, dict):
            continue
        row: dict[str, Any] = {}

        # Dimensions appear as keys like '@cat01', '@time', ...
        for k, val in v.items():
            if not isinstance(k, str):
                continue
            if k.startswith("@"):
                dim = k[1:]
                row[dim] = val
                if dim in dim_maps and isinstance(val, str):
                    row[f"{dim}_name"] = dim_maps[dim].get(val)

        # Actual numeric value typically stored in '$'
        cell = v.get("$")
        row["value"] = cell
        rows.append(row)

    return rows


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--statsDataId", required=True, help="e-Stat statsDataId (e.g. 0004006320)")
    p.add_argument("--endpoint", default=DEFAULT_ENDPOINT, help="API endpoint URL")
    p.add_argument("--timeout", type=int, default=60)
    p.add_argument("--metaGetFlg", default="Y")
    p.add_argument("--cntGetFlg", default="N")
    p.add_argument("--explanationGetFlg", default="Y")
    p.add_argument("--annotationGetFlg", default="Y")
    p.add_argument("--sectionHeaderFlg", default="1")
    p.add_argument("--replaceSpChars", default="0")

    args = p.parse_args()

    app_id = os.environ.get("ESTAT_APP_ID") or ""
    if not app_id.strip():
        raise SystemExit("ESTAT_APP_ID が未設定です。PowerShellで $env:ESTAT_APP_ID = '<appId>' を設定してください。")

    params = {
        "appId": app_id,
        "lang": "J",
        "statsDataId": args.statsDataId,
        "metaGetFlg": args.metaGetFlg,
        "cntGetFlg": args.cntGetFlg,
        "explanationGetFlg": args.explanationGetFlg,
        "annotationGetFlg": args.annotationGetFlg,
        "sectionHeaderFlg": args.sectionHeaderFlg,
        "replaceSpChars": args.replaceSpChars,
    }

    url = f"{args.endpoint}?{urlencode(params)}"

    payload = _http_get_json(url, timeout=args.timeout)

    project_root = Path(__file__).resolve().parents[1]
    raw_dir = project_root / "data" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    processed_dir = project_root / "data" / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    raw_path = raw_dir / f"estat_{args.statsDataId}_{ts}.json"
    raw_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[OK] saved raw json: {raw_path}")

    # Best-effort CSV extraction (long-form)
    try:
        values, class_obj = _extract_values_payload(payload)
        dim_maps = _build_code_to_label_maps(class_obj)
        rows = _values_to_long_rows(values, dim_maps)
        if not rows:
            print("[WARN] No VALUE rows found in payload; CSV not generated.")
            return 0

        # Avoid adding new dependencies; write simple CSV ourselves.
        # Determine columns union
        cols: list[str] = []
        col_set: set[str] = set()
        for r in rows:
            for k in r.keys():
                if k not in col_set:
                    col_set.add(k)
                    cols.append(k)

        out_csv = processed_dir / f"estat_{args.statsDataId}_{ts}.csv"
        import csv

        with out_csv.open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            w.writerows(rows)

        print(f"[OK] saved extracted csv: {out_csv}")
    except Exception as e:
        print(f"[WARN] CSV extract failed: {type(e).__name__}: {e}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
