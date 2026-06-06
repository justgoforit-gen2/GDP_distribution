"""Test the JSIC mapping tables in scripts/build_estat_dataset.py."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT.parent / "estat_client"))


def _load_module():
    # avoid argparse side effects: import via importlib
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "build_estat_dataset", ROOT / "scripts" / "build_estat_dataset.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_jsic_names_19():
    m = _load_module()
    assert len(m.JSIC_NAMES) == 19
    assert set(m.JSIC_NAMES.keys()) == set("ABCDEFGHIJKLMNOPQRS")


def test_census_mapping_covers_known_codes():
    m = _load_module()
    # 経済センサスでは A,B は AB として統合され、S 公務は対象外
    assert "AB" in m.CENSUS_CAT01_TO_JSIC
    assert m.CENSUS_CAT01_TO_JSIC["AB"] == ["A", "B"]
    assert "S" not in m.CENSUS_CAT01_TO_JSIC


def test_sna_mapping_covers_real_economy():
    m = _load_module()
    # 製造業 150 → E
    assert m.SNA_CAT01_TO_JSIC["150"] == ["E"]
    # 金融保険 430 → J
    assert m.SNA_CAT01_TO_JSIC["430"] == ["J"]
    # 公務 480 → S
    assert m.SNA_CAT01_TO_JSIC["480"] == ["S"]


def test_corp_mapping_excludes_finance():
    m = _load_module()
    # 法人企業統計 0003060791 は金融保険業以外なので J 業種コードは無い
    j_targets = [v for v in m.CORP_CAT02_TO_JSIC.values() if "J" in v]
    assert j_targets == []
    # 製造業 108 → E
    assert m.CORP_CAT02_TO_JSIC["108"] == ["E"]


def test_size_codes_distinct():
    m = _load_module()
    assert m.CENSUS_LARGE_CODE not in m.CENSUS_SME_CODES
    assert m.CORP_LARGE_CODE not in m.CORP_SME_CODES
