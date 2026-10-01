"""Reporting a dataset: the card quotes typed identifiers, and `verify` checks them."""

from __future__ import annotations

import json

import pytest

from plantbench import datagen, report
from plantbench.__main__ import main
from plantbench.core.spec import Spec, digest, parse_identifier

SPEC = {
    "study": {"name": "t", "case": "jacketed_cstr", "seed": 1},
    "base": {"structure": "temperature PI"},
    "sample": {"n": 2, "tuning.reactor T.tau_I": [10.0, 30.0]},
    "run": {"t_end": 20.0, "dt": 5.0},
}
TASK = {"target": "t.label", "signals": ["reactor T"], "window": [0.0, 20.0],
        "metrics": ["accuracy"]}


def _quiet(msg):
    pass


def _dataset(path, spec: dict | None = None):
    return datagen.generate(Spec.from_dict(spec or SPEC), path, log=_quiet)


def test_identifiers_are_parsed_by_kind():
    assert parse_identifier("pb-spec:6A6FC673261B") == ("pb-spec", "6a6fc673261b")
    assert parse_identifier("pb-protocol:dab0d47475e9") == ("pb-protocol", "dab0d47475e9")
    assert parse_identifier("6a6fc673261b") == (None, "6a6fc673261b")
    for bad in ("pb-sepc:6a6fc673261b", "pb-spec:6a6fc67", "pb-spec:6a6fc673261z"):
        with pytest.raises(ValueError):
            parse_identifier(bad)
    assert len(digest("x", None)) == 64 and digest("x", None).startswith(digest("x"))


def test_the_card_names_the_specification_and_the_protocol(tmp_path):
    out = _dataset(tmp_path / "d", {**SPEC, "task": TASK})
    spec = Spec.from_dict({**SPEC, "task": TASK})
    c = report.card(out)
    assert c["spec_id"] == f"pb-spec:{spec.digest()}"
    assert c["protocol_id"] == f"pb-protocol:{spec.protocol_digest()}"
    assert c["spec_sha256"] == digest(spec.fingerprint(), None)
    assert c["spec_file"] == str(out / "spec.json") and c["runs"]["ok"] == 2
    assert c["warnings"] == [] or all("uncommitted" in w for w in c["warnings"])
    text = report.render(c)
    assert c["spec_id"] in text and c["protocol_id"] in text and "SHA-256" in text
    latex = report.render(c, "latex")
    assert r"\texttt{jacketed\_cstr}" in latex and latex.count(r"\\") == len(report.rows(c))
    assert json.loads(report.render(c, "json"))["protocol_id"] == c["protocol_id"]
    assert "| Protocol |" in report.render(c, "md")


def test_a_card_without_a_task_has_no_protocol(tmp_path):
    c = report.card(_dataset(tmp_path / "d"))
    assert c["protocol_id"] is None and "pb-protocol" not in report.render(c)


def test_the_card_warns_of_what_weakens_the_report(tmp_path):
    out = _dataset(tmp_path / "d")
    spec = json.loads((out / "spec.json").read_text())
    spec["run"]["t_end"] = 30.0
    (out / "spec.json").write_text(json.dumps(spec))
    assert any("no longer describes" in w for w in report.card(out)["warnings"])
    (out / "manifest.json").unlink()
    c = report.card(out)
    assert c["warnings"] == ["no manifest: how the runs were generated is not recorded"]
    assert c["spec_id"] is None and "not recorded" in report.render(c)


def test_verify_checks_each_identifier_against_its_kind(tmp_path):
    spec = Spec.from_dict({**SPEC, "task": TASK})
    path = tmp_path / "spec.json"
    path.write_text(json.dumps(spec.to_dict()))
    s, p = spec.digest(), spec.protocol_digest()
    ok, lines = report.verify(path, [f"pb-spec:{s}", f"pb-protocol:{p}", p,
                                     digest(spec.fingerprint(), None)[:20]])
    assert ok and all(line.startswith("match") for line in lines)
    ok, lines = report.verify(path, [f"pb-spec:{p}"])
    assert not ok and lines[0].startswith("MISMATCH")


def test_verify_reads_a_dataset_as_well_as_a_file(tmp_path):
    out = _dataset(tmp_path / "d")
    s = Spec.from_dict(SPEC).digest()
    assert report.verify(out, [f"pb-spec:{s}", f"pb-protocol:{s}"])[0]
    with pytest.raises(ValueError):
        report.verify(tmp_path, [f"pb-spec:{s}"])


def test_the_commands(tmp_path, capsys):
    out = _dataset(tmp_path / "d")
    s = Spec.from_dict(SPEC).digest()
    for fmt in report.FORMATS:
        assert main(["card", str(out), "--format", fmt]) == 0
    assert main(["verify", str(out / "spec.json"), f"pb-spec:{s}"]) == 0
    assert main(["verify", str(out / "spec.json"), "pb-spec:000000000000"]) == 1
    assert main(["verify", str(tmp_path / "missing.toml"), f"pb-spec:{s}"]) == 2
    assert f"pb-spec:{s}" in capsys.readouterr().out
