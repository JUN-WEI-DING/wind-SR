import argparse
import sys
import types

import pytest

import windsr.data
from windsr.__main__ import main, parse_model


def test_model_names_two_stores_and_defaults_to_windsr():
    assert parse_model("mine=a/test.zarr,a/train.zarr") == ("mine", "a/test.zarr", "a/train.zarr")
    assert parse_model("test.zarr,train.zarr") == ("windsr", "test.zarr", "train.zarr")
    for bad in ("test.zarr", "mine=test.zarr", "=a,b", "mine=a,b,c"):
        with pytest.raises(argparse.ArgumentTypeError):
            parse_model(bad)


def test_build_data_forwards_every_argument_to_the_build(monkeypatch):
    calls = []
    build = types.SimpleNamespace(cli=calls.append)
    monkeypatch.setitem(sys.modules, "windsr.data.build", build)
    monkeypatch.setattr(windsr.data, "build", build, raising=False)
    main(["build-data", "OUT", "--era5", "/raw/era5", "--help"])
    assert calls == [["OUT", "--era5", "/raw/era5", "--help"]]


def test_train_refuses_a_directory_with_a_training_log(tmp_path):
    pytest.importorskip("torch")
    (tmp_path / "log.jsonl").write_text("{}\n")
    with pytest.raises(SystemExit, match="already has a training log"):
        main(["train", str(tmp_path / "no-data"), str(tmp_path)])


def test_evaluate_needs_a_model(capsys):
    with pytest.raises(SystemExit):
        main(["evaluate", "DATA", "OUT"])
    assert "--model" in capsys.readouterr().err


def test_run_trains_and_predicts_in_out_model_and_evaluates_in_out_evaluation(monkeypatch, tmp_path):
    pytest.importorskip("torch")
    calls = []
    monkeypatch.setattr("windsr.data.store.verify", lambda data: calls.append(("verify", data)) or 1)
    monkeypatch.setattr("windsr.train.train_run", lambda data, run: calls.append(("train", data, run)))
    monkeypatch.setattr("windsr.model.predict.predict_run", lambda data, run: calls.append(("predict", data, run))
                        or {"test": f"{run}/test.zarr", "train": f"{run}/train.zarr"})
    monkeypatch.setattr("windsr.evaluate.report.evaluate", lambda *a, **k: calls.append(("evaluate", *a, k)))
    d, o = tmp_path / "DATA", tmp_path / "OUT"
    main(["run", str(d), str(o)])
    m = o / "model"
    assert calls == [("verify", d), ("train", d, m), ("predict", d, m),
                     ("evaluate", d, o / "evaluation", "windsr", f"{m}/test.zarr", f"{m}/train.zarr", {})]
