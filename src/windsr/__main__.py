"""windsr: build or verify the data directory, train the network, predict, and evaluate.

  windsr build-data OUT [raw source options]   build DATA from the raw sources (see `windsr build-data --help`)
  windsr verify-data DATA                      check every file against DATA/SHA256SUMS
  windsr train DATA RUN                        train 50 epochs into a new directory RUN
  windsr predict DATA RUN                      write RUN/test.zarr and RUN/train.zarr from RUN/epoch050.pt
  windsr evaluate DATA OUT --model [NAME=]TEST_STORE,TRAIN_STORE
                                               write every facet of the evaluation to OUT
  windsr run DATA OUT                          verify-data, train and predict in OUT/model, evaluate in OUT/evaluation
"""
import argparse
import sys
from pathlib import Path

DEFAULT_MODEL = "windsr"


def parse_model(item: str) -> tuple[str, str, str]:
    """[NAME=]TEST_STORE,TRAIN_STORE -> (name, test store, train store)."""
    name, paths = item.split("=", 1) if "=" in item else (DEFAULT_MODEL, item)
    stores = paths.split(",")
    if not name or len(stores) != 2 or not all(stores):
        raise argparse.ArgumentTypeError(f"needs [NAME=]TEST_STORE,TRAIN_STORE, got {item!r}")
    return name, *stores


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="windsr", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("build-data", help="build DATA from the raw sources")
    s = sub.add_parser("verify-data", help="check every file against DATA/SHA256SUMS")
    s.add_argument("data", type=Path)
    for cmd, text in (("train", "train 50 epochs into a new directory RUN"),
                      ("predict", "write RUN/test.zarr and RUN/train.zarr from RUN/epoch050.pt")):
        s = sub.add_parser(cmd, help=text)
        s.add_argument("data", type=Path)
        s.add_argument("run", type=Path)
    s = sub.add_parser("evaluate", help="write every facet of the evaluation to OUT")
    s.add_argument("data", type=Path)
    s.add_argument("out", type=Path)
    s.add_argument("--model", required=True, type=parse_model, metavar="[NAME=]TEST_STORE,TRAIN_STORE",
                   help=f"prediction stores of the test year and the training period; NAME defaults to {DEFAULT_MODEL}")
    s = sub.add_parser("run", help="verify-data, train and predict in OUT/model, evaluate in OUT/evaluation")
    s.add_argument("data", type=Path)
    s.add_argument("out", type=Path)
    return p


def main(argv: list[str] | None = None) -> None:
    argv = sys.argv[1:] if argv is None else argv
    if argv[:1] == ["build-data"]:
        from .data import build
        return build.cli(argv[1:])
    a = parser().parse_args(argv)
    if a.cmd == "run":
        a.run = a.out / "model"
    if a.cmd in ("train", "predict", "run"):
        try:
            import torch
        except ModuleNotFoundError:
            raise SystemExit(f"windsr {a.cmd} needs torch: uv sync --frozen --extra train")
        torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = True
    if a.cmd in ("verify-data", "run"):
        from .data.store import verify
        print(f"{a.data}: {verify(a.data)} files match SHA256SUMS")
    if a.cmd in ("train", "run"):
        from .train import train_run
        train_run(a.data, a.run)
    if a.cmd in ("predict", "run"):
        from .model.predict import predict_run
        stores = predict_run(a.data, a.run)
        print(f"wrote {', '.join(stores.values())}")
    if a.cmd == "evaluate":
        from .evaluate.report import evaluate
        evaluate(a.data, a.out, *a.model)
    if a.cmd == "run":
        from .evaluate.report import evaluate
        evaluate(a.data, a.out / "evaluation", DEFAULT_MODEL, stores["test"], stores["train"])


if __name__ == "__main__":
    sys.exit(main())
