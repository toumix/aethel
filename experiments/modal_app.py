"""
The Modal app that trains the supertagger on a GPU, launched from GitHub
Actions (sessions cannot speak gRPC, see ``TODO.md``)::

    modal run experiments/modal_app.py --epochs 5 --test
    modal run experiments/modal_app.py --smoke  # tiny end-to-end check

The æthel dump ships with the repo; ``prepare`` extracts the supertagging
view onto the ``aethel-tagging`` volume once, and ``train`` writes
checkpoints and ``metrics.jsonl`` to ``/vol/runs/<run_name>``.
"""
from __future__ import annotations

import os
import pathlib
import subprocess
import sys
import zipfile

import modal

GPU = os.environ.get("MODAL_GPU", "A100-40GB")
app = modal.App("aethel-supertagging")
volume = modal.Volume.from_name("aethel-tagging", create_if_missing=True)
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch", "transformers")
    .add_local_dir(
        pathlib.Path(__file__).parent.parent, remote_path="/repo"))


def setup() -> None:
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q", "/repo"], check=True)
    sys.path.insert(0, "/repo/experiments")


@app.function(image=image, volumes={"/vol": volume}, timeout=1800)
def prepare(force: bool = False) -> None:
    """Extract the supertagging view of the dump onto the volume."""
    setup()
    out_dir = pathlib.Path("/vol/tagging")
    if out_dir.exists() and not force:
        return
    with zipfile.ZipFile("/repo/data/aethel_1.0.0a5.zip") as archive:
        archive.extractall("/tmp/dump")
    sys.argv = ["data.py", "/tmp/dump/aethel_1.0.0a5.pickle", str(out_dir)]
    import data
    data.main()
    volume.commit()


@app.function(image=image, volumes={"/vol": volume}, gpu=GPU, timeout=21600)
def train(arguments: list[str]) -> str:
    """Run ``train.py`` on the volume's data, returning the metrics log."""
    setup()
    import train as script
    sys.argv = ["train.py", "--data", "/vol/tagging", *arguments]
    script.main()
    volume.commit()
    out = arguments[arguments.index("--out") + 1]
    return (pathlib.Path(out) / "metrics.jsonl").read_text()


@app.local_entrypoint()
def main(epochs: int = 5, encoder: str = "DTAI-KULeuven/robbert-2023-dutch-base",
         batch_size: int = 32, run_name: str = "run", limit: int = 0,
         test: bool = False, smoke: bool = False, force_data: bool = False):
    prepare.remote(force=force_data)
    arguments = [
        "--out", f"/vol/runs/{run_name}", "--encoder", encoder,
        "--epochs", str(epochs), "--batch-size", str(batch_size)]
    if smoke:
        arguments += ["--limit", "200", "--epochs", "1"]
    elif limit:
        arguments += ["--limit", str(limit)]
    if test:
        arguments += ["--test"]
    print(train.remote(arguments))
