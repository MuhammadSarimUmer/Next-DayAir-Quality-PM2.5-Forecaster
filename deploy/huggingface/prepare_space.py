"""Copy the minimum files needed for a Hugging Face Streamlit Space.

Usage:
    python deploy/huggingface/prepare_space.py ../pm25-space
    cd ../pm25-space
    git init
    git remote add origin https://huggingface.co/spaces/<user>/<space>
    git add .
    git commit -m "Deploy PM2.5 forecaster"
    git push

Re-running against an existing bundle cleans stale files before copying, but
keeps Git metadata (`.git`, `.gitattributes`, `.gitignore`) so an existing Space
remote and history are not destroyed. A non-empty directory that does not look
like a previous Space bundle is refused rather than deleted.
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEPLOY = Path(__file__).resolve().parent
IGNORE = shutil.ignore_patterns("__pycache__", "*.pyc", ".ipynb_checkpoints")
PRESERVE = {".git", ".gitattributes", ".gitignore"}


def copy_tree(src: Path, dst: Path) -> None:
    if not src.exists():
        raise FileNotFoundError(src)
    shutil.copytree(src, dst, ignore=IGNORE, dirs_exist_ok=True)


def prepare_target(target: Path) -> None:
    """Create the target or safely clean a previous Space bundle in place."""
    target = target.resolve()
    root = ROOT.resolve()

    if target == root or root in target.parents or target in root.parents:
        raise SystemExit(
            "Refusing to use the project root, a parent, or a subdirectory of it."
        )
    if target == Path.home().resolve() or target == Path(target.anchor):
        raise SystemExit("Refusing to use the home or filesystem root directory.")

    if not target.exists():
        target.mkdir(parents=True)
        return
    if not target.is_dir():
        raise SystemExit(f"Target is not a directory: {target}")

    entries = [item for item in target.iterdir() if item.name not in PRESERVE]
    if not entries:
        return

    readme = target / "README.md"
    is_previous_bundle = (
        readme.is_file()
        and "sdk: streamlit" in readme.read_text(encoding="utf-8", errors="ignore")
    )
    if not is_previous_bundle:
        raise SystemExit(
            "Refusing to clean a non-empty directory that is not a previous Space "
            f"bundle: {target}\nDelete it manually or choose an empty target."
        )

    for item in entries:
        if item.is_dir() and not item.is_symlink():
            shutil.rmtree(item)
        else:
            item.unlink()
    print(f"Cleaned previous bundle in: {target}")


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python deploy/huggingface/prepare_space.py <target-dir>")
    target = Path(sys.argv[1]).resolve()
    prepare_target(target)

    shutil.copy2(DEPLOY / "README.md", target / "README.md")
    shutil.copy2(DEPLOY / "requirements.txt", target / "requirements.txt")

    copy_tree(ROOT / "app", target / "app")
    copy_tree(ROOT / "src", target / "src")

    model_src = ROOT / "models" / "final_pipeline.joblib"
    if not model_src.exists():
        raise SystemExit("Missing models/final_pipeline.joblib. Run `python -m src.train_final`.")
    (target / "models").mkdir(exist_ok=True)
    shutil.copy2(model_src, target / "models" / "final_pipeline.joblib")
    meta = ROOT / "models" / "model_metadata.json"
    if meta.exists():
        shutil.copy2(meta, target / "models" / "model_metadata.json")

    fallback = ROOT / "data" / "processed" / "live_fallback.csv"
    if not fallback.exists():
        raise SystemExit(
            "Missing data/processed/live_fallback.csv. Run "
            "`python -m src.live_data --refresh-fallback --offline`."
        )
    (target / "data" / "processed").mkdir(parents=True, exist_ok=True)
    shutil.copy2(fallback, target / "data" / "processed" / "live_fallback.csv")

    history = ROOT / "data" / "processed" / "features.csv"
    if not history.exists():
        raise SystemExit(
            "Missing data/processed/features.csv (used by the app's history chart). "
            "Run `python -m src.features`."
        )
    shutil.copy2(history, target / "data" / "processed" / "features.csv")

    figures = ROOT / "reports" / "figures"
    if not figures.exists():
        raise SystemExit(
            "Missing reports/figures (used by the app's accuracy panel). "
            "Run `python -m src.error_analysis`."
        )
    copy_tree(figures, target / "reports" / "figures")

    for doc in ["DATA_SOURCES.md", "Limitations.md", "LICENSE"]:
        if (ROOT / doc).exists():
            shutil.copy2(ROOT / doc, target / doc)

    print(f"Space prepared at: {target}")
    print("Included chart assets: data/processed/features.csv and reports/figures/*.png")
    print("Next: git init, add the Hugging Face remote, commit, and push.")


if __name__ == "__main__":
    main()
