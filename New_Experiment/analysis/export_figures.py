"""
export_figures.py — Post-processing step run at the end of run_analysis.sh.

Copies all generated figures into two output trees:

    outputs/figures_hq/<rq>/   High-quality PNG  (as generated, lossless)
    outputs/figures_lq/<rq>/   Low-quality JPEG  (96 dpi, quality 80, for document insertion)
"""

from __future__ import annotations

import os
import sys
import shutil
import glob
from pathlib import Path

try:
    from PIL import Image
    HAS_PILLOW = True
except ImportError:
    HAS_PILLOW = False
    print("  [WARN] Pillow not available — LQ JPEG export will be skipped.")
    print("         Install with: pip install Pillow")

_HERE        = os.path.dirname(os.path.abspath(__file__))
_NEW_EXP     = os.path.dirname(_HERE)
PROJECT_ROOT = os.path.dirname(_NEW_EXP)

SRC_ROOT = os.path.join(PROJECT_ROOT, "outputs", "new_experiment")
HQ_ROOT  = os.path.join(PROJECT_ROOT, "outputs", "figures_hq")
LQ_ROOT  = os.path.join(PROJECT_ROOT, "outputs", "figures_lq")

RQ_MAP = {
    "RQ1": "RQ1_baseline_heuristic",
    "RQ2": "RQ2_implicit_framing",
    "RQ3": "RQ3_explicit_symmetry",
    "RQ4": "RQ4_architecture_equity",
    "robustness": "robustness",
}

LQ_DPI     = 96
LQ_QUALITY = 80


def export() -> None:
    if not os.path.isdir(SRC_ROOT):
        print(f"[ERROR] Source directory not found: {SRC_ROOT}")
        sys.exit(1)

    total_hq = 0
    total_lq = 0

    for src_subfolder, dest_name in RQ_MAP.items():
        src_dir = os.path.join(SRC_ROOT, src_subfolder)
        if not os.path.isdir(src_dir):
            print(f"  [SKIP] {src_subfolder} — no output directory found")
            continue

        pngs = sorted(glob.glob(os.path.join(src_dir, "fig_*.png")))
        if not pngs:
            print(f"  [SKIP] {src_subfolder} — no fig_*.png files")
            continue

        hq_dir = os.path.join(HQ_ROOT, dest_name)
        lq_dir = os.path.join(LQ_ROOT, dest_name)
        os.makedirs(hq_dir, exist_ok=True)
        os.makedirs(lq_dir, exist_ok=True)

        print(f"\n  {src_subfolder}  ({len(pngs)} figures)")

        for png_path in pngs:
            fname = os.path.basename(png_path)
            stem  = Path(fname).stem

            hq_dest = os.path.join(hq_dir, fname)
            shutil.copy2(png_path, hq_dest)
            total_hq += 1
            print(f"    HQ  {dest_name}/{fname}")

            if HAS_PILLOW:
                lq_dest = os.path.join(lq_dir, f"{stem}.jpg")
                try:
                    with Image.open(png_path) as img:
                        if img.mode in ("RGBA", "LA", "P"):
                            bg = Image.new("RGB", img.size, (255, 255, 255))
                            if img.mode == "P":
                                img = img.convert("RGBA")
                            bg.paste(img, mask=img.split()[-1] if img.mode == "RGBA" else None)
                            img = bg
                        elif img.mode != "RGB":
                            img = img.convert("RGB")
                        img.save(lq_dest, format="JPEG", quality=LQ_QUALITY,
                                 dpi=(LQ_DPI, LQ_DPI), optimize=True)
                    total_lq += 1
                    print(f"    LQ  {dest_name}/{stem}.jpg")
                except Exception as e:
                    print(f"    [WARN] LQ conversion failed for {fname}: {e}")

    print(f"\n{'─' * 50}")
    print(f"  HQ PNG  exported : {total_hq}  →  {HQ_ROOT}")
    if HAS_PILLOW:
        print(f"  LQ JPEG exported : {total_lq}  →  {LQ_ROOT}")
    else:
        print(f"  LQ JPEG skipped  : Pillow not installed")


if __name__ == "__main__":
    print("Figure Export\n" + "=" * 50)
    export()
