"""
Description:
    Cleans and standardizes a 5-class satellite image dataset stored as one folder
    per class. Removes simple filename-based duplicates, normalizes orientation and
    color mode, resizes images to 224×224, and writes cleaned JPEGs to per-class
    folders for downstream modeling.

Inputs:
    - DATA/weather_images/
        Root directory containing one subfolder per weather class (e.g., hurricane,
        wildfires, duststorm, rollconvection, cellconvection). Subfolders may contain
        .jpg/.jpeg/.png/.bmp/.tif/.tiff images.

Process:
    1. Validates the input path and discovers class folders (subdirectories).
    2. Removes *source* duplicates by filename rules:
         • delete any file whose name contains '(' or ')' or "Copy"
    3. For each remaining image:
         • open with PIL, fix EXIF orientation, convert to RGB
         • resize to 224×224
         • save as JPEG to DATA/cleaned_data/{class_name}/ using original basename
           (on name collision, append a short hash)
    4. Prints a per-class and overall summary of processed vs. skipped (unreadable) files.

Outputs:
    - DATA/cleaned_data/{class_name}/*.jpg
        Cleaned, standardized 224×224 RGB JPEG images per class.
    - Printed summary (counts per class; total processed and skipped).
"""

from pathlib import Path
from PIL import Image, ImageOps
import hashlib
import sys

# ========= PATH CONFIG (repo-relative) =========
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent          # repo_root/
DATA_ROOT = REPO_ROOT / "DATA"

# Raw dataset root (folder containing 5 subfolders, one per class)
INPUT_ROOT = DATA_ROOT / "weather_images"

# Where to write the cleaned images
OUTPUT_ROOT = DATA_ROOT / "cleaned_data"

# Dry run: validate and count, but DO NOT write outputs or delete source dupes
DRY_RUN = False
# =============================================

# Validate input path
if not INPUT_ROOT.exists() or not INPUT_ROOT.is_dir():
    print(f"Input path not found or not a directory: {INPUT_ROOT}", file=sys.stderr)
    raise SystemExit(1)

# Prepare output path
if not DRY_RUN:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
TARGET_SIZE = (224, 224)

# 1) Find class folders (each immediate subfolder is treated as a class)
class_dirs = sorted([d for d in INPUT_ROOT.iterdir() if d.is_dir()])
if not class_dirs:
    print(f"No class subfolders found under: {INPUT_ROOT}", file=sys.stderr)
    raise SystemExit(1)

# 2) Delete source duplicates by simple filename rules (BEFORE processing)
#    Rule: delete any file whose name contains '(' or ')' OR whose stem contains '- Copy'
all_files = [p for p in INPUT_ROOT.rglob("*") if p.is_file()]
dupes = [p for p in all_files if "(" in p.name or ")" in p.name or "- Copy" in p.stem]

dupes_removed = 0
if DRY_RUN:
    dupes_removed = len(dupes)  # simulate deletion
else:
    for p in dupes:
        try:
            p.unlink(missing_ok=True)
            dupes_removed += 1
        except Exception:
            # best-effort only; ignore OS/permissions errors
            pass

print(f"Duplicates removed by name rule: {dupes_removed}")

# 3) Preprocess per class: open → EXIF fix → RGB → resize → save JPEG to cleaned_data/{class}
clean_counts = {}
skip_counts = {}
total_processed = 0
total_skipped = 0

for cdir in class_dirs:
    class_name = cdir.name
    out_dir = OUTPUT_ROOT / class_name
    if not DRY_RUN:
        out_dir.mkdir(parents=True, exist_ok=True)

    clean_counts[class_name] = 0
    skip_counts[class_name] = 0

    # Gather supported images (shallow or nested)
    files = [p for p in cdir.rglob("*") if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS]

    for src in files:
        try:
            with Image.open(src) as im:
                if DRY_RUN:
                    # In dry-run, we just verify the file opens successfully.
                    pass
                else:
                    # Normalize: fix EXIF orientation, convert to RGB, resize to 224x224
                    im = ImageOps.exif_transpose(im).convert("RGB")
                    im = im.resize(TARGET_SIZE, resample=Image.BILINEAR)

                    # Save using original basename; if collision, append a short hash
                    base = src.stem
                    out_path = out_dir / f"{base}.jpg"
                    if out_path.exists():
                        h = hashlib.sha1(str(src).encode("utf-8")).hexdigest()[:8]
                        out_path = out_dir / f"{base}_{h}.jpg"

                    im.save(out_path, format="JPEG", quality=95)

        except Exception:
            # unreadable/corrupt/etc. → skip and count
            skip_counts[class_name] += 1
            total_skipped += 1
            continue

        clean_counts[class_name] += 1
        total_processed += 1

# 4) Summary
print("\n=== Preprocessing Summary ===")
print(f"Input root : {INPUT_ROOT}")
print(f"Output root: {OUTPUT_ROOT}")
print(f"Total processed: {total_processed}")
print(f"Total skipped  : {total_skipped}")
print("Per-class counts:")
for cname in sorted(set(list(clean_counts.keys()) + list(skip_counts.keys()))):
    print(f"  {cname:20s} -> processed: {clean_counts.get(cname,0):5d} | skipped: {skip_counts.get(cname,0):5d}")
