"""
Merge all YOLOv11 datasets into one unified dataset.

Final class mapping (3 classes):
    0: Cigarette
    1: smoking              (merged from Smoke, Vape, smoking)
    2: cigarette_like_object (pens, pencils, straws, markers, etc.)

Person class is dropped — not relevant to detection goal.

Sources:
    smoking.yolov11           -> Cigarette(0)->0, Person(1)->DROP, Smoke(2)->1, Vape(3)->1, smoking(4)->1
    straw.v1i.yolov11         -> all class IDs -> 2
    pen.v2i.yolov11           -> all class IDs -> 2
    Pen Detection.v1i.yolov11 -> only class 30 (Pen) -> 2; other classes dropped
    cigrette_like_objects     -> polygon labels converted to bbox; class 0 -> 2
"""

import os
import shutil
import random
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE = Path("e:/projects for clients/Rahul's Dessertation project")
RAW = BASE / "data" / "raw_data"
OUT = BASE / "data" / "merged_dataset"

SOURCES = {
    "smoking":      RAW / "smoking.yolov11",
    "straw":        RAW / "straw.v1i.yolov11",
    "pen_v2":       RAW / "pen.v2i.yolov11",
    "pen_v1":       RAW / "Pen Detection.v1i.yolov11",
    "cig_like":     RAW / "cigrette_like_objects",
}

SPLITS = ["train", "valid", "test"]
RANDOM_SEED = 42


# ── Helpers ────────────────────────────────────────────────────────────────────

def polygon_to_bbox(coords: list[float]) -> tuple[float, float, float, float]:
    """Convert flat polygon coordinate list [x1,y1,x2,y2,...] to (cx,cy,w,h)."""
    xs = coords[0::2]
    ys = coords[1::2]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    cx = (x_min + x_max) / 2
    cy = (y_min + y_max) / 2
    w  = x_max - x_min
    h  = y_max - y_min
    return cx, cy, w, h


def parse_label_file(path: Path, class_remap: dict | None, pen_v1_mode=False) -> list[str]:
    """
    Read a YOLO label file and return remapped bounding-box lines.

    class_remap: {old_class_id: new_class_id}  — classes not in dict are dropped.
    pen_v1_mode: True  -> drop all classes except 30 (mapped to 5).
    """
    lines_out = []
    if not path.exists():
        return lines_out

    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            class_id = int(parts[0])
            coords = list(map(float, parts[1:]))

            # Class filtering / remapping
            if class_remap is not None:
                if class_id not in class_remap:
                    continue
                new_class = class_remap[class_id]
            else:
                new_class = class_id

            # Polygon → bbox conversion
            if len(coords) > 4:
                cx, cy, w, h = polygon_to_bbox(coords)
            else:
                cx, cy, w, h = coords

            lines_out.append(f"{new_class} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")

    return lines_out


def copy_pair(img_path: Path, lbl_path: Path, dest_img_dir: Path,
              dest_lbl_dir: Path, class_remap: dict | None,
              prefix: str = "", pen_v1_mode=False) -> bool:
    """
    Copy an image and its remapped label to the destination directories.
    Returns False if the label has no valid lines (image skipped).
    """
    lines = parse_label_file(lbl_path, class_remap, pen_v1_mode)

    # For pen_v1 we skip images that have zero pen annotations
    if pen_v1_mode and len(lines) == 0:
        return False

    # Unique filename to avoid collisions across datasets
    stem = prefix + img_path.stem
    ext  = img_path.suffix

    dest_img = dest_img_dir / (stem + ext)
    dest_lbl = dest_lbl_dir / (stem + ".txt")

    shutil.copy2(img_path, dest_img)
    with open(dest_lbl, "w") as f:
        f.write("\n".join(lines))

    return True


def make_dirs():
    for split in SPLITS:
        (OUT / split / "images").mkdir(parents=True, exist_ok=True)
        (OUT / split / "labels").mkdir(parents=True, exist_ok=True)


def process_split(src_img_dir: Path, src_lbl_dir: Path, split: str,
                  class_remap: dict | None, prefix: str,
                  pen_v1_mode=False) -> int:
    """Process one split of a source dataset."""
    if not src_img_dir.exists():
        return 0

    dest_img = OUT / split / "images"
    dest_lbl = OUT / split / "labels"
    count = 0

    for img in src_img_dir.iterdir():
        if img.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
            continue
        lbl = src_lbl_dir / (img.stem + ".txt")
        ok = copy_pair(img, lbl, dest_img, dest_lbl, class_remap,
                       prefix=prefix, pen_v1_mode=pen_v1_mode)
        if ok:
            count += 1
    return count


def process_split_with_autosplit(src_img_dir: Path, src_lbl_dir: Path,
                                  class_remap: dict | None, prefix: str,
                                  ratios=(0.8, 0.1, 0.1),
                                  pen_v1_mode=False) -> dict:
    """
    For datasets that only have a train folder — auto-split into train/valid/test.
    Returns counts per split.
    """
    if not src_img_dir.exists():
        return {"train": 0, "valid": 0, "test": 0}

    imgs = [p for p in src_img_dir.iterdir()
            if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}]

    random.seed(RANDOM_SEED)
    random.shuffle(imgs)

    n = len(imgs)
    n_train = int(n * ratios[0])
    n_valid = int(n * ratios[1])

    splits_assigned = (
        [("train", i) for i in imgs[:n_train]] +
        [("valid", i) for i in imgs[n_train:n_train + n_valid]] +
        [("test",  i) for i in imgs[n_train + n_valid:]]
    )

    counts = {"train": 0, "valid": 0, "test": 0}
    for split, img in splits_assigned:
        lbl = src_lbl_dir / (img.stem + ".txt")
        dest_img = OUT / split / "images"
        dest_lbl = OUT / split / "labels"
        ok = copy_pair(img, lbl, dest_img, dest_lbl, class_remap,
                       prefix=prefix, pen_v1_mode=pen_v1_mode)
        if ok:
            counts[split] += 1

    return counts


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    print("Creating output directories...")
    make_dirs()
    total = {"train": 0, "valid": 0, "test": 0}

    # 1. smoking.yolov11 — only train split exists, auto-split 80/10/10
    print("\n[1/5] Processing smoking.yolov11 ...")
    src = SOURCES["smoking"]
    # 0=Cigarette->0, 1=Person->DROP, 2=Smoke->1, 3=Vape->1, 4=smoking->1
    remap = {0: 0, 2: 1, 3: 1, 4: 1}
    counts = process_split_with_autosplit(
        src / "train" / "images",
        src / "train" / "labels",
        class_remap=remap,
        prefix="smk_",
    )
    for s, c in counts.items():
        print(f"    {s}: {c} images")
        total[s] += c

    # 2. straw.v1i.yolov11 — all classes -> 2
    print("\n[2/5] Processing straw.v1i.yolov11 ...")
    src = SOURCES["straw"]
    remap = {0: 2, 1: 2, 2: 2}
    for split in SPLITS:
        c = process_split(
            src / split / "images",
            src / split / "labels",
            split, remap, prefix="str_"
        )
        print(f"    {split}: {c} images")
        total[split] += c

    # 3. pen.v2i.yolov11 — all classes -> 2 (bon, marker, pen, pencil)
    print("\n[3/5] Processing pen.v2i.yolov11 ...")
    src = SOURCES["pen_v2"]
    remap = {0: 2, 1: 2, 2: 2, 3: 2}
    for split in SPLITS:
        c = process_split(
            src / split / "images",
            src / split / "labels",
            split, remap, prefix="pv2_"
        )
        print(f"    {split}: {c} images")
        total[split] += c

    # 4. Pen Detection.v1i.yolov11 — only class 30 (Pen) -> 2; skip images with no pen
    print("\n[4/5] Processing Pen Detection.v1i.yolov11 ...")
    src = SOURCES["pen_v1"]
    remap = {30: 2}
    for split in SPLITS:
        c = process_split(
            src / split / "images",
            src / split / "labels",
            split, remap, prefix="pv1_", pen_v1_mode=True
        )
        print(f"    {split}: {c} images")
        total[split] += c

    # 5. cigrette_like_objects — polygon -> bbox; class 0 -> 2; auto-split 80/10/10
    print("\n[5/5] Processing cigrette_like_objects ...")
    src = SOURCES["cig_like"]
    remap = {0: 2}
    counts = process_split_with_autosplit(
        src / "train" / "images",
        src / "train" / "labels",
        class_remap=remap,
        prefix="clo_",
    )
    for s, c in counts.items():
        print(f"    {s}: {c} images")
        total[s] += c

    # ── Write data.yaml ────────────────────────────────────────────────────────
    yaml_path = OUT / "data.yaml"
    yaml_content = f"""train: {(OUT / 'train' / 'images').as_posix()}
val:   {(OUT / 'valid' / 'images').as_posix()}
test:  {(OUT / 'test'  / 'images').as_posix()}

nc: 3
names:
  - Cigarette
  - smoking
  - cigarette_like_object
"""
    with open(yaml_path, "w") as f:
        f.write(yaml_content)
    print(f"\ndata.yaml written to {yaml_path}")

    # ── Summary ────────────────────────────────────────────────────────────────
    grand_total = sum(total.values())
    print("\n" + "=" * 45)
    print("MERGE COMPLETE")
    print("=" * 45)
    print(f"  train : {total['train']:>5} images")
    print(f"  valid : {total['valid']:>5} images")
    print(f"  test  : {total['test']:>5} images")
    print(f"  TOTAL : {grand_total:>5} images")
    print(f"\nOutput: {OUT}")


if __name__ == "__main__":
    main()
