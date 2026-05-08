from pathlib import Path

DATASET = Path(__file__).parent / "data" / "merged_dataset"

# Remap labels: 0→0, 1→0, 2→1  (3 classes → 2 classes)
# Old:  0=Cigarette  1=smoking  2=cigarette_like_object
# New:  0=cigarette  1=cigarette_like_object

def remap_label_file(path: Path):
    lines = path.read_text().strip().splitlines()
    new_lines = []
    for line in lines:
        if not line.strip():
            continue
        parts = line.split()
        cls = int(parts[0])
        if cls == 0 or cls == 1:
            new_cls = 0
        elif cls == 2:
            new_cls = 1
        else:
            continue
        new_lines.append(f"{new_cls} " + " ".join(parts[1:]))
    path.write_text("\n".join(new_lines))

total = 0
for split in ["train", "valid", "test"]:
    label_dir = DATASET / split / "labels"
    if not label_dir.exists():
        continue
    files = list(label_dir.glob("*.txt"))
    for f in files:
        remap_label_file(f)
    total += len(files)
    print(f"  {split}: {len(files)} label files remapped")

print(f"\nDone — {total} files remapped")

# Update data.yaml to 2 classes
yaml_content = f"""train: {DATASET}/train/images
val:   {DATASET}/valid/images
test:  {DATASET}/test/images

nc: 2
names:
  - cigarette
  - cigarette_like_object
"""

yaml_path = DATASET / "data.yaml"
yaml_path.write_text(yaml_content)
print(f"data.yaml updated:\n{yaml_content}")

# Verify: count class distribution in train labels
from collections import Counter

counts = Counter()
for f in (DATASET / "train" / "labels").glob("*.txt"):
    for line in f.read_text().splitlines():
        if line.strip():
            counts[int(line.split()[0])] += 1

print("Train label distribution:")
names = {0: "cigarette", 1: "cigarette_like_object"}
for cls, cnt in sorted(counts.items()):
    print(f"  {names.get(cls, cls)}: {cnt:,} annotations")
