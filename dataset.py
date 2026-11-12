"""Reproducible Fruits-360 downloader; training and test source sets stay separate.

Source: https://github.com/fruits-360/fruits-360-100x100
Original dataset authors: Mihai Oltean and collaborators; CC BY-SA 4.0.
"""
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import BytesIO
from pathlib import Path
from urllib.parse import quote
import json
import random
import requests
from PIL import Image

API = "https://api.github.com/repos/fruits-360/fruits-360-100x100/contents/"
SOURCE = "https://github.com/fruits-360/fruits-360-100x100"
SEED = 42
FOLDERS = {
    "apple": ["Apple Red 1", "Apple Golden 1", "Apple Granny Smith 1"],
    "orange": ["Orange 1", "Orange 2"]
}

def list_images(split, folder):
    path = quote(split + "/" + folder, safe="/")
    response = requests.get(API + path, params={"ref": "main"},
                            headers={"Accept": "application/vnd.github+json",
                                     "User-Agent": "fruitlens-course-project"}, timeout=40)
    response.raise_for_status()
    return [item for item in response.json()
            if item["type"] == "file" and item["name"].lower().endswith((".jpg", ".png", ".jpeg"))]

def choose(pool_by_folder, count, seed):
    shuffled = []
    for name, items in pool_by_folder:
        choices = list(items)
        random.Random(str(seed) + name).shuffle(choices)
        shuffled.append(choices)
    selected = []
    for index in range(max(map(len, shuffled))):
        for items in shuffled:
            if index < len(items):
                selected.append(items[index])
                if len(selected) == count:
                    return selected
    raise ValueError("Not enough source images for requested sample")

def download(task):
    source_url, target = task
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_file() and target.stat().st_size > 100:
        return
    response = requests.get(source_url, headers={"User-Agent": "fruitlens-course-project"}, timeout=60)
    response.raise_for_status()
    with Image.open(BytesIO(response.content)) as image:
        image.convert("RGB").save(target, "JPEG", quality=93)

def prepare_dataset(root="data", train_per_class=90, val_per_class=24, test_per_class=16):
    if train_per_class * 2 < 50:
        raise ValueError("Professor requires at least 50 training images")
    if val_per_class < 0.2 * train_per_class:
        raise ValueError("Validation count must be at least 20% of training count")
    if test_per_class < 2:
        raise ValueError("Four separate test photos are required")
    root = Path(root)
    manifest_path = root / "dataset_manifest.json"
    expected = {"train": 2 * train_per_class, "val": 2 * val_per_class,
                "test": 2 * test_per_class}
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("counts") == expected and all(
            len(list((root / split / fruit).glob("*.jpg"))) == expected[split] // 2
            for split in expected for fruit in FOLDERS
        ):
            return manifest
        raise RuntimeError("Dataset folder has mismatched counts. Remove data/ and retry.")

    tasks = []
    sources = {}
    for fruit, folders in FOLDERS.items():
        sources[fruit] = list(folders)
        pools = {}
        for partition in ("Training", "Test"):
            pools[partition] = [(folder, list_images(partition, folder)) for folder in folders]
        chosen_train = choose(pools["Training"], train_per_class + val_per_class, SEED)
        chosen_test = choose(pools["Test"], test_per_class, SEED + 100)
        splits = {"train": chosen_train[:train_per_class],
                  "val": chosen_train[train_per_class:],
                  "test": chosen_test}
        for split, selected in splits.items():
            for index, entry in enumerate(selected):
                target = root / split / fruit / ("%04d.jpg" % index)
                tasks.append((entry["download_url"], target))
    print("Downloading %d Fruits-360 images..." % len(tasks))
    with ThreadPoolExecutor(max_workers=8) as executor:
        for idx, future in enumerate(as_completed([executor.submit(download, task) for task in tasks]), 1):
            future.result()
            if idx % 50 == 0 or idx == len(tasks):
                print("Verified %d/%d" % (idx, len(tasks)))
    manifest = {
        "source": SOURCE, "license": "CC BY-SA 4.0",
        "seed": SEED, "counts": expected,
        "class_folders": sources, "classes": {"0": "Apple", "1": "Orange"}
    }
    root.mkdir(exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest

if __name__ == "__main__":
    prepare_dataset()
