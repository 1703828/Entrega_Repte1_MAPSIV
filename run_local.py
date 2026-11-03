"""
run_local.py - Test your submission locally before uploading it to Codabench.

It runs your model exactly like the platform does (same interface, same
normalization, same metrics), so if it works here it will work there.

Usage:
    python run_local.py --submission my_submission/ --data public_val/
    python run_local.py --submission my_submission.zip --data public_val/

Expected data folder:
    public_val/
    ├── images/        (.jpg images)
    └── labels.csv     (columns: filename,plate)

Expected submission (folder or .zip, files at the ROOT):
    model.py           (defines class Model)
    weights/           (your trained weights)
"""

import argparse
import csv
import importlib
import os
import re
import shutil
import sys
import tempfile
import time
import traceback
import zipfile

TIME_LIMIT_SECONDS = 900   # same limit as on Codabench (whole test set)


def norm(s):
    """Same normalization as the official scoring program."""
    return re.sub(r"[^A-Z0-9]", "", str(s).upper())


def fail(msg):
    print(f"\n[ERROR] {msg}")
    sys.exit(1)


def prepare_submission(path):
    """Accept a folder or a .zip and check the structure."""
    if os.path.isfile(path) and path.lower().endswith(".zip"):
        tmp = tempfile.mkdtemp(prefix="submission_")
        with zipfile.ZipFile(path) as z:
            names = z.namelist()
            if "model.py" not in names:
                inner = [n for n in names if n.endswith("model.py")]
                hint = f" Found it at '{inner[0]}' instead." if inner else ""
                fail("model.py must be at the ROOT of the zip, not inside a folder." + hint +
                     "\n        Select the files (not the folder) when creating the zip.")
            z.extractall(tmp)
        return tmp
    if os.path.isdir(path):
        if not os.path.isfile(os.path.join(path, "model.py")):
            fail(f"No model.py found in '{path}'.")
        return os.path.abspath(path)
    fail(f"'{path}' is not a folder or a .zip file.")


def load_labels(data_dir):
    labels_path = os.path.join(data_dir, "labels.csv")
    if not os.path.isfile(labels_path):
        fail(f"No labels.csv found in '{data_dir}'.")
    with open(labels_path, newline="", encoding="utf-8") as f:
        return {r["filename"]: norm(r["plate"]) for r in csv.DictReader(f)}


def main():
    ap = argparse.ArgumentParser(description="Test your submission locally.")
    ap.add_argument("--submission", required=True, help="Submission folder or .zip")
    ap.add_argument("--data", required=True, help="Folder with images/ and labels.csv")
    ap.add_argument("--show-errors", type=int, default=10, help="Wrong predictions to display")
    args = ap.parse_args()

    sub_dir = prepare_submission(args.submission)
    data_dir = os.path.abspath(args.data)
    img_dir = os.path.join(data_dir, "images")
    if not os.path.isdir(img_dir):
        fail(f"No images/ folder found in '{data_dir}'.")
    ref = load_labels(data_dir)
    images = sorted(os.listdir(img_dir))
    print(f"Submission : {sub_dir}")
    print(f"Images     : {len(images)}")

    # Run from a neutral directory, like the platform does.
    # If your code uses relative paths (e.g. open('weights/model.pt')), it will fail
    # here too. Always build paths from weights_dir.
    sys.path.insert(0, sub_dir)
    work_dir = tempfile.mkdtemp(prefix="run_")
    os.chdir(work_dir)

    # 1. Load the model
    try:
        model_module = importlib.import_module("model")
        model = model_module.Model(weights_dir=os.path.join(sub_dir, "weights"))
    except Exception:
        traceback.print_exc()
        fail("Your model could not be loaded (error above). On Codabench this submission would FAIL.")

    # 2. Predict every image
    preds, crashes = {}, 0
    start = time.time()
    for i, name in enumerate(images, 1):
        try:
            p = model.predict(os.path.join(img_dir, name))
            if not isinstance(p, str):
                print(f"[WARNING] predict() returned {type(p).__name__} for {name}, expected str.")
                p = str(p)
        except Exception:
            crashes += 1
            if crashes <= 3:
                print(f"\n[WARNING] predict() crashed on {name}:")
                traceback.print_exc()
            p = ""                       # a crash counts as a wrong prediction
        preds[name] = norm(p)
        if i % 100 == 0:
            print(f"  {i}/{len(images)} images...")
    elapsed = time.time() - start

    # 3. Score (same as the official scoring program)
    plate_ok = sum(preds.get(k, "") == v for k, v in ref.items())
    chars_ok = sum(sum(a == b for a, b in zip(preds.get(k, ""), v)) for k, v in ref.items())
    chars_tot = sum(len(v) for v in ref.values())

    print("\n" + "=" * 45)
    print(f"Plate accuracy : {plate_ok / len(ref):.4f}   ({plate_ok}/{len(ref)})")
    print(f"Char accuracy  : {chars_ok / chars_tot:.4f}")
    print(f"Crashed images : {crashes}")
    print(f"Time           : {elapsed:.1f} s  ({elapsed / max(len(images), 1) * 1000:.0f} ms/image)")
    print("=" * 45)

    wrong = [(k, v, preds.get(k, "")) for k, v in ref.items() if preds.get(k, "") != v]
    if wrong and args.show_errors > 0:
        print(f"\nSome wrong predictions (showing {min(args.show_errors, len(wrong))} of {len(wrong)}):")
        for k, truth, pred in wrong[: args.show_errors]:
            print(f"  {k:<20} truth={truth:<10} pred={pred or '(empty)'}")

    missing = set(ref) - set(images)
    if missing:
        print(f"\n[WARNING] {len(missing)} labels have no matching image (counted as wrong).")
    if elapsed > TIME_LIMIT_SECONDS:
        print(f"\n[WARNING] This took longer than the {TIME_LIMIT_SECONDS} s limit. "
              "On Codabench it would be killed.")
    if crashes:
        print("\n[WARNING] Some images crashed. Fix this before submitting.")

    print("\nIf this ran without errors, your submission should run on Codabench.")
    shutil.rmtree(work_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
