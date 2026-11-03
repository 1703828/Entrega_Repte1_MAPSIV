"""Dibuixa matrícules a fitxers JPG per revisar-les sense finestra (la mateixa imatge que mostra l'eina).

Mateixa selecció que ground_truth_maker/annotate.py: sense filtres, les incidències obertes de prioritat alta.
Ús (des de l'arrel):  python -m ground_truth_maker.check [--kind ...] [--plate ...] [--subset ...] --name revisio
Sortida: outputs/gt_check/<name>/<plate_id>.jpg
"""
import argparse
from pathlib import Path

import cv2

from common import manifest
from ground_truth_maker.annotate import Store, render, select

ap = argparse.ArgumentParser()
ap.add_argument("--priority", nargs="+", choices=["alta", "baixa"])
for a in ("--kind", "--plate", "--image", "--subset", "--split", "--quality"):
    ap.add_argument(a, nargs="+")
ap.add_argument("--name", default="check")
args = ap.parse_args()

store = Store("")
sel = select(args, store)
out = Path("outputs/gt_check") / args.name
out.mkdir(parents=True, exist_ok=True)
split = manifest.splits()
for iid, g in sel.groupby("image_id", sort=False):
    plates = store.plates[store.plates.image_id == iid].reset_index(drop=True)
    plates["view"], plates["split"] = g.view.iloc[0], split[iid]
    img = cv2.imread(g.path.iloc[0])
    for pid in g.plate_id:
        canvas, _ = render(img, plates, int(plates.index[plates.plate_id == pid][0]), [], iid)
        cv2.imwrite(str(out / f"{pid}.jpg"), canvas, [cv2.IMWRITE_JPEG_QUALITY, 85])
print(f"{len(sel)} matrícules -> {out}/")
