"""Lectura i escriptura de les anotacions definitives (annotations/). Són la font de veritat.

  images()       una fila per imatge (image_id): ruta, subconjunt, vista, protocol, grups, tasques
  plates()       una fila per matrícula (plate_id): text, geometria, qualitat de les cantonades, objectiu
  splits()       image_id -> 'train' / 'validation' / 'test'. L'ÚNICA partició
  annotations()  matrícules amb les dades de la seva foto i el split (el que fan servir les activitats)
  for_task()     imatges permeses per a una tasca, amb el split
  save()         desa un CSV de manera segura (còpia de seguretat a outputs/backups/ + substitució atòmica)
"""
import datetime
import os
import shutil

import numpy as np
import pandas as pd

from common.paths import ANN, OUT

TASKS = ("detection", "segmentation", "recognition")
XY = [f"{c}{i}" for i in range(1, 5) for c in "xy"]


def images():
    return pd.read_csv(ANN / "images.csv", dtype=str, keep_default_na=False)


def plates():
    p = pd.read_csv(ANN / "plates.csv", dtype=str, keep_default_na=False)
    p[XY] = p[XY].astype(float)
    return p


def splits():
    s = pd.read_csv(ANN / "splits.csv", dtype=str, keep_default_na=False)
    return s.set_index("image_id").split


def corners(row):
    """Fila de plates -> 4x2 float32 (TL, TR, BR, BL)."""
    return np.float32([[row[f"x{i}"], row[f"y{i}"]] for i in range(1, 5)])


def annotations(subsets=None, splits_=None):
    """Matrícules + path, vista, protocol, violació i split de la foto."""
    im = images().assign(split=lambda d: d.image_id.map(splits()))
    a = plates().merge(im[["image_id", "path", "original_path", "view", "protocol", "violation", "split", "tasks"]],
                       on="image_id")
    if subsets is not None:
        a = a[a.subset.isin(subsets)]
    if splits_ is not None:
        a = a[a.split.isin(splits_)]
    return a.reset_index(drop=True)


def for_task(task, subsets=None):
    """Imatges permeses per a una tasca (columna 'tasks'), opcionalment filtrades per subconjunt, amb el split."""
    assert task in TASKS, task
    im = images()
    im = im[im.tasks.str.split(";").map(lambda t: task in t)]
    if subsets is not None:
        im = im[im.subset.isin(subsets)]
    return im.assign(split=im.image_id.map(splits()))


def save(df, name, backup=True):
    """Escriu annotations/<name> sense perdre res: primer una còpia del fitxer actual a outputs/backups/
    (si backup) i després un fitxer temporal que substitueix l'original d'una sola vegada (os.replace)."""
    path = ANN / name
    if backup and path.exists():
        os.makedirs(OUT / "backups", exist_ok=True)
        stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        shutil.copy2(path, OUT / "backups" / f"{stamp}_{name}")
    tmp = path.with_suffix(".tmp")
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)


def by_original():
    """Ruta original (p. ex. 'cv/Frontal/0216KZP.jpg', també les còpies) -> image_id. Per relacionar material extern
    amb noms originals (p. ex. els retalls dels professors de data/cv_plates_crops) amb les imatges del dataset."""
    out = {}
    for r in images().itertuples():
        for p in [r.original_path] + [x for x in r.original_copies.split(";") if x]:
            out[p] = r.image_id
    return out
