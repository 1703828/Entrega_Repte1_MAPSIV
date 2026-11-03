"""
Funcions comunes per escriure un dataset YOLO-pose (4 punts: les cantonades TL, TR, BR, BL de la matrícula).
  pose_line()  4 cantonades (píxels) -> '0 cx cy w h  x1 y1 v1 ... x4 y4 v4' normalitzat 0-1
  export()     escriu images/<carpeta>/ i labels/<carpeta>/ (fotos reduïdes a 1280 px pel costat llarg)
  finish()     resum per pantalla, <dataset>_files.csv i un .tar per pujar-lo a la MV
Totes les matrícules anotades de cada foto hi són (també les que no són l'objectiu). Punts visibles (v = 2) si les
cantonades són 'real' o 'approx'; si no, la caixa amb els punts a 0 (v = 0).
"""
import os
import shutil
import tarfile

import cv2
import numpy as np
import pandas as pd

from common.geometry import order_corners

MAX_SIDE = 1280


def pose_line(pts, W, H, visible):
    pts = np.clip(order_corners(pts) / [W, H], 0, 1)
    (x1, y1), (x2, y2) = pts.min(axis=0), pts.max(axis=0)
    kpts = ' '.join(f'{x:.6f} {y:.6f} 2' if visible else '0 0 0' for x, y in pts)
    return f'0 {(x1 + x2) / 2:.6f} {(y1 + y2) / 2:.6f} {x2 - x1:.6f} {y2 - y1:.6f} {kpts}'


def export(images, plates, out_dir, folder_of, line):
    shutil.rmtree(out_dir, ignore_errors=True)
    done = []
    for k, im in enumerate(images.itertuples(), start=1):
        for folder in folder_of(im.split):
            os.makedirs(os.path.join(out_dir, 'images', folder), exist_ok=True)
            os.makedirs(os.path.join(out_dir, 'labels', folder), exist_ok=True)
            image = cv2.imread(im.path)
            H, W = image.shape[:2]
            s = MAX_SIDE / max(W, H)
            if s < 1:
                image = cv2.resize(image, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
            name = f'{im.subset}_{im.image_id}.jpg'
            cv2.imwrite(os.path.join(out_dir, 'images', folder, name), image, [cv2.IMWRITE_JPEG_QUALITY, 90])
            lines = [line(p, W, H) for p in plates[plates.image_id == im.image_id].itertuples()]
            with open(os.path.join(out_dir, 'labels', folder, name[:-4] + '.txt'), 'w') as f:
                f.write(''.join(x + '\n' for x in lines))
            done.append({'image_id': im.image_id, 'subset': im.subset, 'split': im.split, 'folder': folder,
                         'file': name, 'n_plates': len(lines)})
        if k % 50 == 0:
            print(f'{k}/{len(images)} fotos')
    return pd.DataFrame(done)


def finish(out_dir, done):
    done.to_csv(out_dir + '_files.csv', index=False)
    print(done.groupby(['subset', 'folder']).agg(fotos=('file', 'size'), matricules=('n_plates', 'sum')).to_string())
    with tarfile.open(out_dir + '.tar', 'w') as t:
        t.add(out_dir, arcname=os.path.basename(out_dir))
    print(f'Desat {out_dir}/ i {out_dir}.tar')
