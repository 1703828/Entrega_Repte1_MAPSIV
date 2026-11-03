"""
Caràcters d'uc3m per al reconeixedor: cada matrícula d'una fila de les fotos d'uc3m del train es redreça amb el
polígon anotat i es segmenta com a model.py (opcions 0 i 2); cada regió rep l'etiqueta del caràcter anotat d'uc3m
amb què se solapa més (IoU >= 0,5). Resultat: 5.158 caràcters.
Sortida: recognizer/cache/uc3m_chars.pkl (llista de (màscara, caràcter, matrícula))
Executar des de l'arrel: python -B -m recognizer.chars_uc3m
"""
import csv
import json
import pickle
from collections import Counter

import cv2
import numpy as np

import model as mf
split = {r['image_id']: r['split'] for r in csv.DictReader(open('annotations/splits.csv'))}
images = [r for r in csv.DictReader(open('annotations/images.csv'))
          if r['subset'] == 'uc3m' and split.get(r['image_id']) == 'train']
m = mf.MARGIN
DST = np.float32([[m, m], [m + mf.PLATE_W - 1, m], [m + mf.PLATE_W - 1, m + mf.PLATE_H - 1], [m, m + mf.PLATE_H - 1]])


def order(q):
    s, d = q.sum(1), np.diff(q, axis=1).ravel()
    return np.float32([q[np.argmin(s)], q[np.argmin(d)], q[np.argmax(s)], q[np.argmax(d)]])


def iou(a, b):
    ix = max(0, min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1]))
    inter = ix * iy
    return inter / (a[2] * a[3] + b[2] * b[3] - inter) if inter else 0.0


out, stats = [], Counter()
for k, im in enumerate(images, 1):
    d = json.load(open(im['path'][:-4] + '.json'))
    image = None
    for i, lp in enumerate(d['lps']):
        poly = np.float32(lp['poly_coord'])
        if len(poly) != 4:
            continue
        c = order(poly)
        if not 3 <= np.linalg.norm(c[1] - c[0]) / max(np.linalg.norm(c[3] - c[0]), 1) <= 6.5:
            continue
        image = cv2.imread(im['path']) if image is None else image
        H = cv2.getPerspectiveTransform(c, DST)
        plate = cv2.warpPerspective(image, H, (mf.PLATE_W + 2 * m, mf.PLATE_H + 2 * m), borderMode=cv2.BORDER_REPLICATE)
        ann = []
        for ch in lp['characters']:
            t = str(ch['char_id']).upper()
            if len(t) != 1 or not t.isalnum():
                continue
            (x0, y0), (x1, y1) = ch['bbox_coord']
            q = cv2.perspectiveTransform(np.float32([[[x0, y0], [x1, y0], [x1, y1], [x0, y1]]]), H)[0]
            (a, b), (e, f) = q.min(0), q.max(0)
            ann.append(((a, b, e - a, f - b), t))
        if not ann:
            continue
        seen = set()
        for opt in (mf.SEGMENTATIONS[0], mf.SEGMENTATIONS[2]):
            chars, boxes = mf.segment_characters(plate, **opt)
            for mask, box in zip(chars, boxes):
                j = max(range(len(ann)), key=lambda j: iou(box, ann[j][0]))
                if iou(box, ann[j][0]) < 0.5 or (j, tuple(box)) in seen:
                    continue
                seen.add((j, tuple(box)))
                out.append((mask, ann[j][1], f"{im['image_id']}_{i}"))
                stats[ann[j][1]] += 1
    if k % 200 == 0:
        print(f'{k}/{len(images)} fotos, {len(out)} caràcters', flush=True)
import os
os.makedirs('recognizer/cache', exist_ok=True)
pickle.dump(out, open('recognizer/cache/uc3m_chars.pkl', 'wb'))
print(len(out), dict(sorted(stats.items())))
