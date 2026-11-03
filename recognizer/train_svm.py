"""
Reconeixedor anterior (taules 4 i 6 de l'informe): HOG (màscara estirada a 32 x 64, 8 orientacions, cel·la 4,
blocs 2 x 2, L2-Hys) + LinearSVC (C = 1), un per a dígits i un per a lletres. Màscares: segmentació de model.py
(opció 0) de cada matrícula del corpus amb text 4 + 3, redreçada amb les cantonades del pose i amb les anotades;
només les que donen 7 regions.
Sortida: <out> (.npz amb digits_W, digits_b, digits_classes, letters_*). model.py fa servir la CNN, no aquest SVM.
Executar des de l'arrel: python -B -m recognizer.train_svm --splits train --out recognizer/weights/svm/recognizer.npz
"""
import argparse
import csv
import os

import cv2
import numpy as np
import torch
from skimage.feature import hog
from sklearn.svm import LinearSVC

import model as mf

ap = argparse.ArgumentParser()
ap.add_argument('--pose', default='detector/dl/weights/pose.torchscript')
ap.add_argument('--splits', nargs='+', default=['train'])
ap.add_argument('--out', required=True)
args = ap.parse_args()
pose = torch.jit.load(args.pose, map_location='cpu').eval()

split = {r['image_id']: r['split'] for r in csv.DictReader(open('annotations/splits.csv'))}
images = {r['image_id']: r['path'] for r in csv.DictReader(open('annotations/images.csv'))
          if r['subset'] != 'uc3m' and split.get(r['image_id']) in args.splits}
X, y, n = [], [], 0
for p in csv.DictReader(open('annotations/plates.csv')):
    t = p['text']
    if p['image_id'] not in images or p['is_target'] != 'yes' or p['text_status'] != 'full' \
            or not (len(t) == 7 and t[:4].isdigit() and t[4:].isalpha()):
        continue
    image = cv2.imread(images[p['image_id']])
    sources = [mf.find_plate(pose, image)]
    if p['corner_quality'] in ('real', 'approx'):
        sources.append(np.float32([[float(p[f'x{i}']), float(p[f'y{i}'])] for i in range(1, 5)]))
    for corners in sources:
        chars, boxes = mf.segment_characters(mf.rectify_plate(image, corners))
        if len(boxes) != 7:
            continue
        for c, label in zip(chars, t):
            X.append(hog(cv2.resize(c, (32, 64)), orientations=8, pixels_per_cell=(4, 4),
                                     cells_per_block=(2, 2), block_norm='L2-Hys', feature_vector=True))
            y.append(label)
        n += 1
X, y = np.array(X), np.array(y)
weights = {'cell': 4}
for kind, is_kind in (('digits', str.isdigit), ('letters', str.isalpha)):
    sel = np.array([is_kind(c) for c in y])
    clf = LinearSVC(C=1.0, max_iter=20000).fit(X[sel], y[sel])
    weights.update({f'{kind}_W': clf.coef_.astype(np.float32), f'{kind}_b': clf.intercept_.astype(np.float32),
                    f'{kind}_classes': clf.classes_.astype('U1')})
os.makedirs(os.path.dirname(args.out), exist_ok=True)
np.savez_compressed(args.out, **weights)
print(f"{'+'.join(args.splits)}: {n} matrícules, {len(y)} caràcters -> {args.out}")
