"""Avalua el detector clàssic (detector/ml/detector.py, adaptat de LicensePlateDetector.py dels professors) amb el manifest.

  recall      = matrícules anotades trobades / total (trobada: alguna regió amb IoU >= 0.5)
  FP per foto = regions que no coincideixen amb cap matrícula, per foto
Fotos: les de la tasca 'detection' (cv / own / own_2) dels splits triats; resultats per subconjunt i vista
(own: també in / out). uc3m no s'hi fa servir (només reconeixement).

Paràmetres:
  profes -> els de LicensePlateDetector.py dels professors
  millor -> la millor combinació de S2 Ex3 e (triada mirant own amb els splits antics: resultat optimista a own)

Ús (des de l'arrel):
  python -m detector.ml.eval_detection --subsets cv own own_2 --splits train validation
  python -m detector.ml.eval_detection --params millor --max 100
Sortida: una fila per foto a outputs/eval_detection/<subsets>_<splits>_<params>.csv
"""
import argparse
import os
import cv2
import imutils
import numpy as np
import pandas as pd

from common import manifest
from detector.ml.detector import DEFAULT, detect, iou

IOU_MIN = 0.5
PARAMS = {
    'profes': DEFAULT,
    'millor': dict(DEFAULT, rect_w=9, close_it=3, ar_min=2.0,  # S2 Ex3 e, mínims dels filtres x0.75
                   w_min=DEFAULT['w_min'] * 0.75, h_min=DEFAULT['h_min'] * 0.75,
                   area_min=DEFAULT['area_min'] * 0.75 ** 2),
}

ap = argparse.ArgumentParser()
ap.add_argument('--subsets', nargs='+', default=['cv', 'own', 'own_2'])
ap.add_argument('--params', default='profes', choices=PARAMS)
ap.add_argument('--splits', nargs='+', default=['train', 'validation'])
ap.add_argument('--max', type=int, default=None, help='com a molt N fotos (mostra fixa)')
args = ap.parse_args()
P = PARAMS[args.params]
if 'test' in args.splits:
    print('AVÍS: avaluar al test el gasta. Fes-ho només amb el sistema ja triat.')

images = manifest.for_task('detection', args.subsets)
images = images[images.split.isin(args.splits)]
if args.max and len(images) > args.max:
    images = images.sample(args.max, random_state=0)
images = images.assign(group=images.subset + ' ' + images.view + np.where(images.subset == 'own', ' ' + images.protocol, ''))
group = dict(zip(images.image_id, images.group))
ann = manifest.plates()
ann = ann[ann.image_id.isin(group)].assign(group=lambda d: d.image_id.map(group),
                                           image_path=lambda d: d.image_id.map(dict(zip(images.image_id, images.path))))
paths = sorted(ann.image_path.unique())

rows = []
for k, (path, plates) in enumerate(ann.groupby('image_path', sort=True), start=1):
    image = cv2.imread(path)
    H, W = image.shape[:2]
    scale = 640 / W if W > 640 else 1
    small = imutils.resize(image, width=640) if W > 640 else image
    gts = [np.float32([[r[f'x{i}'], r[f'y{i}']] for i in range(1, 5)]) * scale for _, r in plates.iterrows()]
    regions = detect(small, W, H, P)
    rows.append({'image_path': path, 'group': plates.group.iloc[0], 'plates': len(gts),
                 'found': sum(any(iou(r, g) >= IOU_MIN for r in regions) for g in gts),
                 'fp': sum(not any(iou(r, g) >= IOU_MIN for g in gts) for r in regions)})
    if k % 100 == 0:
        print(f'{k}/{len(paths)} fotos')

d = pd.DataFrame(rows)
os.makedirs('outputs/eval_detection', exist_ok=True)
out = f"outputs/eval_detection/{'+'.join(args.subsets)}_{'+'.join(args.splits)}_{args.params}.csv"
d.to_csv(out, index=False)

print(f'\nDetector: {args.params} | splits: {", ".join(args.splits)}')
for g, x in d.groupby('group'):
    print(f'{g:20s}: {len(x):4d} fotos, {x.plates.sum():4d} matrícules -> recall {x.found.sum() / x.plates.sum():.2f} '
          f'({x.found.sum()}/{x.plates.sum()}), FP per foto {x.fp.sum() / len(x):.2f}')
print('Desada', out)
