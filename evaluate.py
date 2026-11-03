"""
Avalua model.py sobre el corpus: fotos amb una matrícula objectiu de 4 dígits + 3 lletres dels
splits indicats (annotations/splits.csv). Una lectura és correcta si els 7 caràcters són els de la matrícula objectiu
o, si la foto en té més d'una anotada amb text complet, els de qualsevol d'elles.
  --recognizer recognizer/weights/cnn   CNN del sistema final (per defecte)
Sortida: resum per split i per font, i outputs/evaluate_<tag>.csv foto a foto.
Executar des de l'arrel: python -B evaluate.py
"""
import argparse
import csv
import os
import re
from collections import defaultdict

import model as M
from common.metrics import char_accuracy, plate_accuracy

ap = argparse.ArgumentParser()
ap.add_argument('--detector', default='detector/dl/weights')
ap.add_argument('--recognizer', default='recognizer/weights/cnn')
ap.add_argument('--splits', nargs='+', default=['train', 'validation', 'test'])
ap.add_argument('--tag', default='')
args = ap.parse_args()

split = {r['image_id']: r['split'] for r in csv.DictReader(open('annotations/splits.csv'))}
images = {r['image_id']: r for r in csv.DictReader(open('annotations/images.csv'))}
target, others = {}, defaultdict(set)
for p in csv.DictReader(open('annotations/plates.csv')):
    if p['subset'] == 'uc3m' or p['text_status'] != 'full':
        continue
    if p['is_target'] == 'yes' and re.fullmatch(r'\d{4}[A-Z]{3}', p['text']):
        target[p['image_id']] = p['text']
    elif p['is_target'] != 'yes':
        others[p['image_id']].add(p['text'])
ids = sorted(i for i in target if split.get(i) in args.splits)

model = M.Model(detector_dir=args.detector, recognizer_dir=args.recognizer)
rows = []
for k, iid in enumerate(ids, 1):
    pred = model.predict(images[iid]['path'])
    ok = pred == target[iid] or pred in others[iid]
    rows.append({'image_id': iid, 'subset': images[iid]['subset'], 'split': split[iid], 'truth': target[iid],
                 'pred': pred, 'correct': ok})
    print(f'{k}/{len(ids)}', end='\r', flush=True)
os.makedirs('outputs', exist_ok=True)
out = f'outputs/evaluate{"_" + args.tag if args.tag else ""}.csv'
with open(out, 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)
print(f'\n{args.recognizer}')
for key in args.splits + ['cv', 'own', 'own_2', 'total']:
    g = [r for r in rows if key in (r['split'], r['subset'], 'total')]
    if g:
        pairs = [(r['pred'] if r['correct'] else r['truth'], r['pred']) for r in g]
        print(f'{key:11s} {sum(r["correct"] for r in g):3d}/{len(g):3d} matrícules ({100 * plate_accuracy(pairs):.1f} %), '
              f'caràcters {100 * char_accuracy(pairs):.1f} %')
for r in rows:
    if not r['correct']:
        print('  error:', r['image_id'], r['subset'], r['split'], r['truth'], '->', r['pred'])
print('Desat', out)
