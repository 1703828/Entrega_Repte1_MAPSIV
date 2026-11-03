"""
Exporta un split del corpus en el format de Codabench perquè run_local.py el pugui avaluar:
  <out>/images/<image_id>.jpg  i  <out>/labels.csv (filename,plate)
Només fotos amb una matrícula objectiu de 4 dígits + 3 lletres.
Executar des de l'arrel: python -m splits.export_split --split validation --out outputs/codabench_validation
"""
import argparse
import csv
import os
import re
import shutil

ap = argparse.ArgumentParser()
ap.add_argument('--split', default='validation')
ap.add_argument('--out', default=None)
args = ap.parse_args()
out = args.out or f'outputs/codabench_{args.split}'
split = {r['image_id']: r['split'] for r in csv.DictReader(open('annotations/splits.csv'))}
path = {r['image_id']: r['path'] for r in csv.DictReader(open('annotations/images.csv'))}
os.makedirs(os.path.join(out, 'images'), exist_ok=True)
rows = []
for p in csv.DictReader(open('annotations/plates.csv')):
    iid = p['image_id']
    if p['subset'] != 'uc3m' and p['is_target'] == 'yes' and p['text_status'] == 'full' and split.get(iid) == args.split \
            and re.fullmatch(r'\d{4}[A-Z]{3}', p['text']):
        shutil.copy2(path[iid], os.path.join(out, 'images', f'{iid}.jpg'))
        rows.append([f'{iid}.jpg', p['text']])
with open(os.path.join(out, 'labels.csv'), 'w', newline='') as f:
    csv.writer(f).writerows([['filename', 'plate']] + rows)
print(f'{len(rows)} fotos -> {out}/')
