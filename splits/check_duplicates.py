"""Busca fotos repetides al corpus (dades/cv, dades/own, dades/own_2). Només llegeix; no toca les anotacions.

  1. idèntiques byte a byte          SHA-256 del fitxer
  2. mateixos píxels                 SHA-256 de la imatge descodificada (mateixa foto amb una altra capçalera/EXIF)
  3. quasi-duplicats                 dHash de 16 x 16 (256 bits) de la foto en grisos; parelles a distància de
                                     Hamming <= --hamming. Es comprova també amb la foto girada en horitzontal.
  4. mateix text de matrícula        el mateix vehicle en dues fotos (no és un duplicat, però no pot quedar
                                     en splits diferents)
Per a cada parella diu si les dues fotos comparteixen split_group i split (si no, és una fuita).
Sortida: outputs/checks/duplicates.csv (parelles sospitoses) i un resum per pantalla.
Executar: python -m splits.check_duplicates [--hamming 20]
"""
import argparse
import hashlib
import itertools
import os

import cv2
import numpy as np
import pandas as pd

from common import manifest

ap = argparse.ArgumentParser()
ap.add_argument('--hamming', type=int, default=20, help='distància màxima de dHash (de 256 bits) per avisar')
args = ap.parse_args()

im = manifest.images().set_index('image_id')
corpus = im[im.subset != 'uc3m']
split = manifest.splits()


def dhash(gray):
    """Diferència entre píxels veïns d'una miniatura de 17 x 16: 256 bits, robust a mida i compressió."""
    small = cv2.resize(gray, (17, 16), interpolation=cv2.INTER_AREA).astype(int)
    return (small[:, 1:] > small[:, :-1]).flatten()


rows = []
for iid, path in corpus.path.items():
    raw = open(path, 'rb').read()
    img = cv2.imread(path)  # amb l'orientació EXIF aplicada, com la resta del projecte
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    rows.append({'image_id': iid, 'file_sha': hashlib.sha256(raw).hexdigest(),
                 'pixel_sha': hashlib.sha256(img.tobytes()).hexdigest(),
                 'dhash': dhash(gray), 'dhash_flip': dhash(gray[:, ::-1])})
d = pd.DataFrame(rows)
print(f'{len(d)} fotos del corpus (dades/cv, dades/own, dades/own_2)')

# text de la matrícula objectiu (complet) de cada foto
p = manifest.plates()
text = p[(p.is_target == 'yes') & (p.text_status == 'full')].set_index('image_id').text

pairs = []
H = np.stack(d.dhash.values)
Hf = np.stack(d.dhash_flip.values)
for i, j in itertools.combinations(range(len(d)), 2):
    a, b = d.iloc[i], d.iloc[j]
    ham = int(min((H[i] != H[j]).sum(), (H[i] != Hf[j]).sum()))
    why = []
    if a.file_sha == b.file_sha:
        why.append('bytes')
    if a.pixel_sha == b.pixel_sha:
        why.append('pixels')
    if ham <= args.hamming:
        why.append(f'dhash={ham}')
    if text.get(a.image_id, '') and text.get(a.image_id) == text.get(b.image_id):
        why.append('mateix_text')
    if why:
        ia, ib = a.image_id, b.image_id
        pairs.append({'a': ia, 'b': ib, 'motiu': ';'.join(why), 'dhash': ham,
                      'subset_a': im.subset.get(ia), 'subset_b': im.subset.get(ib),
                      'text_a': text.get(ia, ''), 'text_b': text.get(ib, ''),
                      'split_a': split.get(ia), 'split_b': split.get(ib),
                      'mateix_grup': im.split_group.get(ia) == im.split_group.get(ib),
                      'orig_a': im.original_path.get(ia), 'orig_b': im.original_path.get(ib)})

out = pd.DataFrame(pairs).sort_values('dhash') if pairs else pd.DataFrame()
os.makedirs('outputs/checks', exist_ok=True)
out.to_csv('outputs/checks/duplicates.csv', index=False)
print(f'Idèntiques byte a byte: {int(d.file_sha.duplicated().sum())} fotos sobrants')
print(f'Mateixos píxels:        {int(d.pixel_sha.duplicated().sum())} fotos sobrants')
if len(out):
    lim = (out.split_a == 'limitation') | (out.split_b == 'limitation')  # fotos apartades: no entren a cap split
    leak = out[(out.split_a != out.split_b) & ~lim]
    print(f'Parelles sospitoses: {len(out)} ({int(lim.sum())} amb una foto apartada); '
          f'FUITES entre train / validation / test: {len(leak)}')
    print(out[['a', 'b', 'motiu', 'subset_a', 'subset_b', 'split_a', 'split_b', 'mateix_grup']].to_string(index=False))
print('Desat outputs/checks/duplicates.csv')
