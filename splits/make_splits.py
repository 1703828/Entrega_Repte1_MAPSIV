"""
Partició nova (1-nov-2026) per als experiments de detecció amb YOLO-pose. No toca annotations/: escriu annotations/splits.csv.

Corpus (cv, own, own_2), sense les 20 fotos apartades (LIMITATIONS, a sota; queden amb el split 'limitation'):
  - es reparteix de nou en train / validation / test (~70 / 15 / 15) per estrats: cv frontal, cv lateral,
    own frontal, own lateral, own_2 (així cada split té de tot);
  - la unitat és el split_group de images.csv: quasi-duplicats, mateix vehicle i mateixa captura (càmera i < 60 s)
    van sempre junts. Els grups grans es col·loquen primer; cada grup va al split que té més dèficit.
uc3m (només per a la detecció; per a la segmentació no serveix):
  - el seu split de reconeixement no canvia (columna split);
  - es tria una MOSTRA PETITA per a train de detecció (columna det_uc3m = yes), perquè uc3m no domini l'entrenament.
    Mida per defecte: la mitjana de fotos de train de cv, own i own_2 (uc3m pesa com un subconjunt més, ~25 %).
  - només fotos del train original d'uc3m (el test d'uc3m queda net per al reconeixement), amb totes les matrícules
    de 4 punts i proporció de cotxe (3-6,5: fora motos), i com a molt una foto per vehicle.
Comprovacions (s'atura si n'hi ha cap): cap grup (split, duplicat, vehicle, captura) ni cap text de matrícula a dos
splits; i les parelles de fotos més semblants (dHash) entre splits diferents, per mirar-les.
Sortida: annotations/splits.csv (image_id, subset, split, split_group, det_uc3m) i splits/splits_resum.txt
Executar: python -m splits.make_splits [--seed 0] [--uc3m-n N]
"""
import argparse
import itertools

import cv2
import numpy as np
import pandas as pd

from common import manifest

ap = argparse.ArgumentParser()
ap.add_argument('--seed', type=int, default=0)
ap.add_argument('--val', type=float, default=0.15)
ap.add_argument('--test', type=float, default=0.15)
ap.add_argument('--uc3m-n', type=int, default=None, help='fotos uc3m per al train de detecció (per defecte: mitjana)')
args = ap.parse_args()
rng = np.random.default_rng(args.seed)

im = manifest.images()
pl = manifest.plates()
old = pd.read_csv('annotations/splits.csv', dtype=str, keep_default_na=False).set_index('image_id')
# fotos apartades: motos (dues files), ciclomotors, formats antic i especial, i fotos 4 + 3 apartades per revisar
LIMITATIONS = [
    'im_2fe5a2528e6d', 'im_4a658840bcf4', 'im_86bd81bcf352', 'im_a2971bc1fceb', 'im_bf0ca7e6bacb',   # motos
    'im_84533e6fa2cc', 'im_ae5e74440e04',
    'im_1d0cae2847b0', 'im_18ae8c88c824',                                                            # ciclomotors
    'im_4dcc46d1b2e9', 'im_ce6e01a783b4', 'im_67de1e1c599d',                                          # antic i especial
    'im_1a53cd2869b7', 'im_5c162731de46', 'im_31bd49b28fe4', 'im_153c3c59a8a7',                       # 4 + 3 apartades
    'im_074ed20bdf52', 'im_0242d9bff28c', 'im_a1dc86af88f1', 'im_bda9dc295014',
]
lim = set(LIMITATIONS)

# 1. Corpus ---------------------------------------------------------------------------------------------------------
c = im[(im.subset != 'uc3m') & ~im.image_id.isin(lim)].copy()
c['stratum'] = c.subset + np.where(c.subset == 'own_2', '', ' ' + c.view)
# un grup -> l'estrat de la majoria de les seves fotos (4 grups barregen subconjunts: mateix cotxe a cv i own_2)
groups = c.groupby('split_group').agg(n=('image_id', 'size'), stratum=('stratum', lambda s: s.mode()[0]))
split_of = {}
for stratum, g in groups.groupby('stratum'):
    total = g.n.sum()
    target = {'validation': args.val * total, 'test': args.test * total}
    target['train'] = total - target['validation'] - target['test']
    have = dict.fromkeys(target, 0)
    order = g.sample(frac=1, random_state=args.seed).sort_values('n', ascending=False, kind='stable')
    for gid, r in order.iterrows():
        s = max(target, key=lambda k: (target[k] - have[k]) / target[k])  # el que més lluny està del seu objectiu
        split_of[gid] = s
        have[s] += r.n
c['split'] = c.split_group.map(split_of)

# 2. Mostra d'uc3m per al train de detecció ---------------------------------------------------------------------------
u = im[im.subset == 'uc3m'].copy()
u['split'] = u.image_id.map(old.split)
pu = pl[pl.subset == 'uc3m'].assign(aspect=lambda d: d.aspect.astype(float))
ok = pu.groupby('image_id').apply(lambda d: (d.geometry_type == 'polygon4').all() and d.aspect.between(3, 6.5).all())
cand = u[(u.split == 'train') & u.image_id.map(ok).fillna(False).astype(bool)]
cand = cand.sample(frac=1, random_state=args.seed).drop_duplicates('vehicle_group')  # una foto per vehicle
n_uc3m = args.uc3m_n or int(round(c[c.split == 'train'].groupby('subset').size().mean()))
chosen = set(cand.image_id.iloc[:n_uc3m])
u['det_uc3m'] = np.where(u.image_id.isin(chosen), 'yes', 'no')

# 3. Fitxer de partició -------------------------------------------------------------------------------------------------
lim_rows = im[im.image_id.isin(lim)].assign(split='limitation')
rows = pd.concat([c, lim_rows, u])
rows['det_uc3m'] = rows.det_uc3m.fillna('no')
rows[['image_id', 'subset', 'split', 'split_group', 'det_uc3m']].to_csv('annotations/splits.csv', index=False)

# 4. Comprovacions ------------------------------------------------------------------------------------------------------
errors = []
for col in ('split_group', 'duplicate_group', 'vehicle_group', 'capture_group'):
    n = c.groupby(col).split.nunique()
    if (n > 1).any():
        errors.append(f'{col}: {int((n > 1).sum())} grups a més d\'un split')
t = pl[pl.image_id.isin(c.image_id) & (pl.text_status == 'full')].assign(split=lambda d: d.image_id.map(
    c.set_index('image_id').split))
n = t.groupby('text').split.nunique()
if (n > 1).any():
    errors.append(f'mateix text a més d\'un split: {list(n[n > 1].index)}')
if set(t.text) & set(pl[pl.image_id.isin(chosen)].text):
    errors.append('text compartit entre la mostra uc3m i el corpus')


def dhash(path):
    g = cv2.resize(cv2.imread(path, cv2.IMREAD_GRAYSCALE), (17, 16), interpolation=cv2.INTER_AREA).astype(int)
    return (g[:, 1:] > g[:, :-1]).flatten()


H = {r.image_id: dhash(r.path) for r in c.itertuples()}
sp = c.set_index('image_id').split
near = sorted((int((H[a] != H[b]).sum()), a, b) for a, b in itertools.combinations(H, 2) if sp[a] != sp[b])[:5]

lines = [f'Partició 1_11 (seed {args.seed}); corpus sense les {len(lim)} fotos apartades',
         c.groupby(['stratum', 'split']).size().unstack(fill_value=0).to_string(), '',
         c.groupby(['subset', 'split']).size().unstack(fill_value=0).to_string(), '',
         f'uc3m per al train de detecció: {len(chosen)} fotos ({pl.image_id.isin(chosen).sum()} matrícules) '
         f'de {len(cand)} candidates; el corpus de train en té {int((c.split == "train").sum())}', '',
         'Parelles més semblants entre splits diferents (dHash de 256 bits; < 40 seria sospitós):']
lines += [f'  {d:3d}  {a} ({sp[a]})  {b} ({sp[b]})' for d, a, b in near]
if near and near[0][0] < 40:
    errors.append('hi ha fotos molt semblants en splits diferents: mira-les')
lines += ['', 'ERRORS:\n' + '\n'.join(errors) if errors else 'Cap fuita entre splits.']
open('splits/splits_resum.txt', 'w').write('\n'.join(lines) + '\n')
print('\n'.join(lines))
print('Desat annotations/splits.csv i splits/splits_resum.txt')
if errors:
    raise SystemExit(1)
