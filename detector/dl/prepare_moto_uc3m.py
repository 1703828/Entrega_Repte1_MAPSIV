"""
Fotos extra per al pose: les fotos d'uc3m (train i validation de reconeixement) amb alguna MATRÍCULA DE MOTO
(polígon de 4 punts, proporció < 2,5): 41 fotos. Serveixen perquè el pose aprengui plaques de dues files.
Sortida: detector/dl/datasets/moto_uc3m/images/train + labels/train
Executar des de l'arrel: python -m detector.dl.prepare_moto_uc3m
"""
import pandas as pd

from common import manifest
from detector.dl.dataset import export, pose_line

sp = pd.read_csv('annotations/splits.csv', dtype=str, keep_default_na=False)
pl = manifest.plates()
moto = pl[(pl.subset == 'uc3m') & (pl.aspect.astype(float) < 2.5) & (pl.geometry_type == 'polygon4')].image_id.unique()
ids = sp[sp.image_id.isin(moto) & sp.split.isin(['train', 'validation'])].image_id
images = manifest.images().merge(pd.DataFrame({'image_id': ids, 'split': 'train'}), on='image_id')
done = export(images, pl, 'detector/dl/datasets/moto_uc3m', lambda s: ['train'],
              lambda p, W, H: pose_line(manifest.corners(p._asdict()), W, H, p.corner_quality in ('real', 'approx')))
print(f'{len(done)} fotos, {int(done.n_plates.sum())} matrícules -> detector/dl/datasets/moto_uc3m/')
