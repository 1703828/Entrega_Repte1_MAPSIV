"""
Dataset YOLO-pose amb la partició de annotations/splits.csv:
  images/train   corpus (cv, own, own_2) de train
  images/uc3m    mostra petita d'uc3m per a la detecció (det_uc3m = yes, 53 fotos del train d'uc3m)
  images/val     corpus de validation (ultralytics hi tria el millor epoch)
  images/test    corpus de test
Les 20 fotos apartades (split 'limitation') no hi són.
Sortida: detector/dl/datasets/pose/ i pose.tar
Executar des de l'arrel: python -m detector.dl.prepare_pose
"""
import pandas as pd

from common import manifest
from detector.dl.dataset import export, finish, pose_line

OUT_DIR = 'detector/dl/datasets/pose'
FOLDER = {'train': 'train', 'uc3m': 'uc3m', 'validation': 'val', 'test': 'test'}

sp = pd.read_csv('annotations/splits.csv', dtype=str, keep_default_na=False)
corpus = sp[(sp.subset != 'uc3m') & (sp.split != 'limitation')]
uc3m = sp[sp.det_uc3m == 'yes'].assign(split='uc3m')
images = manifest.images().merge(pd.concat([corpus, uc3m])[['image_id', 'split']], on='image_id')
done = export(images, manifest.plates(), OUT_DIR, lambda s: [FOLDER[s]],
              lambda p, W, H: pose_line(manifest.corners(p._asdict()), W, H, p.corner_quality in ('real', 'approx')))
finish(OUT_DIR, done)
