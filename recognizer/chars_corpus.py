"""
Caràcters del corpus per entrenar i comparar reconeixedors. Per a cada foto amb text complet 4 + 3 (train, validation
i test de annotations/splits.csv, i les 7 motos apartades: matrícula de dues files, proporció < 1,5) fa el mateix que model.py fins abans de reconèixer:
cantonades (les del pose i les anotades), matrícula redreçada i, per a cada opció de SEGMENTATIONS, les caixes i
màscares després de force_seven. Pose: detector/dl/weights (pose i pose de motos).
Sortida: recognizer/cache/segments.pkl (només per entrenar; model.py no fa servir pickle)
Executar des de l'arrel: python -B -m recognizer.chars_corpus
"""
import csv
import os
import pickle

import cv2
import numpy as np

import model as mf
model = mf.Model(detector_dir='detector/dl/weights', recognizer_dir='recognizer/weights/cnn')

split = {r['image_id']: r for r in csv.DictReader(open('annotations/splits.csv'))}
# motos: fotos apartades (split 'limitation') amb una matrícula objectiu de dues files (amplada / alçada < 1,5)
motos = {p['image_id'] for p in csv.DictReader(open('annotations/plates.csv')) if p['is_target'] == 'yes'
         and split.get(p['image_id'], {}).get('split') == 'limitation' and p['aspect'] and float(p['aspect']) < 1.5}
images = {r['image_id']: r for r in csv.DictReader(open('annotations/images.csv')) if r['subset'] != 'uc3m'}

rows = []
for p in csv.DictReader(open('annotations/plates.csv')):
    iid, t = p['image_id'], p['text']
    if iid not in images or p['is_target'] != 'yes' or p['text_status'] != 'full':
        continue
    if not (len(t) == 7 and t[:4].isdigit() and t[4:].isalpha()):
        continue
    if iid in motos:
        sp, group = 'motos', 'moto_' + iid
    elif iid in split and split[iid]['split'] in ('train', 'validation', 'test'):
        sp, group = split[iid]['split'], split[iid]['split_group']
    else:
        continue
    image = cv2.imread(images[iid]['path'])
    corners = mf.find_plate(model.pose, image)
    moto_corners = mf.find_plate(model.pose_moto, image)
    if mf.plate_ratio(moto_corners) < mf.MOTO_RATIO:
        corners = moto_corners
    gt = (np.float32([[float(p[f'x{i}']), float(p[f'y{i}'])] for i in range(1, 5)])
          if p['corner_quality'] in ('real', 'approx') else None)
    sources = {}
    for name, c in (('pose', corners), ('gt', gt)):
        if c is None:
            continue
        plate = mf.rectify_plate(image, c)
        segs = []
        for opt in mf.SEGMENTATIONS:
            chars, boxes = mf.segment_characters(plate, **opt)
            exact = len(boxes) == 7
            chars, boxes = mf.force_seven(chars, boxes)
            segs.append({'exact': exact, 'chars': chars if len(boxes) == 7 else None, 'boxes': boxes})
        sources[name] = {'corners': c, 'plate': plate, 'segs': segs}
    rows.append({'image_id': iid, 'subset': images[iid]['subset'], 'split': sp, 'group': group, 'text': t,
                 'sources': sources})
    print(len(rows), iid, sp, t, end='\r')
os.makedirs('recognizer/cache', exist_ok=True)
pickle.dump(rows, open('recognizer/cache/segments.pkl', 'wb'))
print('\n', {s: sum(r['split'] == s for r in rows) for s in ('train', 'validation', 'test', 'motos')})
