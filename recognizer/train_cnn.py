"""
Reconeixedor del sistema final: CNN petita (3 blocs conv + 1 capa de 128) que classifica una màscara de caràcter
(64 x 32) en dígits i lletres (taula 2 de l'informe).
Dades:
  - caràcters del corpus (recognizer/cache/segments.pkl, de chars_corpus.py): opció 0 de la segmentació, només les
    matrícules amb 7 regions sense forçar, redreçades amb les cantonades del pose (si coincideixen amb les anotades,
    IoU >= 0,5) i amb les anotades; amb deskew. Es fan servir totes les fotos de segments.pkl
  - els 5.158 caràcters d'uc3m (recognizer/cache/uc3m_chars.pkl, de chars_uc3m.py)
Augmentació: gir petit, escala, inclinació, desplaçament, traç més gruixut o prim i vores retallades.
Sortida: <out>/recognizer.torchscript i <out>/recognizer_classes.txt
Executar des de l'arrel: python -B -m recognizer.train_cnn
"""
import argparse
import os
import pickle

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import model as M

ap = argparse.ArgumentParser()
ap.add_argument('--epochs', type=int, default=30)
ap.add_argument('--seed', type=int, default=0)
ap.add_argument('--out', default='recognizer/weights/cnn')
args = ap.parse_args()
torch.manual_seed(args.seed)
DEV = 'mps' if torch.backends.mps.is_available() else 'cpu'


class Net(nn.Module):
    def __init__(self, n_classes, emb=128):
        super().__init__()

        def block(a, b):
            return nn.Sequential(nn.Conv2d(a, b, 3, padding=1), nn.BatchNorm2d(b), nn.ReLU(),
                                 nn.Conv2d(b, b, 3, padding=1), nn.BatchNorm2d(b), nn.ReLU(), nn.MaxPool2d(2))
        self.features = nn.Sequential(block(1, 32), block(32, 64), block(64, 128))   # 64 x 32 -> 8 x 4
        self.emb = nn.Sequential(nn.Flatten(), nn.Dropout(0.3), nn.Linear(128 * 8 * 4, emb), nn.BatchNorm1d(emb),
                                 nn.ReLU())
        self.head = nn.Linear(emb, n_classes)

    def forward(self, x):
        return self.head(self.emb(self.features(x)))


def augment(m, rng):
    h, w = m.shape
    A = cv2.getRotationMatrix2D((w / 2, h / 2), rng.uniform(-6, 6), rng.uniform(0.85, 1.12))
    A[0, 1] += rng.uniform(-0.15, 0.15)
    A[:, 2] += rng.uniform(-0.08, 0.08, 2) * [w, h]
    m = cv2.warpAffine(m, A, (w, h), flags=cv2.INTER_NEAREST, borderValue=0)
    r = rng.random()
    if r < 0.2:
        m = cv2.dilate(m, np.ones((2, 2), np.uint8))
    elif r < 0.4:
        m = cv2.erode(m, np.ones((2, 2), np.uint8))
    if rng.random() < 0.3:
        t, b = rng.integers(0, max(1, h // 10), 2)
        l, rr = rng.integers(0, max(1, w // 8), 2)
        if h - b - t > 8 and w - rr - l > 4:
            m = m[t:h - b, l:w - rr]
    return m


def to_input(m):
    return cv2.resize(m, (32, 64), interpolation=cv2.INTER_AREA).astype(np.float32) / 255


def iou(a, b):
    inter, _ = cv2.intersectConvexConvex(cv2.convexHull(np.float32(a)), cv2.convexHull(np.float32(b)))
    return inter / (cv2.contourArea(np.float32(a)) + cv2.contourArea(np.float32(b)) - inter + 1e-9)


X, y = [], []
for r in pickle.load(open('recognizer/cache/segments.pkl', 'rb')):
    for name, src in r['sources'].items():
        # les cantonades del pose només valen si són de la matrícula objectiu (en una foto amb dues matrícules el pose
        # pot trobar l'altra, i llavors els caràcters no corresponen al text)
        if name == 'pose' and 'gt' in r['sources'] and iou(src['corners'], r['sources']['gt']['corners']) < 0.5:
            continue
        seg = src['segs'][0]
        if seg['exact'] and seg['chars'] is not None:
            X += M.deskew(seg['chars'])
            y += list(r['text'])
n_corpus = len(X)
for mask, ch, _ in pickle.load(open('recognizer/cache/uc3m_chars.pkl', 'rb')):
    X.append(mask)
    y.append(ch)
classes = ''.join(sorted(set(y)))
t = torch.tensor([classes.index(c) for c in y], device=DEV)

net = Net(len(classes)).to(DEV)
opt = torch.optim.AdamW(net.parameters(), 2e-3, weight_decay=1e-4)
steps = args.epochs * ((len(X) + 127) // 128)
sched = torch.optim.lr_scheduler.OneCycleLR(opt, 3e-3, total_steps=steps)
rng = np.random.default_rng(args.seed)
net.train()
for ep in range(args.epochs):
    order = rng.permutation(len(X))
    total = 0.0
    for b in range(0, len(X), 128):
        ids = order[b:b + 128]
        if len(ids) < 2:              # BatchNorm no pot entrenar amb un lot d'1 sol exemple
            continue
        xb = torch.from_numpy(np.stack([to_input(augment(X[i], rng)) for i in ids]))[:, None].to(DEV)
        loss = F.cross_entropy(net(xb), t[torch.from_numpy(ids).to(DEV)], label_smoothing=0.05)
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        total += loss.item() * len(ids)
    print(f'epoch {ep + 1}/{args.epochs} loss {total / len(X):.3f}', flush=True)

net = net.cpu().eval()
os.makedirs(args.out, exist_ok=True)
torch.jit.trace(net, torch.zeros(7, 1, 64, 32)).save(os.path.join(args.out, 'recognizer.torchscript'))
open(os.path.join(args.out, 'recognizer_classes.txt'), 'w').write(classes + '\n')
print(f'{n_corpus} caràcters del corpus + {len(X) - n_corpus} d\'uc3m, classes {classes} -> {args.out}')
