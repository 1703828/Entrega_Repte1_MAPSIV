"""
Validació creuada dels reconeixedors (taula 4 i figura 2 de l'informe). El pipeline de model.py (pose, redreçat,
4 segmentacions, force_seven i tria de la lectura més segura) és fix i només canvia el reconeixedor. 5 parts per
grup (un vehicle sempre a la mateixa part) sobre les 231 fotos de cotxe; a cada volta el reconeixedor s'entrena amb
els caràcters de les altres 4 parts. Les 7 motos s'avaluen amb el reconeixedor entrenat amb les 231.
Reconeixedors (--rec): svm (HOG + LinearSVC), svm_lbp, svm_lbpror, svm_hoglbp, knn (retrieval 1-NN amb HOG),
cnn (la CNN de train_cnn.py), cnnk (retrieval 1-NN amb l'embedding de la CNN). --deskew i --uc3m com al sistema final.
Cal haver executat abans recognizer/chars_corpus.py i recognizer/chars_uc3m.py.
Sortida: outputs/cross_validation/<rec>_<tag>.csv (foto a foto) i una línia a outputs/cross_validation/resum.txt
Executar des de l'arrel (taula 4):
  python -B -m recognizer.cross_validation --rec svm knn --tag base
  python -B -m recognizer.cross_validation --rec svm svm_lbp cnn cnnk --deskew --uc3m --epochs 30 --tag d_u
"""
import argparse
import csv
import os
import pickle

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from skimage.feature import hog
from sklearn.svm import LinearSVC

from common.metrics import plate_accuracy, char_accuracy

ap = argparse.ArgumentParser()
ap.add_argument('--rec', nargs='+', default=['svm', 'knn', 'cnn', 'cnnk'])
ap.add_argument('--train_segs', nargs='+', type=int, default=[0])
ap.add_argument('--epochs', type=int, default=40)
ap.add_argument('--k', type=int, default=5)
ap.add_argument('--seed', type=int, default=0)
ap.add_argument('--tag', default='')
ap.add_argument('--deskew', action='store_true', help='treu la inclinació comuna dels 7 caràcters abans de llegir')
ap.add_argument('--uc3m', action='store_true', help='afegeix els caràcters d\'uc3m (recognizer/cache/uc3m_chars.pkl)')
args = ap.parse_args()
torch.manual_seed(args.seed)
DEV = 'mps' if torch.backends.mps.is_available() else 'cpu'
np.random.seed(args.seed)

rows = pickle.load(open('recognizer/cache/segments.pkl', 'rb'))
cars = [r for r in rows if r['split'] != 'motos']
motos = [r for r in rows if r['split'] == 'motos']


def folds(rows, k):
    rng = np.random.default_rng(0)
    groups = {}
    for r in rows:
        groups.setdefault(r['group'], []).append(r['image_id'])
    size, out = [0] * k, {}
    for g in sorted(groups, key=lambda g: (-len(groups[g]), rng.random())):
        f = int(np.argmin(size))
        for iid in groups[g]:
            out[iid] = f
        size[f] += len(groups[g])
    return out


def iou(a, b):
    inter, _ = cv2.intersectConvexConvex(cv2.convexHull(np.float32(a)), cv2.convexHull(np.float32(b)))
    return inter / (cv2.contourArea(np.float32(a)) + cv2.contourArea(np.float32(b)) - inter + 1e-9)


def slant(mask):
    """Inclinació d'una màscara (moments: mu11 / mu02, com el deskew clàssic de MNIST)."""
    m = cv2.moments(mask, binaryImage=True)
    return m['mu11'] / m['mu02'] if m['mu02'] > 1e-2 else 0.0


def deskew(chars):
    """Els 7 caràcters d'una matrícula tenen la mateixa inclinació (perspectiva o placa en cursiva): s'estima amb la
    mediana dels 7 i s'aplica la mateixa cisalla horitzontal a tots."""
    if not args.deskew:
        return list(chars)
    k = float(np.clip(np.median([slant(c) for c in chars]), -0.5, 0.5))
    out = []
    for c in chars:
        h, w = c.shape
        pad = int(abs(k) * h / 2) + 1
        c = cv2.copyMakeBorder(c, 0, 0, pad, pad, cv2.BORDER_CONSTANT, value=0)
        M = np.float32([[1, -k, k * h / 2], [0, 1, 0]])
        c = cv2.warpAffine(c, M, (c.shape[1], h), flags=cv2.INTER_NEAREST)
        cols = np.flatnonzero(c.any(0))
        out.append(cv2.copyMakeBorder(c[:, cols[0]:cols[-1] + 1], 0, 0, 2, 2, cv2.BORDER_CONSTANT, value=0)
                   if len(cols) else c)
    return out


UC3M = pickle.load(open('recognizer/cache/uc3m_chars.pkl', 'rb')) if args.uc3m else []


def training_chars(rows):
    X, y = [], []
    for r in rows:
        for name, src in r['sources'].items():
            if name == 'pose' and 'gt' in r['sources'] and iou(src['corners'], r['sources']['gt']['corners']) < 0.5:
                continue        # el pose ha trobat una altra matrícula: els caràcters no són del text
            for s in args.train_segs:
                seg = src['segs'][s]
                if seg['exact'] and seg['chars'] is not None:
                    X += deskew(seg['chars'])
                    y += list(r['text'])
    for mask, ch, _ in UC3M:
        X.append(mask)
        y.append(ch)
    return X, np.array(y)


# Descriptors --------------------------------------------------------------------------------------------------------

def hog_feat(mask):
    return hog(cv2.resize(mask, (32, 64)), orientations=8, pixels_per_cell=(4, 4), cells_per_block=(2, 2),
               block_norm='L2-Hys', feature_vector=True)


def lbp_feat(mask, method='uniform'):
    """LBP (8 veïns, radi 1) per blocs 4 x 2 de la màscara a 32 x 64: histograma de cada bloc. 'uniform' distingeix
    l'orientació dels patrons; 'ror' els fa invariants a la rotació (un traç horitzontal = un de vertical)."""
    from skimage.feature import local_binary_pattern
    img = cv2.resize(mask, (32, 64))
    codes = local_binary_pattern(img, 8, 1, method)
    nb = 10 if method == 'uniform' else 36
    feats = []
    for by in range(4):
        for bx in range(2):
            block = codes[by * 16:(by + 1) * 16, bx * 16:(bx + 1) * 16].ravel()
            if method == 'uniform':
                h = np.bincount(block.astype(int), minlength=nb)[:nb]
            else:
                h = np.array([(block == v).sum() for v in ROR_CODES])
            feats.append(h / 256.0)
    return np.concatenate(feats)


ROR_CODES = sorted({int(min(((c >> i) | (c << (8 - i))) & 255 for i in range(8))) for c in range(256)})

FEATS = {'hog': hog_feat, 'lbp': lambda m: lbp_feat(m, 'uniform'), 'lbpror': lambda m: lbp_feat(m, 'ror'),
         'hoglbp': lambda m: np.concatenate([hog_feat(m), lbp_feat(m, 'uniform')])}


def to_input(mask):
    """Màscara -> 64 x 32 float [0, 1] (estirada, com el HOG)."""
    return cv2.resize(mask, (32, 64), interpolation=cv2.INTER_AREA).astype(np.float32) / 255


# Reconeixedors: fit(X, y) i scores(mask, kind) -> (classes, puntuacions) ----------------------------------------------

class SVM:
    def __init__(self, feat='hog'):
        self.f = FEATS[feat]

    def fit(self, X, y):
        F_ = np.array([self.f(m) for m in X])
        self.m = {}
        for kind, sel in (('d', np.char.isdigit(y)), ('l', np.char.isalpha(y))):
            clf = LinearSVC(C=1.0, max_iter=20000).fit(F_[sel], y[sel])
            self.m[kind] = clf
        return self

    def scores(self, masks, kind):
        clf = self.m[kind]
        return clf.classes_, clf.decision_function(np.array([self.f(m) for m in masks]))


def knn_scores(G, gy, Q, classes, k=1):
    """Retrieval: per a cada classe, la similitud (cosinus) del seu exemplar més semblant (mitjana dels k millors)."""
    sim = Q @ G.T
    out = np.full((len(Q), len(classes)), -1.0)
    for j, c in enumerate(classes):
        s = np.sort(sim[:, gy == c], axis=1)[:, ::-1][:, :k]
        out[:, j] = s.mean(1)
    return out


class KNN:
    def fit(self, X, y):
        G = np.array([hog_feat(m) for m in X])
        self.G = G / np.linalg.norm(G, axis=1, keepdims=True)
        self.y = y
        return self

    def scores(self, masks, kind):
        sel = np.char.isdigit(self.y) if kind == 'd' else np.char.isalpha(self.y)
        classes = np.unique(self.y[sel])
        Q = np.array([hog_feat(m) for m in masks])
        Q = Q / np.linalg.norm(Q, axis=1, keepdims=True)
        return classes, knn_scores(self.G[sel], self.y[sel], Q, classes)


class Net(nn.Module):
    def __init__(self, n_classes, emb=128):
        super().__init__()
        def block(a, b):
            return nn.Sequential(nn.Conv2d(a, b, 3, padding=1), nn.BatchNorm2d(b), nn.ReLU(),
                                 nn.Conv2d(b, b, 3, padding=1), nn.BatchNorm2d(b), nn.ReLU(), nn.MaxPool2d(2))
        self.features = nn.Sequential(block(1, 32), block(32, 64), block(64, 128))   # 64x32 -> 8x4
        self.emb = nn.Sequential(nn.Flatten(), nn.Dropout(0.3), nn.Linear(128 * 8 * 4, emb), nn.BatchNorm1d(emb),
                                 nn.ReLU())
        self.head = nn.Linear(emb, n_classes)

    def embed(self, x):
        return self.emb(self.features(x))

    def forward(self, x):
        return self.head(self.embed(x))


def augment(m, rng):
    """Màscara binària: petit gir, escala, desplaçament i inclinació, i engruixir o aprimar el traç."""
    h, w = m.shape
    a = rng.uniform(-6, 6)
    s = rng.uniform(0.85, 1.12)
    M = cv2.getRotationMatrix2D((w / 2, h / 2), a, s)
    M[0, 1] += rng.uniform(-0.15, 0.15)
    M[:, 2] += rng.uniform(-0.08, 0.08, 2) * [w, h]
    m = cv2.warpAffine(m, M, (w, h), flags=cv2.INTER_NEAREST, borderValue=0)
    r = rng.random()
    if r < 0.2:
        m = cv2.dilate(m, np.ones((2, 2), np.uint8))
    elif r < 0.4:
        m = cv2.erode(m, np.ones((2, 2), np.uint8))
    # retall aleatori de les vores (caràcter tallat per la segmentació)
    if rng.random() < 0.3:
        t, b = rng.integers(0, max(1, h // 10), 2)
        l, rr = rng.integers(0, max(1, w // 8), 2)
        m = m[t:h - b, l:w - rr] if h - b - t > 8 and w - rr - l > 4 else m
    return m


class CNN:
    def __init__(self, retrieval=False):
        self.retrieval = retrieval

    def fit(self, X, y):
        self.classes = np.unique(y)
        idx = {c: i for i, c in enumerate(self.classes)}
        t = torch.tensor([idx[c] for c in y])
        self.net = Net(len(self.classes)).to(DEV)
        t = t.to(DEV)
        opt = torch.optim.AdamW(self.net.parameters(), 2e-3, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, 3e-3, total_steps=args.epochs * ((len(X) + 127) // 128))
        rng = np.random.default_rng(args.seed)
        self.net.train()
        for ep in range(args.epochs):
            order = rng.permutation(len(X))
            for b in range(0, len(X), 128):
                ids = order[b:b + 128]
                if len(ids) < 2:          # BatchNorm no pot entrenar amb un lot d'1 sol exemple
                    continue
                xb = torch.from_numpy(np.stack([to_input(augment(X[i], rng)) for i in ids]))[:, None].to(DEV)
                loss = F.cross_entropy(self.net(xb), t[torch.from_numpy(ids).to(DEV)], label_smoothing=0.05)
                opt.zero_grad()
                loss.backward()
                opt.step()
                sched.step()
        self.net = self.net.cpu().eval()
        if self.retrieval:
            self.G = self._embed(X)
            self.y = y
        return self

    def _embed(self, masks):
        with torch.no_grad():
            e = self.net.embed(torch.from_numpy(np.stack([to_input(m) for m in masks]))[:, None]).numpy()
        return e / (np.linalg.norm(e, axis=1, keepdims=True) + 1e-9)

    def scores(self, masks, kind):
        isk = np.char.isdigit if kind == 'd' else np.char.isalpha
        if self.retrieval:
            sel = isk(self.y)
            classes = np.unique(self.y[sel])
            return classes, knn_scores(self.G[sel], self.y[sel], self._embed(masks), classes)
        with torch.no_grad():
            logits = self.net(torch.from_numpy(np.stack([to_input(m) for m in masks]))[:, None])
        logp = F.log_softmax(logits, 1).numpy()
        sel = isk(self.classes)
        return self.classes[sel], logp[:, sel]


RECS = {'svm': SVM, 'svm_lbp': lambda: SVM('lbp'), 'svm_lbpror': lambda: SVM('lbpror'),
        'svm_hoglbp': lambda: SVM('hoglbp'), 'knn': KNN, 'cnn': lambda: CNN(False), 'cnnk': lambda: CNN(True)}


def read(rec, chars):
    chars = deskew(chars)
    text, conf = '', np.inf
    for kind, part in (('d', chars[:4]), ('l', chars[4:])):
        classes, s = rec.scores(part, kind)
        for row in s:
            o = np.argsort(row)[::-1]
            text += str(classes[o[0]])
            conf = min(conf, float(row[o[0]] - row[o[1]]))
    return text, conf


def evaluate(rec, r):
    """Lectura de model_final (tria per (sense forçar, seguretat)) amb les cantonades del pose; i l'oracle."""
    best, reads = None, []
    for seg in r['sources']['pose']['segs']:
        if seg['chars'] is None:
            continue
        text, conf = read(rec, seg['chars'])
        reads.append(text)
        if best is None or (seg['exact'], conf) > best[0]:
            best = ((seg['exact'], conf), text)
    return (best[1] if best else '0000BBB'), r['text'] in reads


os.makedirs('outputs/cross_validation', exist_ok=True)
fold = folds(cars, args.k)
for name in args.rec:
    out = []
    for f in range(args.k):
        rec = RECS[name]().fit(*training_chars([r for r in cars if fold[r['image_id']] != f]))
        for r in cars:
            if fold[r['image_id']] == f:
                pred, oracle = evaluate(rec, r)
                out.append({'image_id': r['image_id'], 'split': r['split'], 'subset': r['subset'], 'truth': r['text'],
                            'pred': pred, 'oracle': oracle})
    rec = RECS[name]().fit(*training_chars(cars))
    for r in motos:
        pred, oracle = evaluate(rec, r)
        out.append({'image_id': r['image_id'], 'split': 'motos', 'subset': r['subset'], 'truth': r['text'],
                    'pred': pred, 'oracle': oracle})
    with open(f'outputs/cross_validation/{name}_{args.tag}.csv', 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(out[0]))
        w.writeheader()
        w.writerows(out)
    parts = []
    for sp in ('all', 'train', 'validation', 'test', 'motos'):
        sel = [o for o in out if (o['split'] != 'motos' if sp == 'all' else o['split'] == sp)]
        pairs = [(o['truth'], o['pred']) for o in sel]
        parts.append(f"{sp} {plate_accuracy(pairs):.3f}/{char_accuracy(pairs):.3f} "
                     f"({sum(o['truth'] == o['pred'] for o in sel)}/{len(sel)}, oracle {sum(o['oracle'] for o in sel)})")
    line = f"{name:5s} segs={args.train_segs} ep={args.epochs} {args.tag}: " + ' | '.join(parts)
    print(line, flush=True)
    open('outputs/cross_validation/resum.txt', 'a').write(line + '\n')
