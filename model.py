"""
Lectura de matrícules espanyoles (Codabench): foto -> "1234BCD".

Passos (un per funció, en ordre):
  1. read_image          llegir la foto
  2. find_plate          YOLO-pose: les 4 cantonades de la matrícula. Dues passades: a la foto sencera i, després, a
                         un retall al voltant de la matrícula trobada (hi surt més gran i les cantonades són més justes)
  3. rectify_plate       homografia: la matrícula es redreça a 400 x 85 píxels (+ 8 de vora). Si és de MOTO (dues
                         files: proporció < MOTO_RATIO), es redreça a 200 x 170, es talla la franja blava i la fila de
                         baix es posa al costat de la de dalt: queda com una placa d'una fila i la resta no canvia
  4. segment_characters  els caràcters són taques fosques sobre la placa clara:
                           a) fora la franja blava de la UE (no té caràcters)
                           b) black-hat + Otsu: màscara de tinta (traços foscos més prims que un caràcter)
                           c) (opcional) fora les línies llargues i rectes: són el marc de la placa, no caràcters
                           d) components connexos amb mida de caràcter
                           e) es tornen a ajuntar els trossos d'un caràcter partit en vertical
                           f) fora les vores: a tocar d'un costat i estretes, massa altes o desalineades
                              (opcional: si no són estretes, es retallen a l'alçada del text)
  5. force_seven         una matrícula té 7 caràcters: si en sobren, fora el més diferent; si en falten, es parteix
                         la caixa més ampla
                         (si n'hi ha 8 i el de més a l'esquerra és a la zona de la franja, és la franja)
  6. read_characters     es treu la inclinació comuna dels 7 caràcters (deskew) i una CNN petita classifica cada
                         màscara: 4 dígits i 3 lletres de les que fan servir les matrícules espanyoles
Cap manera de segmentar ni cap retall va bé a totes les fotos: els passos 3-6 es fan amb dues versions de les
cantonades (VARIANTS) i quatre opcions de segmentació (SEGMENTATIONS), i es queda la lectura que ha donat 7 caràcters
sense haver de forçar-ho i que la CNN veu més segura.

Pesos: pose.torchscript (YOLO-pose de 4 punts) i pose_moto.torchscript (pose que també coneix les motos) a
detector/dl/weights/; recognizer.torchscript (CNN) i recognizer_classes.txt (classes, en ordre) a recognizer/weights/cnn/.
A la submission de Codabench (submission/submission.zip) tots quatre van a weights/: Model('weights').
Ús al repositori: Model(detector_dir='detector/dl/weights', recognizer_dir='recognizer/weights/cnn').
Només torch, OpenCV i numpy. predict() sempre retorna 7 caràcters i no peta mai.
"""
import os

import cv2
import numpy as np
import torch

FALLBACK = '0000BBB'          # resposta quan no es pot llegir res
IMGSZ = 640                   # mida d'entrada del YOLO-pose
PAD = 2.5                     # 2a passada: marge del retall, en mides de la matrícula per banda
PLATE_W, PLATE_H = 400, 85    # matrícula redreçada (proporció de la placa espanyola, 520 x 110 mm)
MARGIN = 8                    # vora al voltant de la matrícula redreçada
CHAR_W = 34                   # amplada típica d'un caràcter a la matrícula redreçada
STRIP_ZONE = 0.09             # la franja blava ocupa el ~9 % esquerre de la placa
# Cantonades que es proven: les del pose i la placa un 8 % més alta per dalt i per baix. L'error del pose és sobretot
# vertical (percentil 90: 14 % de l'alçada) i una placa massa justa talla els caràcters.
VARIANTS = [0.0, 0.08]
LETTERS = 'BCDFGHJKLMNPRSTVWXYZ'   # lletres de les matrícules espanyoles actuals (sense vocals, Ñ ni Q)
MOTO_RATIO = 2.0              # amplada / alçada: cotxe 3-6, moto (dues files) 1-1,5

# Opcions del pas 4 que es proven, per ordre (en cas d'empat, guanya la primera)
SEGMENTATIONS = [
    dict(strip='rectangle', no_frame=False, trim_border=False),   # la franja s'esborra sencera
    dict(strip='blue', no_frame=False, trim_border=False),        # només es descarten les taques blaves
    dict(strip='rectangle', no_frame=True, trim_border=False),    # + fora el marc
    dict(strip='rectangle', no_frame=True, trim_border=True),     # + caràcter enganxat a la vora: es retalla
]


class Model:
    def __init__(self, weights_dir='weights', detector_dir=None, recognizer_dir=None):
        # Codabench: tots els pesos a weights_dir. Al repositori: detector/dl/weights i recognizer/weights/cnn
        det, rec = detector_dir or weights_dir, recognizer_dir or weights_dir
        self.pose = torch.jit.load(os.path.join(det, 'pose.torchscript'), map_location='cpu').eval()
        self.cnn = torch.jit.load(os.path.join(rec, 'recognizer.torchscript'), map_location='cpu').eval()
        self.classes = np.array(list(open(os.path.join(rec, 'recognizer_classes.txt')).read().strip()))
        moto = os.path.join(det, 'pose_moto.torchscript')   # opcional: pose que també ha vist motos
        self.pose_moto = torch.jit.load(moto, map_location='cpu').eval() if os.path.exists(moto) else None

    def predict(self, image_path):
        try:
            return self.predict_details(image_path)['text']
        except Exception:
            return FALLBACK

    def predict_details(self, image_path):
        """Per avaluar: el resultat de cada etapa (cantonades, confiança, matrícula redreçada, segmentacions)."""
        image = read_image(image_path)
        corners, conf = find_plate(self.pose, image, with_conf=True)
        moto = False
        if self.pose_moto is not None:
            moto_corners, moto_conf = find_plate(self.pose_moto, image, with_conf=True)
            if plate_ratio(moto_corners) < MOTO_RATIO:
                corners, conf, moto = moto_corners, moto_conf, True
        details = {'corners': corners, 'conf': conf, 'moto': moto, 'plate': rectify_plate(image, corners)}
        best = None
        for k, extra in enumerate(VARIANTS):
            plate = rectify_plate(image, grow(corners, extra))
            read = read_plate(self.cnn, self.classes, plate, details if k == 0 else None)
            if read and (best is None or read[0] > best[0]):
                best = read
        details['text'] = best[1] if best else FALLBACK
        return details


def read_plate(cnn, classes, plate, details=None):
    """Passos 4-6 amb cada opció de SEGMENTATIONS. Torna ((sense forçar, seguretat), text) de la millor lectura."""
    best = None
    for options in SEGMENTATIONS:
        chars, boxes = segment_characters(plate, **options)                  # 4
        exact = len(boxes) == 7
        n_regions = len(boxes)
        chars, boxes = force_seven(chars, boxes)                             # 5
        if details is not None:
            details.setdefault('segs', []).append({'n': n_regions, 'boxes': boxes, 'chars': chars})
        if len(boxes) != 7:
            continue
        text, confidence = read_characters(cnn, classes, chars)              # 6
        if details is not None:
            details['segs'][-1].update(text=text, conf=confidence)
        if best is None or (exact, confidence) > best[0]:
            best = ((exact, confidence), text)
    return best


# 1. Llegir la foto ---------------------------------------------------------------------------------------------------

def read_image(path):
    image = cv2.imread(path)
    if image is None:
        raise ValueError(f'no es pot llegir {path}')
    return image


# 2. Trobar la matrícula: YOLO-pose -------------------------------------------------------------------------------------

def detect_corners(pose, image):
    """Les 4 cantonades (TL, TR, BR, BL) de la detecció amb més confiança. La foto es redueix a 640 x 640 mantenint la
    proporció i s'omple de gris (com en l'entrenament); després els punts es tornen a la mida de la foto."""
    h, w = image.shape[:2]
    r = min(IMGSZ / h, IMGSZ / w)
    nw, nh = round(w * r), round(h * r)
    left, top = round((IMGSZ - nw) / 2 - 0.1), round((IMGSZ - nh) / 2 - 0.1)
    img = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_LINEAR)
    img = cv2.copyMakeBorder(img, top, IMGSZ - nh - top, left, IMGSZ - nw - left, cv2.BORDER_CONSTANT,
                             value=(114, 114, 114))
    x = torch.from_numpy(img[:, :, ::-1].transpose(2, 0, 1).copy()).float().div(255).unsqueeze(0)  # BGR -> RGB
    with torch.no_grad():
        out = pose(x)[0].numpy()        # 17 x 8400: cx, cy, w, h, confiança, 4 x (x, y, visibilitat)
    best = int(np.argmax(out[4]))
    kpts = out[5:, best].reshape(4, 3)[:, :2]
    kpts[:, 0] = (kpts[:, 0] - left) / r
    kpts[:, 1] = (kpts[:, 1] - top) / r
    detect_corners.conf = float(out[4, best])
    return kpts.astype(np.float32)


def find_plate(pose, image, with_conf=False):
    """1a passada a la foto sencera; 2a passada a un retall quadrat al voltant de la matrícula trobada.
    with_conf: torna també la confiança de la 1a passada (la de la foto sencera)."""
    if with_conf:
        corners = find_plate(pose, image)
        return corners, find_plate.conf
    corners = detect_corners(pose, image)
    find_plate.conf = detect_corners.conf
    H, W = image.shape[:2]
    (x1, y1), (x2, y2) = corners.min(0), corners.max(0)
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    side = max(x2 - x1, y2 - y1) * (1 + 2 * PAD)
    a, b = int(max(0, cx - side / 2)), int(max(0, cy - side / 2))
    c, d = int(min(W, cx + side / 2)), int(min(H, cy + side / 2))
    crop = image[b:d, a:c]
    if crop.shape[0] < 32 or crop.shape[1] < 32:
        return corners
    return detect_corners(pose, crop) + np.float32([a, b])


# 3. Redreçar la matrícula ----------------------------------------------------------------------------------------------

def rectify_plate(image, corners):
    """Placa d'una fila -> 400 x 85 (+ vora). Placa de moto (dues files) -> rectify_moto."""
    if plate_ratio(corners) < MOTO_RATIO:
        return rectify_moto(image, corners)
    m = MARGIN
    dst = np.float32([[m, m], [m + PLATE_W - 1, m], [m + PLATE_W - 1, m + PLATE_H - 1], [m, m + PLATE_H - 1]])
    H = cv2.getPerspectiveTransform(corners.astype(np.float32), dst)
    return cv2.warpPerspective(image, H, (PLATE_W + 2 * m, PLATE_H + 2 * m), borderMode=cv2.BORDER_REPLICATE)


def grow(corners, extra):
    """Allarga el quadrilàter cap amunt i cap avall una fracció 'extra' de l'alçada, seguint els costats."""
    tl, tr, br, bl = corners
    return np.float32([tl + extra * (tl - bl), tr + extra * (tr - br), br + extra * (br - tr), bl + extra * (bl - tl)])


def plate_ratio(corners):
    """Amplada / alçada del quadrilàter (mitjana dels costats oposats)."""
    tl, tr, br, bl = corners
    return (np.linalg.norm(tr - tl) + np.linalg.norm(br - bl)) / (np.linalg.norm(bl - tl) + np.linalg.norm(br - tr))


def rectify_moto(image, corners):
    """Moto: 4 dígits a dalt i 3 lletres a baix, amb la franja blava a l'esquerra de les dues files. Es redreça a
    200 x 170, es tallen les columnes blaves de l'esquerra i les dues files es posen una al costat de l'altra."""
    H = cv2.getPerspectiveTransform(corners.astype(np.float32), np.float32([[0, 0], [199, 0], [199, 169], [0, 169]]))
    two_rows = cv2.warpPerspective(image, H, (200, 170), borderMode=cv2.BORDER_REPLICATE)
    blue = cv2.inRange(cv2.cvtColor(two_rows, cv2.COLOR_BGR2HSV), (90, 90, 80), (135, 255, 255)) > 0
    blue_cols = np.flatnonzero(blue.mean(0)[:70] > 0.25)
    cut = int(blue_cols[-1]) + 3 if len(blue_cols) else 0
    one_row = cv2.resize(np.hstack([two_rows[:85, cut:], two_rows[85:, cut:]]), (PLATE_W, PLATE_H))
    return cv2.copyMakeBorder(one_row, MARGIN, MARGIN, MARGIN, MARGIN, cv2.BORDER_REPLICATE)


# 4. Segmentar els caràcters ---------------------------------------------------------------------------------------------

def segment_characters(plate, strip='rectangle', no_frame=False, trim_border=False):
    """Màscares (traç blanc sobre negre) i caixes (x, y, w, h) dels caràcters, d'esquerra a dreta."""
    H, W = plate.shape[:2]
    hsv = cv2.cvtColor(plate, cv2.COLOR_BGR2HSV)

    # a) franja blava: rectangle d'un component blau, a l'esquerra, alt i estret
    strip_mask = np.zeros((H, W), np.uint8)
    n_blue, _, blue_stats, _ = cv2.connectedComponentsWithStats(cv2.inRange(hsv, (90, 70, 40), (135, 255, 255)))
    for x, y, w, h, area in blue_stats[1:]:
        if x < 0.15 * W and x + w < 0.25 * W and w < 0.20 * W and h > 0.55 * H and area / (w * h) > 0.35:
            strip_mask[y:y + h, x:x + w] = 255
    strong_blue = cv2.inRange(hsv, (90, 90, 80), (135, 255, 255))      # blau saturat i clar: la tinta no hi entra

    # b) tinta: el black-hat deixa els traços foscos més prims que un caràcter; Otsu els binaritza
    gray = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, cv2.getStructuringElement(cv2.MORPH_RECT, (CHAR_W, CHAR_W)))
    ink = cv2.threshold(blackhat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    if strip == 'rectangle':
        ink[strip_mask > 0] = 0

    # c) marc: línies verticals de gairebé tota l'alçada i horitzontals llargues
    if no_frame:
        vertical = cv2.morphologyEx(ink, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, int(0.8 * (H - 16)))))
        horizontal = cv2.morphologyEx(ink, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (int(0.25 * W), 1)))
        ink[(vertical > 0) | (horizontal > 0)] = 0

    # d) components: fora el soroll, el que és més ample que mitja placa i (strip='blue') el que és sobretot blau
    n, labels, stats, _ = cv2.connectedComponentsWithStats(ink, connectivity=8)
    candidates = []
    for i in range(1, n):
        x, y, w, h, area = map(int, stats[i])
        if h < 0.25 * H or area < max(3, 0.0002 * H * W):
            continue
        if strip == 'blue' and (strong_blue[labels == i] > 0).mean() > 0.5:
            continue
        if w > 0.50 * W:
            continue
        candidates.append((x, y, w, h, [i]))
    if not candidates:
        return [], []

    # e) i f)
    candidates = merge_broken(candidates)
    candidates = drop_borders(candidates, labels, W, trim_border)

    chars, boxes = [], []
    for x, y, w, h, ids in sorted(candidates, key=lambda c: c[0]):
        mask = np.isin(labels[y:y + h, x:x + w], ids).astype(np.uint8) * 255
        chars.append(cv2.copyMakeBorder(mask, 2, 2, 2, 2, cv2.BORDER_CONSTANT, value=0))
        boxes.append((x, y, w, h))
    return chars, boxes


def merge_broken(candidates):
    """e) Dos trossos un sobre l'altre (mateixa columna, gairebé sense solapar en vertical i a prop) que junts tenen la
    mida d'un caràcter són un sol caràcter trencat (p. ex. una C partida)."""
    median_h = float(np.median([c[3] for c in candidates]))
    median_w = float(np.median([c[2] for c in candidates]))
    changed = True
    while changed:
        changed = False
        for i, j in [(i, j) for i in range(len(candidates)) for j in range(i + 1, len(candidates))]:
            x1, y1, w1, h1, ids1 = candidates[i]
            x2, y2, w2, h2, ids2 = candidates[j]
            overlap_x = max(0, min(x1 + w1, x2 + w2) - max(x1, x2))
            overlap_y = max(0, min(y1 + h1, y2 + h2) - max(y1, y2))
            gap_y = max(0, max(y1, y2) - min(y1 + h1, y2 + h2))
            x, y = min(x1, x2), min(y1, y2)
            w, h = max(x1 + w1, x2 + w2) - x, max(y1 + h1, y2 + h2) - y
            if (overlap_x >= 0.60 * min(w1, w2) and overlap_y <= 0.15 * min(h1, h2) and gap_y <= 0.20 * median_h
                    and 0.75 * median_h <= h <= 1.25 * median_h and w <= 1.50 * median_w):
                candidates[i] = (x, y, w, h, ids1 + ids2)
                candidates.pop(j)
                changed = True
                break
    return candidates


def drop_borders(candidates, labels, W, trim_border):
    """f) Una taca a tocar d'un costat de la placa és una vora (no un caràcter) si és molt estreta, massa alta o
    sobresurt de la línia del text. Amb trim_border, si no és estreta, es retalla a l'alçada del text i es conserva
    si el que queda té l'amplada d'un caràcter (un caràcter enganxat al marc)."""
    median_h = float(np.median([c[3] for c in candidates]))
    median_w = float(np.median([c[2] for c in candidates]))
    top = float(np.median([c[1] for c in candidates]))
    bottom = float(np.median([c[1] + c[3] for c in candidates]))
    kept = []
    for x, y, w, h, ids in candidates:
        near_side = x <= 0.04 * W or x + w >= 0.96 * W
        narrow = w < median_w / 3
        tall = h > 1.20 * median_h
        out_of_line = y < top - 0.15 * median_h or y + h > bottom + 0.15 * median_h
        if not (near_side and (narrow or tall or out_of_line)):
            kept.append((x, y, w, h, ids))
        elif trim_border and not narrow:
            t, b = int(round(top)), int(round(bottom))
            cols = np.flatnonzero(np.isin(labels[t:b, x:x + w], ids).any(0))
            if len(cols) and 0.5 * median_w <= cols[-1] - cols[0] + 1 <= 1.5 * median_w:
                kept.append((x + int(cols[0]), t, int(cols[-1] - cols[0] + 1), b - t, ids))
    return kept


# 5. Set caràcters --------------------------------------------------------------------------------------------------------

def force_seven(chars, boxes):
    """Si en sobren, fora la regió de la zona de la franja (a l'esquerra de tot, on una matrícula espanyola no té cap
    caràcter) o, si no n'hi ha, la més diferent (alçada, posició vertical i amplada respecte de la mediana).
    Si en falten, la caixa més ampla (caràcters enganxats) es parteix en trossos iguals."""
    chars, boxes = list(chars), list(boxes)
    if len(boxes) < 3:
        return chars, boxes
    while len(boxes) > 7:
        h = np.array([b[3] for b in boxes], float)
        yc = np.array([b[1] + b[3] / 2 for b in boxes])
        w = np.array([b[2] for b in boxes], float)
        odd = (np.abs(h - np.median(h)) + np.abs(yc - np.median(yc)) + 0.5 * np.abs(w - np.median(w))) / np.median(h)
        i = int(np.argmax(odd))
        if boxes[0][0] < MARGIN + STRIP_ZONE * PLATE_W:
            i = 0
        chars.pop(i)
        boxes.pop(i)
    while len(boxes) < 7:
        w = np.array([b[2] for b in boxes], float)
        i = int(np.argmax(w))
        if w[i] < 1.3 * np.median(w):
            break
        k = int(min(max(2, round(w[i] / np.median(w))), 8 - len(boxes)))
        x, y, bw, bh = boxes.pop(i)
        inner = chars.pop(i)[2:-2, 2:-2]
        for j in reversed(range(k)):
            a, b = int(j * bw / k), int((j + 1) * bw / k)
            chars.insert(i, cv2.copyMakeBorder(inner[:, a:b], 2, 2, 2, 2, cv2.BORDER_CONSTANT, value=0))
            boxes.insert(i, (x + a, y, b - a, bh))
    return chars, boxes


# 6. Llegir els caràcters -------------------------------------------------------------------------------------------------

def slant(mask):
    """Inclinació d'una màscara amb els moments (mu11 / mu02), com el deskew clàssic de MNIST."""
    m = cv2.moments(mask, binaryImage=True)
    return m['mu11'] / m['mu02'] if m['mu02'] > 1e-2 else 0.0


def deskew(chars):
    """Els 7 caràcters d'una matrícula tenen la mateixa inclinació (perspectiva que el redreçat no ha tret del tot, o
    placa en cursiva): s'estima amb la mediana dels 7 i a tots se'ls aplica la mateixa cisalla horitzontal."""
    k = float(np.clip(np.median([slant(c) for c in chars]), -0.5, 0.5))
    out = []
    for c in chars:
        h = c.shape[0]
        pad = int(abs(k) * h / 2) + 1
        c = cv2.copyMakeBorder(c, 0, 0, pad, pad, cv2.BORDER_CONSTANT, value=0)
        c = cv2.warpAffine(c, np.float32([[1, -k, k * h / 2], [0, 1, 0]]), (c.shape[1], h), flags=cv2.INTER_NEAREST)
        cols = np.flatnonzero(c.any(0))
        out.append(cv2.copyMakeBorder(c[:, cols[0]:cols[-1] + 1], 0, 0, 2, 2, cv2.BORDER_CONSTANT, value=0)
                   if len(cols) else c)
    return out


def read_characters(cnn, classes, chars):
    """Text i seguretat de la lectura. Es treu la inclinació; la CNN rep cada màscara a 32 x 64 (valors 0-1) i dona una
    log-probabilitat per classe. Els 4 primers només poden ser dígits i els 3 últims, lletres de LETTERS.
    Seguretat = la diferència més petita, entre els 7 caràcters, entre la millor classe i la segona."""
    chars = deskew(chars)
    x = np.stack([cv2.resize(c, (32, 64), interpolation=cv2.INTER_AREA) for c in chars]).astype(np.float32) / 255
    with torch.no_grad():
        logp = torch.log_softmax(cnn(torch.from_numpy(x)[:, None]), 1).numpy()
    text, confidence = '', np.inf
    for i, row in enumerate(logp):
        allowed = np.char.isdigit(classes) if i < 4 else np.isin(classes, list(LETTERS))
        scores = np.where(allowed, row, -np.inf)
        first, second = np.argsort(scores)[::-1][:2]
        text += classes[first]
        confidence = min(confidence, float(scores[first] - scores[second]))
    return text, confidence
