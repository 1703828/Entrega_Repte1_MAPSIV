"""Geometria de les cantonades de la matrícula.

Simplificat de alpr/geometry.py del repo vell. Només hi ha el que fem servir:
  order_corners    -> posa 4 punts en ordre TL, TR, BR, BL
  simplify_to_quad -> redueix els polígons d'UC3M de 5-6 punts a 4
"""
import cv2
import numpy as np


def order_corners(pts):
    """Ordena 4 punts com a TL, TR, BR, BL (dalt-esq, dalt-dreta, baix-dreta, baix-esq).

    Funciona encara que la matrícula estigui girada:
      1. ordena els punts en cercle al voltant del centre -> sabem quins són veïns;
      2. dels dos parells de costats oposats, el més horitzontal és dalt/baix;
      3. el costat amb la y més petita és el de dalt; dins de cada costat, la x més petita és l'esquerra.
    """
    pts = np.asarray(pts, dtype=np.float32).reshape(4, 2)
    c = pts.mean(axis=0)
    pts = pts[np.argsort(np.arctan2(pts[:, 1] - c[1], pts[:, 0] - c[0]))]  # ordre circular

    def horizontality(i):
        # quant d'horitzontal és el costat i -> i+1 (1 = horitzontal, 0 = vertical)
        d = pts[(i + 1) % 4] - pts[i]
        return abs(d[0]) / (np.hypot(*d) + 1e-9)

    # costats 0 i 2 són oposats, igual que 1 i 3
    i = 0 if horizontality(0) + horizontality(2) >= horizontality(1) + horizontality(3) else 1
    side_a = pts[[i, (i + 1) % 4]]
    side_b = pts[[(i + 2) % 4, (i + 3) % 4]]
    top, bottom = (side_a, side_b) if side_a[:, 1].mean() < side_b[:, 1].mean() else (side_b, side_a)

    tl, tr = sorted(top, key=lambda p: p[0])
    bl, br = sorted(bottom, key=lambda p: p[0])
    return np.array([tl, tr, br, bl], dtype=np.float32)


def simplify_to_quad(pts):
    """Alguns polígons d'UC3M tenen 5-6 punts (punts extra sobre un costat).
    Els redueix als 4 vèrtexs reals. Retorna None si no se'n surt."""
    pts = np.asarray(pts, dtype=np.float32)
    if len(pts) == 4:
        return pts
    hull = cv2.convexHull(pts)
    perimeter = cv2.arcLength(hull, True)
    for frac in (0.01, 0.02, 0.03, 0.05, 0.08, 0.12, 0.18):  # cada cop més tolerant
        approx = cv2.approxPolyDP(hull, frac * perimeter, True)
        if len(approx) == 4:
            return approx.reshape(4, 2).astype(np.float32)
    return None


def plate_area_angle(corners, width):
    """Àrea (px²) i angle (graus) de la matrícula anotada, com a l'Ex1 dels profes.

    Es reescala a 640 px d'amplada (com fa detectPlates) perquè l'àrea sigui comparable
    entre fotos de mides diferents. L'angle es dona respecte del costat llarg.
    """
    scale = 640 / width if width > 640 else 1
    rect = cv2.minAreaRect(np.asarray(corners, dtype=np.float32) * scale)
    w, h = rect[1]
    angle = rect[2] - 90 if w < h else rect[2]
    return w * h, angle
