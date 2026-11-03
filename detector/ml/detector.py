"""detectPlates() dels profes (LicensePlateDetector.py dels professors) amb els valors com a paràmetres.

Amb DEFAULT dona exactament el mateix que la funció dels profes (ho comproven els scripts de S2 Ex3).
  blobs()  -> contorns de les taques, abans dels filtres
  checks() -> quins filtres passa un contorn
  detect() -> regions acceptades (com detectPlates)
  iou()    -> IoU de dos quadrilàters convexos
La imatge s'ha de passar ja reduïda a 640 px d'amplada; imWidth/imHeight són la mida ORIGINAL,
perquè el codi dels profes compara la vora amb la mida original.
"""
import cv2
import numpy as np

# valors dels profes
DEFAULT = dict(blackhat_k=3, rect_w=15, thr=0.40, area_min=3400, area_max=8000, ar_min=2.5, ar_max=7,
               w_min=100, w_max=250, h_min=30, h_max=60,
               blackhat_it=3, close_it=2, open_it=4, dilate_it=2)


def blobs(image, P):
    """Taques candidates (contorns) abans dels filtres. 0 iteracions = es salta el pas."""
    rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (P['rect_w'], 5))
    squareKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    bhKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (P['blackhat_k'], P['blackhat_k']))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, bhKernel, iterations=P['blackhat_it'])
    gradX = np.absolute(cv2.Sobel(blackhat, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1))
    (minVal, maxVal) = (np.min(gradX), np.max(gradX))
    if maxVal == minVal:
        return []  # imatge plana (p. ex. blackhat 1x1): res a detectar
    gradX = (255 * ((gradX - minVal) / (maxVal - minVal))).astype('uint8')
    gradX = cv2.GaussianBlur(gradX, (7, 7), 0)
    if P['close_it'] > 0:
        gradX = cv2.morphologyEx(gradX, cv2.MORPH_CLOSE, rectKernel, iterations=P['close_it'])
    thresh = cv2.threshold(gradX, P['thr'] * np.max(gradX), 255, cv2.THRESH_BINARY)[1]
    if P['open_it'] > 0:
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, squareKernel, iterations=P['open_it'])
    if P['dilate_it'] > 0:
        thresh = cv2.dilate(thresh, rectKernel, iterations=P['dilate_it'])
    (cnts, _) = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return cnts


def checks(c, imWidth, imHeight, P):
    """Filtres dels profes per a un contorn: {nom: True si el passa}."""
    (x, y, w, h) = cv2.boundingRect(c)
    area = cv2.contourArea(c)
    return {'vora': x != 0 and y != 0 and x + w != imWidth and y + h != imHeight,
            'proporcio': P['ar_min'] < w / h < P['ar_max'],
            'amplada': P['w_min'] < w <= P['w_max'],
            'alcada': P['h_min'] < h <= P['h_max'],
            'area': P['area_min'] < area < P['area_max']}


def detect(image, imWidth, imHeight, P):
    """Regions acceptades, com detectPlates: un rectangle girat (4x2) per regió."""
    return [cv2.boxPoints(cv2.minAreaRect(c)) for c in blobs(image, P)
            if all(checks(c, imWidth, imHeight, P).values())]


def iou(a, b):
    """IoU de dos quadrilàters convexos (4x2)."""
    a, b = np.float32(a), np.float32(b)
    inter = cv2.intersectConvexConvex(a, b)[0]
    union = cv2.contourArea(a) + cv2.contourArea(b) - inter
    return inter / union if union > 0 else 0


def detectPlates(image, P=DEFAULT):
    """Mateixa interfície que detectPlates() dels profes: foto original -> regions (4x2) en px de la foto reduïda a
    640 d'amplada (si és més ampla), com fa la seva funció."""
    import imutils
    H, W = image.shape[:2]
    small = imutils.resize(image, width=640) if W > 640 else image
    return detect(small, W, H, P)
