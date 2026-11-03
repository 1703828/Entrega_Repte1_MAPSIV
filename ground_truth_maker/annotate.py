"""Eina d'anotació: consultar i corregir directament les anotacions definitives (annotations/*.csv).

Ús (des de l'arrel):
    python -m ground_truth_maker.annotate --list                          incidències obertes de prioritat alta (terminal)
    python -m ground_truth_maker.annotate --annotator alicia              revisar-les a la finestra
    python -m ground_truth_maker.annotate --annotator alicia --kind text_unverified --priority baixa
    python -m ground_truth_maker.annotate --annotator alicia --plate im_xxx_p0     (o --image, --subset, --split, --quality)
Sense cap filtre es mostren les incidències obertes de prioritat alta (annotations/issues.csv, es crea quan cal). Amb --plate, --image,
--subset, --split o --quality es mostren aquelles matrícules, tinguin incidència o no.

Cada canvi es desa al moment a plates.csv / images.csv / issues.csv: primer una còpia del fitxer a outputs/backups/
(un cop per sessió) i després es substitueix d'una sola vegada (common/manifest.save). Els splits NO es recalculen
mai: si un text nou coincideix amb el d'un altre grup de partició, s'afegeix una incidència 'vehicle_conflict'.

Finestra: la foto amb totes les matrícules (verd; la seleccionada en vermell, TL en lila) i la seleccionada redreçada.
  n / b   matrícula següent / anterior                    Espai   foto següent      q   sortir
  k       text correcte                                   t       escriure el text (terminal; '-' = il·legible)
  o       cantonades correctes (corner_quality = real)    c       tornar a marcar les 4 cantonades (clics + Enter)
  a       afegir una matrícula (4 clics + Enter + text)   x       esborrar la matrícula (confirmació)
  g       és la matrícula objectiu de la foto             v       vista de la foto (frontal / lateral / unknown)
  e       resoldre les incidències d'aquesta matrícula amb una nota (terminal)
  (mode cantonades: u desfer clic, r esborrar clics, Esc cancel·lar)
"""
import argparse
import os
import re

import cv2
import numpy as np
import pandas as pd

from common import manifest
from common.geometry import order_corners


def warp_margin(gray, corners, m, width=400, plate_h=85):
    """4 cantonades -> matrícula de 400 x 85 amb una vora de m * 85 px a cada costat, a 400 px d'amplada."""
    p = round(m * plate_h)
    dst = np.float32([[p, p], [p + width - 1, p], [p + width - 1, p + plate_h - 1], [p, p + plate_h - 1]])
    M = cv2.getPerspectiveTransform(order_corners(np.asarray(corners, 'float32')), dst)
    out = cv2.warpPerspective(gray, M, (width + 2 * p, plate_h + 2 * p), borderMode=cv2.BORDER_REPLICATE)
    return cv2.resize(out, (width, int(out.shape[0] * width / out.shape[1])), interpolation=cv2.INTER_AREA)

PLATE_RE = re.compile(r"^\d{4}[BCDFGHJKLMNPRSTVWXYZ]{3}$")  # 4 dígits + 3 consonants
MAX_W, MAX_H, PLATE_W = 1300, 700, 520


def text_status(t):
    return "illegible" if not t else ("full" if PLATE_RE.match(t) else "other_format")


class Store:
    """Les tres taules en memòria; cada canvi es desa tot seguit (amb una còpia de seguretat la primera vegada)."""

    def __init__(self, author):
        self.author = author
        self.plates, self.images = manifest.plates(), manifest.images()
        # incidències obertes (annotations/issues.csv); si el fitxer no hi és, es comença sense cap
        cols = ["kind", "priority", "image_id", "plate_id", "detail", "status", "note"]
        path = "annotations/issues.csv"
        self.issues = (pd.read_csv(path, dtype=str, keep_default_na=False) if os.path.exists(path)
                       else pd.DataFrame(columns=cols))
        self.backed = set()

    def save(self, name):
        df = {"plates.csv": self.plates, "images.csv": self.images, "issues.csv": self.issues}[name]
        manifest.save(df, name, backup=name not in self.backed)
        self.backed.add(name)

    def set_plate(self, pid, tag=None, **values):
        i = self.plates.index[self.plates.plate_id == pid][0]
        for k, v in values.items():
            self.plates.loc[i, k] = v
        if tag:
            cur = [t for t in self.plates.loc[i, "review_status"].split("+") if t and t not in ("unreviewed", tag)]
            self.plates.loc[i, "review_status"] = "+".join(cur + [tag])
        self.plates.loc[i, "annotator"] = self.author
        self.save("plates.csv")

    def resolve(self, pid, kinds, note):
        m = (self.issues.plate_id == pid) & (self.issues.status == "open")
        if kinds:
            m &= self.issues.kind.isin(kinds)
        if m.any():
            self.issues.loc[m, ["status", "note"]] = ["resolved", f"{self.author}: {note}"]
            self.save("issues.csv")

    def add_issue(self, kind, image_id, plate_id, detail):
        row = dict(kind=kind, priority="alta", image_id=image_id, plate_id=plate_id, detail=detail, status="open", note="")
        self.issues = pd.concat([self.issues, pd.DataFrame([row])], ignore_index=True)
        self.save("issues.csv")
        print(f"  ATENCIÓ: {detail}")

    def check_vehicle(self, pid, text):
        """Un text nou que ja és a una foto d'un altre grup de partició: possible mateix vehicle a splits diferents."""
        if text_status(text) != "full":
            return
        im = self.images.set_index("image_id").split_group
        me = self.plates[self.plates.plate_id == pid].iloc[0]
        others = self.plates[(self.plates.text == text) & (self.plates.plate_id != pid) & (self.plates.subset != "uc3m")]
        others = others[others.image_id.map(im) != im[me.image_id]]
        if len(others):
            self.add_issue("vehicle_conflict", ";".join([me.image_id] + list(others.image_id)), pid,
                           f"text {text} també a {', '.join(others.image_id)} (un altre grup de partició): decidir si és el "
                           "mateix vehicle; els splits no s'han canviat")


def select(args, store):
    """Matrícules a revisar: amb filtres explícits, aquelles matrícules; si no, les de les incidències obertes."""
    p = store.plates.merge(store.images[["image_id", "path", "view", "split_group"]], on="image_id")
    p["split"] = p.image_id.map(manifest.splits())
    explicit = any([args.plate, args.image, args.subset, args.split, args.quality])
    if not explicit or args.kind or args.priority:
        iss = store.issues[store.issues.status == "open"]
        iss = iss[iss.priority.isin(args.priority or ["alta"])]
        if args.kind:
            iss = iss[iss.kind.isin(args.kind)]
        ids = set(iss.plate_id) - {""}
        img_only = {i for x in iss[iss.plate_id == ""].image_id for i in x.split(";")}
        p = p[p.plate_id.isin(ids) | p.image_id.isin(img_only)]
    for col, val in (("plate_id", args.plate), ("image_id", args.image), ("subset", args.subset),
                     ("split", args.split), ("corner_quality", args.quality)):
        if val:
            p = p[p[col].isin(val)]
    return p


def render(img, plates, sel, clicks, header):
    """Foto amb les matrícules + la seleccionada redreçada a sota + línies d'informació."""
    s = min(MAX_W / img.shape[1], MAX_H / img.shape[0], 1.0)
    view = cv2.resize(img, None, fx=s, fy=s)
    for k, p in enumerate(plates.itertuples()):
        q = (manifest.corners(p._asdict()) * s).astype(np.int32)
        cv2.polylines(view, [q], True, (0, 0, 255) if k == sel else (0, 200, 0), 2)
        cv2.circle(view, tuple(q[0]), 5, (255, 0, 255), -1)
        cv2.putText(view, p.text or "?", tuple(q[3] + [0, 20]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 220, 255), 2)
    for i, (x, y) in enumerate(clicks):
        cv2.circle(view, (int(x * s), int(y * s)), 5, (0, 255, 255), -1)
        cv2.putText(view, str(i + 1), (int(x * s) + 6, int(y * s) - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    p = plates.iloc[sel]
    plate = warp_margin(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), manifest.corners(p), 0.15)
    plate = cv2.cvtColor(cv2.resize(plate, (PLATE_W, int(plate.shape[0] * PLATE_W / plate.shape[1]))), cv2.COLOR_GRAY2BGR)
    lines = [header,
             f"{p.plate_id}  text: {p.text or '-'} ({p.text_status}, {p.text_source}; nom: {p.name_text or '-'})  "
             f"objectiu: {p.is_target or '-'}",
             f"geometria: {p.geometry_type} / {p.corner_quality}   revisió: {p.review_status}   vista: {p.view}   split: {p.split}",
             "n/b mat.  Espai foto  k text ok  t text  o cant. ok  c cant.  a afegir  x esborrar  g objectiu  v vista  e resoldre  q"]
    W = max(view.shape[1], PLATE_W)
    canvas = np.zeros((view.shape[0] + plate.shape[0] + 28 * len(lines) + 10, W, 3), np.uint8)
    canvas[:view.shape[0], :view.shape[1]] = view
    canvas[view.shape[0]:view.shape[0] + plate.shape[0], :PLATE_W] = plate
    y0 = view.shape[0] + plate.shape[0]
    for i, line in enumerate(lines):
        cv2.putText(canvas, line, (8, y0 + 22 + 28 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (230, 230, 230), 1, cv2.LINE_AA)
    return canvas, s


def ask_text():
    t = input("  Text de la matrícula ('-' = il·legible): ").strip().upper()
    if t == "-":
        return ""
    t = re.sub(r"[^A-Z0-9]", "", t)
    if not PLATE_RE.match(t):
        print(f"  Atenció: '{t}' no té el format 1234BCD (es desa com a other_format).")
    return t


def geometry(quad):
    """Àrea i proporció amplada/alçada (costat de dalt / costat esquerre), com al manifest."""
    q = np.float32(quad)
    w, h = np.linalg.norm(q[1] - q[0]), np.linalg.norm(q[3] - q[0])
    return str(round(float(cv2.contourArea(q)))), str(round(w / h, 2)) if h > 0 else ""


def review(store, sel_plates):
    window = "annotate"
    cv2.namedWindow(window)
    state = {"clicks": [], "scale": 1.0}

    def on_mouse(event, x, y, _flags, _param):
        if event == cv2.EVENT_LBUTTONDOWN and len(state["clicks"]) < 4:
            state["clicks"].append((x / state["scale"], y / state["scale"]))

    cv2.setMouseCallback(window, on_mouse)
    image_ids = list(dict.fromkeys(sel_plates.image_id))
    for n, iid in enumerate(image_ids):
        focus = set(sel_plates[sel_plates.image_id == iid].plate_id)
        path = store.images.set_index("image_id").loc[iid, "path"]
        img, sel, mode = cv2.imread(path), None, None
        while True:
            plates = store.plates[store.plates.image_id == iid].reset_index(drop=True)
            if plates.empty:
                break
            plates["view"] = store.images.set_index("image_id").loc[iid, "view"]
            plates["split"] = manifest.splits()[iid]
            if sel is None or sel >= len(plates):
                hit = plates.index[plates.plate_id.isin(focus)]
                sel = int(hit[0]) if len(hit) else 0
            header = f"[{n + 1}/{len(image_ids)}] {iid}" + (f"   MODE {mode}: 4 clics + Enter (Esc cancel·la)" if mode else "")
            canvas, state["scale"] = render(img, plates, sel, state["clicks"], header)
            cv2.imshow(window, canvas)
            key = cv2.waitKey(30) & 0xFF
            p = plates.iloc[sel]
            if key == ord("q"):
                cv2.destroyAllWindows()
                return
            if mode:
                if key == 27:
                    mode, state["clicks"] = None, []
                elif key == ord("u") and state["clicks"]:
                    state["clicks"].pop()
                elif key == ord("r"):
                    state["clicks"] = []
                elif key in (13, 10) and len(state["clicks"]) == 4:
                    quad = order_corners(state["clicks"]).round(1)
                    xy = {f"{c}{i + 1}": float(quad[i][k]) for i in range(4) for k, c in enumerate("xy")}
                    area, aspect = geometry(quad)
                    if mode == "corners":
                        store.set_plate(p.plate_id, tag="corners_corrected", geometry_type="quad_manual",
                                        corner_quality="real", area_px=area, aspect=aspect, **xy)
                        store.resolve(p.plate_id, ["no_corners"], "cantonades marcades")
                    else:
                        text = ask_text()
                        k = 1
                        while f"{iid}_a{k}" in set(store.plates.plate_id):
                            k += 1
                        row = {c: "" for c in store.plates.columns}
                        row.update(plate_id=f"{iid}_a{k}", image_id=iid, subset=p.subset, annotation="annotate.py",
                                   text_raw=text, text=text, text_status=text_status(text), text_source="manual",
                                   geometry_type="quad_manual", corner_quality="real", is_target="no", area_px=area,
                                   aspect=aspect, review_status="added", annotator=store.author, **xy)
                        store.plates = pd.concat([store.plates, pd.DataFrame([row])], ignore_index=True)
                        store.save("plates.csv")
                        store.check_vehicle(row["plate_id"], text)
                    mode, state["clicks"] = None, []
                continue
            if key == ord("n"):
                sel = (sel + 1) % len(plates)
            elif key == ord("b"):
                sel = (sel - 1) % len(plates)
            elif key == ord(" "):
                break
            elif key == ord("k"):
                store.set_plate(p.plate_id, tag="text_ok")
                store.resolve(p.plate_id, ["text_unverified", "text_contradiction"], f"text {p.text} correcte")
            elif key == ord("t"):
                text = ask_text()
                store.set_plate(p.plate_id, tag="text_corrected", text=text, text_source="manual",
                                text_status=text_status(text))
                store.resolve(p.plate_id, ["text_unverified", "text_contradiction"], f"text {p.text} -> {text}")
                store.check_vehicle(p.plate_id, text)
            elif key == ord("o"):
                store.set_plate(p.plate_id, tag="corners_ok", corner_quality="real")
            elif key == ord("c"):
                mode, state["clicks"] = "corners", []
            elif key == ord("a"):
                mode, state["clicks"] = "add", []
            elif key == ord("x"):
                if input(f"  Esborrar {p.plate_id} ({p.text})? (s/n): ").strip().lower() == "s":
                    store.plates = store.plates[store.plates.plate_id != p.plate_id].reset_index(drop=True)
                    store.save("plates.csv")
                    store.resolve(p.plate_id, [], "matrícula esborrada")
                    sel = 0
            elif key == ord("g"):
                for other in plates.itertuples():
                    store.set_plate(other.plate_id, is_target="yes" if other.plate_id == p.plate_id else "no")
            elif key == ord("v"):
                v = input("  Vista (frontal/lateral/unknown): ").strip().lower()
                if v in ("frontal", "lateral", "unknown"):
                    store.images.loc[store.images.image_id == iid, "view"] = v
                    store.save("images.csv")
            elif key == ord("e"):
                store.resolve(p.plate_id, [], input("  Nota (decisió presa): ").strip())
    cv2.destroyAllWindows()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--annotator")
    ap.add_argument("--list", action="store_true", help="només consultar al terminal")
    ap.add_argument("--kind", nargs="+", help="tipus d'incidència")
    ap.add_argument("--priority", nargs="+", choices=["alta", "baixa"])
    for a in ("--plate", "--image", "--subset", "--split", "--quality"):
        ap.add_argument(a, nargs="+")
    args = ap.parse_args()
    store = Store(args.annotator or "")
    sel = select(args, store)
    print(f"{len(sel)} matrícules en {sel.image_id.nunique()} fotos")
    if sel.empty:
        print("Cap matrícula per revisar amb aquest filtre (sense filtre: incidències obertes de prioritat alta).")
        return
    if args.list:
        iss = store.issues[store.issues.status == "open"].groupby("plate_id").kind.agg(",".join)
        out = sel.assign(issue=sel.plate_id.map(iss).fillna(""))
        with pd.option_context("display.max_rows", 500, "display.width", 250, "display.max_colwidth", 50):
            print(out[["plate_id", "subset", "split", "text", "text_status", "text_source", "name_text",
                       "geometry_type", "corner_quality", "is_target", "issue"]].to_string(index=False))
        return
    if not args.annotator:
        raise SystemExit("Cal --annotator per corregir (o --list per només consultar).")
    review(store, sel)
    print("Fet. Els canvis ja són desats (còpies anteriors a outputs/backups/).")


if __name__ == "__main__":
    main()
