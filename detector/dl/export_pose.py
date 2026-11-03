"""
Exporta un pose d'ultralytics (.pt) a TorchScript (el format que llegeix model.py, sense ultralytics).
Executar des de l'arrel:
  python -m detector.dl.export_pose --pt detector/dl/weights/pose_best.pt --out detector/dl/weights/pose.torchscript
  python -m detector.dl.export_pose --pt detector/dl/weights/pose_moto_best.pt --out detector/dl/weights/pose_moto.torchscript
"""
import argparse
import os
import shutil
import tempfile

import torch
from ultralytics import YOLO

ap = argparse.ArgumentParser()
ap.add_argument('--pt', required=True)
ap.add_argument('--out', required=True)
ap.add_argument('--imgsz', type=int, default=640)
args = ap.parse_args()
tmp = os.path.join(tempfile.mkdtemp(), 'model.pt')
shutil.copy(args.pt, tmp)
shutil.move(YOLO(tmp).export(format='torchscript', imgsz=args.imgsz, device='cpu'), args.out)
y = torch.jit.load(args.out, map_location='cpu')(torch.zeros(1, 3, args.imgsz, args.imgsz))
print(f'{args.pt} -> {args.out}, sortida {tuple(y.shape)}')   # (1, 17, 8400) a 640
