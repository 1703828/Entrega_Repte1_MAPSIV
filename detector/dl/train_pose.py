"""
Entrena un YOLO-pose de 4 punts (les cantonades de la matrícula). Es fa servir a la MÀQUINA VIRTUAL (GPU, --device 0)
dins de tmux; també funciona al Mac amb --device mps.
Dades: el dataset de prepare_pose.py (+ prepare_moto_uc3m.py per a --extra). Validation: sempre images/val.
Pesos de partida: yolov8n-pose preentrenat amb COCO (detector/dl/weights/yolov8n-pose_coco.pt).

Els dos poses del sistema final (taula 2 de l'informe):
  pose (cotxes):  python train_pose.py --name pose --train train uc3m --extra ../moto_uc3m/images/train \
                      --epochs 80 --batch 8 --degrees 10 --shear 2 --perspective 0.0003
  pose de motos:  python train_pose.py --name pose_moto --train train uc3m --extra ../moto_uc3m/images/train \
                      --epochs 60 --batch 8 --degrees 25 --shear 5 --perspective 0.0005
Experiments A, B i C de l'informe (taula 3; 50 epochs, sense augmentació geomètrica):
  A: --name A --train train    B: --name B --train train uc3m
  C: --name C --train train uc3m --model yolov8n-pose.yaml --load LP-detection.pt  (inici des de zero + capes de LP)
Sortida: runs/<name>/weights/best.pt (millor epoch a validation) i results.csv. Després: export_pose.py.
Executar (MV, des de detector/dl/datasets):  python ../train_pose.py --data pose --name ... [opcions]
"""
import argparse
import os
import time

ap = argparse.ArgumentParser()
ap.add_argument('--data', default='pose', help='carpeta del dataset (prepare_pose.py)')
ap.add_argument('--model', default='../weights/yolov8n-pose_coco.pt')
ap.add_argument('--load', default=None, help='pesos de partida per a un .yaml (experiment C)')
ap.add_argument('--train', nargs='+', default=['train', 'uc3m'], help='carpetes de images/ per entrenar')
ap.add_argument('--extra', nargs='*', default=[], help='carpetes d\'imatges addicionals (p. ex. ../moto_uc3m/images/train)')
ap.add_argument('--name', required=True)
ap.add_argument('--epochs', type=int, default=50)
ap.add_argument('--imgsz', type=int, default=640)
ap.add_argument('--batch', type=int, default=16)
ap.add_argument('--degrees', type=float, default=0.0)
ap.add_argument('--shear', type=float, default=0.0)
ap.add_argument('--perspective', type=float, default=0.0)
ap.add_argument('--device', default='0', help="'0' (GPU de la MV) o 'mps' (Mac)")
args = ap.parse_args()

data_dir = os.path.abspath(args.data)
yaml_path = os.path.join(data_dir, f'data_{args.name}.yaml')
train = '[' + ', '.join([f'images/{t}' for t in args.train] + [os.path.abspath(e) for e in args.extra]) + ']'
with open(yaml_path, 'w') as f:
    f.write(f'path: {data_dir}\ntrain: {train}\nval: images/val\ntest: images/test\n'
            'kpt_shape: [4, 3]\nflip_idx: [1, 0, 3, 2]\nnames:\n  0: plate\n')   # gir: TL <-> TR, BR <-> BL

from ultralytics import YOLO  # noqa: E402

t0 = time.time()
model = YOLO(args.model)
if args.load:
    model = model.load(args.load)
device = int(args.device) if args.device.isdigit() else args.device
model.train(data=yaml_path, epochs=args.epochs, imgsz=args.imgsz, batch=args.batch, workers=4, device=device, seed=0,
            degrees=args.degrees, shear=args.shear, perspective=args.perspective,
            project=os.path.abspath('runs'), name=args.name, exist_ok=True)
print(f'TRAIN DONE {args.name}: {(time.time() - t0) / 60:.1f} min -> runs/{args.name}/weights/best.pt')
