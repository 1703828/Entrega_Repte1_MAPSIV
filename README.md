# Challenge 1: Car Plate Recognition (Grup 7)

Lectura de matrícules espanyoles de 4 dígits i 3 lletres: foto → `"1234BCD"`.

El sistema és el de l'informe (`Challenge1_Group_7_Report_2026.pdf`):
- **Localització:** YOLO-pose amb 4 cantonades, en dues passades, i un segon pose per a les motos.
- **Rectificació:** homografia a 400 × 85, amb dues variants de cantonades.
- **Segmentació:** black-hat + Otsu, amb 4 opcions i la regla dels 7 caràcters (primer la franja blava).
- **Reconeixement:** deskew i una CNN petita, amb el format de 4 dígits + 3 lletres.
- **Tria:** la lectura que dona 7 regions sense forçar i és més segura.

## Estructura

```
challenge_1/
├── model.py                  sistema final (classe Model, com a Codabench)
├── run_local.py              script de Codabench per provar una submission en local
├── evaluate.py               avaluació per split i per font
├── submission/submission.zip submission final de Codabench (model.py + weights/)
├── detector/
│   ├── ml/                   detector clàssic dels professors, adaptat perquè torni 4 punts
│   └── dl/                   YOLO-pose: preparar dades, entrenar (MV), exportar
│       └── weights/          pesos del detector final
├── recognizer/               caràcters, entrenament de la CNN i de l'SVM, validació creuada
│   └── weights/              cnn/ (sistema final i submission), svm/ (reconeixedor anterior)
├── ground_truth_maker/       eina d'anotació del ground truth
├── splits/                   partició per grups de vehicle, duplicats i export per a run_local
├── annotations/              el nostre ground truth
├── dades/                    cv/, own/, own_2/ (fotos del corpus) i uc3m/
└── common/                   lectura de les anotacions, geometria i mètriques
```

## Instal·lació

```bash
conda create -n challenge1 python=3.11 -y && conda activate challenge1
pip install -r requirements.txt
```

`model.py` només necessita `torch`, `opencv-python` i `numpy`. `ultralytics` només cal per entrenar o exportar el
YOLO-pose.

**Totes les comandes s'executen des de l'arrel `challenge_1/`.**

## Executar el pipeline

```python
import model
m = model.Model(detector_dir='detector/dl/weights', recognizer_dir='recognizer/weights/cnn')
m.predict('dades/own/im_2a392ebdc323.jpg')          # -> '4725DGT'
```

Amb el script de Codabench (`run_local.py`), sobre un split exportat en el seu format:

```bash
python -m splits.export_split --split validation        # -> outputs/codabench_validation/{images, labels.csv}
python -B run_local.py --submission submission/submission.zip --data outputs/codabench_validation
```

`predict()` sempre torna 7 caràcters i no peta mai: si no es pot llegir res, torna `0000BBB`.

## Avaluar

```bash
python -B evaluate.py          # -> encert per split i per font, i outputs/evaluate.csv foto a foto
```

## Entrenar

### 1. Particions (`splits/`)

```bash
python -m splits.make_splits            # -> annotations/splits.csv (train / validation / test / limitation) + splits/splits_resum.txt
python -m splits.check_duplicates       # fotos repetides o del mateix vehicle en splits diferents
```

- **Repartiment:** per grups (`split_group`: quasi-duplicats, mateix vehicle i mateixa captura), estratificat per font
  i vista, aproximadament 70 / 15 / 15, amb llavor 0.
- **UC3M:** manté la seva partició, i se'n tria una mostra de 53 fotos per a la detecció (`det_uc3m`).

### 2. Detector (`detector/`)

**Detector clàssic dels professors** (`detector/ml/detector.py`). És el seu `detectPlates()` amb els paràmetres
oberts, i torna cada regió com un rectangle girat de 4 punts:

```bash
python -m detector.ml.eval_detection --params profes --splits train validation
```

**YOLO-pose** (`detector/dl/`). Es va entrenar a la màquina virtual (GPU) dins de `tmux`:

```bash
python -m detector.dl.prepare_pose          # -> detector/dl/datasets/pose/ i pose.tar (per pujar a la MV)
python -m detector.dl.prepare_moto_uc3m     # -> detector/dl/datasets/moto_uc3m/ (41 fotos de moto d'UC3M)
# a la MV, des de detector/dl/datasets/:
python ../train_pose.py --name pose --train train uc3m --extra ../moto_uc3m/images/train \
    --epochs 80 --batch 8 --degrees 10 --shear 2 --perspective 0.0003
python ../train_pose.py --name pose_moto --train train uc3m --extra ../moto_uc3m/images/train \
    --epochs 60 --batch 8 --degrees 25 --shear 5 --perspective 0.0005
# de tornada, a l'arrel:
python -m detector.dl.export_pose --pt <runs/pose/weights/best.pt> --out detector/dl/weights/pose.torchscript
python -m detector.dl.export_pose --pt <runs/pose_moto/weights/best.pt> --out detector/dl/weights/pose_moto.torchscript
```

A `train_pose.py` també hi ha les comandes dels experiments A, B i C de la taula 3.

Pesos inclosos a `detector/dl/weights/`:

| Fitxer | Què és |
|---|---|
| `pose.torchscript`, `pose_best.pt` | pose de cotxes |
| `pose_moto.torchscript`, `pose_moto_best.pt` | pose de motos |
| `yolov8n-pose_coco.pt` | pesos preentrenats de COCO de partida |

### 3. Reconeixedor (`recognizer/`)

```bash
python -B -m recognizer.chars_corpus     # caràcters del corpus segmentats pel pipeline -> recognizer/cache/segments.pkl
python -B -m recognizer.chars_uc3m       # 5.158 caràcters d'UC3M -> recognizer/cache/uc3m_chars.pkl
python -B -m recognizer.train_cnn        # CNN del sistema final -> recognizer/weights/cnn
python -B -m recognizer.train_svm --out recognizer/weights/svm/recognizer.npz
# taula 4 (validació creuada de 5 parts per vehicle, 231 fotos de cotxe)
python -B -m recognizer.cross_validation --rec svm knn --tag base
python -B -m recognizer.cross_validation --rec svm svm_lbp cnn cnnk --deskew --uc3m --epochs 30 --tag d_u
```

- **CNN:** s'entrena amb tots els caràcters de `segments.pkl` i els d'UC3M. Fa servir la GPU si n'hi ha, i triga uns
  5 minuts.
- **CNN de la submission:** els resultats de l'informe són amb els models entrenats només amb train. Un cop triat el
  sistema, per a la submission la CNN es va reentrenar amb tot el corpus (train, validació, test i les 7 motos) i UC3M.
  Per això, amb `evaluate.py`, la validació i el test ja no són dades no vistes per aquesta CNN.
- **Reproduïbilitat:** si es tornen a entrenar, s'obtenen pesos equivalents, però no idèntics bit a bit (la
  inicialització i l'ordre a la GPU varien). Els pesos de la submission són els de `recognizer/weights/cnn`.
- **SVM:** és el reconeixedor anterior (taules 4 i 6). `model.py` fa servir la CNN.

## Dades (`dades/`) i anotacions (`annotations/`)

| Carpeta | Fotos | Origen |
|---|---:|---|
| `dades/cv` | 69 | campus virtual |
| `dades/own` | 45 | captures pròpies |
| `dades/own_2` | 121 | segon conjunt de captures pròpies |
| `dades/uc3m` | 1.975 | UC3M-LP (https://doi.org/10.21950/OS5W4Z), amb les seves anotacions JSON |

`dades/uc3m` és un enllaç simbòlic al dataset descarregat (4,3 GB). Si es mou el repositori, cal tornar a descarregar
UC3M-LP i deixar-lo a `dades/uc3m/images/`.

| Fitxer | Contingut |
|---|---|
| `images.csv` | una fila per foto: ruta, font (`subset`), vista, grups de vehicle i captura |
| `plates.csv` | una fila per matrícula: text, estat del text, 4 cantonades (x1, y1 … x4, y4), qualitat de les cantonades, si és l'objectiu |
| `splits.csv` | partició de l'informe (train / validation / test / limitation; `det_uc3m`) |

Les anotacions s'editen amb l'eina:

```bash
python -m ground_truth_maker.annotate --list                       # incidències pendents (terminal)
python -m ground_truth_maker.annotate --annotator <nom>            # revisar-les a la finestra
python -m ground_truth_maker.annotate --annotator <nom> --plate im_xxx_p0
```

Abans de cada canvi, desa una còpia de seguretat a `outputs/backups/`.

## Notes

- **Ordre de les cantonades:** sempre superior esquerra, superior dreta, inferior dreta, inferior esquerra.
- **`model.py` respecte del ZIP:** és el de la submission amb un sol canvi. A més de `Model('weights')` (Codabench), accepta
  `detector_dir` i `recognizer_dir` per llegir els pesos de les carpetes del repositori. Les prediccions són les
  mateixes.
- **Fitxers generats:** van a `outputs/` (avaluacions, datasets exportats, còpies de seguretat).
