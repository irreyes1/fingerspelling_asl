Fingerspelling ASL

Proyecto para entrenamiento de reconocimiento de fingerspelling con CTC.

## 1) Clonar
```bash
git clone https://github.com/irreyes1/fingerspelling_asl.git
cd fingerspelling_asl
```

## 2) Crear entorno
### Windows (PowerShell)
```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

### Linux
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 3) Datos (no incluidos en Git)
Este repo no sube parquets ni checkpoints pesados.

Estructura esperada:
```text
data/
  asl-fingerspelling/
    train.csv
    supplemental_metadata.csv
    character_to_prediction_index.json
    train_landmarks/*.parquet
    supplemental_landmarks/*.parquet
```

## 4) Entrenamiento
Comando base:
```bash
python -m src.train --data_dir data/asl-fingerspelling
```

Con W&B:
```bash
python -m src.train --data_dir data/asl-fingerspelling --use_wandb --wandb_project fingerspelling_asl
```

Flags útiles:
- `--wandb_entity <usuario_o_team>`
- `--wandb_run_name <nombre_run>`
- `--wandb_mode offline`
- `--wandb_tags tag1,tag2`

## 5) Inferencia rápida de checkpoint
```bash
python -m src.quick_infer --ckpt artifacts/models/<checkpoint>.pt --n 16
```

## 6) Webcam (MediaPipe)
```bash
python -m src.realtime_webcam
```

Nota: `src/realtime_webcam.py` usa `artifacts/models/hand_landmarker.task` para detección de manos.

## 7) Modelo entrenado (ejemplo)
Checkpoint best:
```text
artifacts/models/archcmp2_tcn_bilstm_full_20260303_best.pt
```

Copiar desde GCP a local:
```powershell
gcloud compute scp --zone=us-central1-b --project=buoyant-purpose-479417-t8 `
  instance-fingerspeling:~/fingerspelling_asl/artifacts/models/archcmp2_tcn_bilstm_full_20260303_best.pt `
  .\artifacts\models\
```
